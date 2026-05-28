"""
Training and Testing with Single-Way Stop-signed Intersection + Reward Machine

This script integrates the stop-sign reward machine with
the default intersection environment.
"""

# import sys
# import os

# # Add local HighwayEnv to path BEFORE importing
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HighwayEnv'))

import gymnasium as gym
from gymnasium import spaces
import highway_env  # not using local HighwayEnv/highway_env but registers highway-v0 with Gymnasium
from stable_baselines3 import PPO, DQN
import numpy as np
import math
from train_callback import TrainStatsCallback
from highway_env import utils
#from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
#from highway_env.vehicle.realistic import RealisticVehicle, IntersectionState
from reward_machine.reward_machine import RewardMachine

from matplotlib import pyplot as plt
import imageio
import logging
from tqdm import trange
import cv2


def draw_stop_line(frame, env, stop_line_x, ego_vehicle):
    """
    Draws a vertical stop line on the highway relative to the ego vehicle.
    """
    if ego_vehicle is None: return frame

    # --- FIX: FORCE CONTIGUOUS MEMORY LAYOUT ---
    # This fixes the "Layout of the output array img is incompatible with cv::Mat" error
    frame = np.ascontiguousarray(frame, dtype=np.uint8) 
    # -------------------------------------------
    
    # 1. Get Rendering Config
    config = env.unwrapped.config
    W = config.get("screen_width", 600)
    H = config.get("screen_height", 150) 
    scaling = config.get("scaling", 5.5)
    
    # 2. Get Camera Centering 
    center_ratio_x = config.get("centering_position", [0.3, 0.5])[0]
    
    # 3. Calculate Relative Position 
    ego_x = ego_vehicle.position[0]
    rel_x = stop_line_x - ego_x
    
    pixel_x = int((center_ratio_x * W) + (rel_x * scaling))
    
    # 4. Draw Logic 
    if 0 <= pixel_x <= W:
        pt1 = (pixel_x, 0)
        pt2 = (pixel_x, H)
        cv2.line(frame, pt1, pt2, (255, 0, 0), 3) 
        
        cv2.putText(frame, "STOP SIGN", (pixel_x + 5, H - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
    return frame


class HighwayStopEnvRM(gym.Wrapper):
    """
    Product MDP with Reward Machine Wrapping for highway-v0 to enforce a Stop-and-Go task at X = Start + 60m.

    Enforces: Approach -> Stop (Speed=0 at Line) -> Cross -> Goal
    """
    def __init__(self, env, rm, delta_t=None, a_max=6.0, a_min=-6.0, a_min_c=-5.0):
        super().__init__(env)
        self.rm = rm
        # Count only non-terminal states (exclude -1) for observation encoding
        self.num_rm_states = len([s for s in rm.get_states() if s >= 0])
        
        # State Tracking
        self.stop_line_x = None
        self.approach_lane_index = None

        # Task Config
        self.STOP_DIST_OFFSET = 150.0 # Virtual line is 150m ahead of start (was 100m)
        self.STOP_SPEED_THRESH = 0.1  # m/s - relaxed to make stopping easier to detect
        # Distance from center where the stop line is located
        self.STOP_ZONE_START = 0.0
        self.STOP_ZONE_END = 20.0  # Stop zone: within 10m of stop line
        self.GOAL = 50.0
        self.STOP_REGISTRATION_DISTANCE = 5
        self.REACT_TO_BRAKE = self.STOP_ZONE_END  # Trigger forced stop at 50m (gives time to brake from 20 m/s)

        # Get timestep for time calculations
        self.dt = 1.0 / env.unwrapped.config.get("simulation_frequency", 15)
        self.kt = 1.0 / env.unwrapped.config.get("policy_frequency", 5)
        self.delta_t = self.kt if not delta_t else delta_t
        self.a_max = a_max
        self.a_min = a_min
        self.a_min_conf = a_min_c
        self.r_a = 0.4
        self.v_max = 30#env.unwrapped.config.get("reward_speed_range")[1]
        self.v_min = 0#env.unwrapped.config.get("reward_speed_range")[0]
        self.prev_u_id = 0
        self.gamma = 0.99
        self.prev_dist_s = self.STOP_DIST_OFFSET
        self.STOP_REGISTRATION_DISTANCE = 5
        self.finished_stopped = False
        self.ego_stopped_time = 0.0

        assert isinstance(env.observation_space, gym.spaces.Box)

        # CRITICAL: The env observation_space might not match actual observations
        # due to vehicle spawning vs observation config mismatch.
        # We need to check the ACTUAL observation size by doing a reset.
        actual_obs, _ = env.reset()
        actual_obs_size = actual_obs.flatten().shape[0]

        # Initialize stop_line_x from this internal reset so it is never None.
        # reset() will re-set it correctly at the start of every episode.
        for _v in env.unwrapped.road.vehicles:
            if isinstance(_v, ControlledVehicle):
                self.stop_line_x = _v.position[0] + self.STOP_DIST_OFFSET
                break

        # The observation space is an augmented space including the ACTUAL obs size and RM states
        low  = env.observation_space.low.flatten()[0] if len(env.observation_space.low.flatten()) > 0 else -np.inf
        high = env.observation_space.high.flatten()[0] if len(env.observation_space.high.flatten()) > 0 else np.inf

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(actual_obs_size + 1 + self.num_rm_states,), #(?)
            dtype=np.float32)

        # Computing one-hot encodings for the non-terminal RM states
        # Terminal state -1 is excluded, only encode states 0, 1, 2, 3, 4
        self.rm_state_features = {}
        non_terminal_states = sorted([s for s in rm.get_states() if s >= 0])
        state_to_index = {u_id: i for i, u_id in enumerate(non_terminal_states)}

        for u_id in non_terminal_states:
            self.rm_state_features[u_id] = np.eye(self.num_rm_states)[state_to_index[u_id]]

        # Terminal state feature is all zeros
        self.rm_done_feat = np.zeros(self.num_rm_states)

    def reset(self, **kwargs):
        # Reseting the environment and selecting the next RM tasks
        assert isinstance(self.observation_space, spaces.Box)

        self.obs, info = self.env.reset(**kwargs)
        self.current_u_id = self.rm.reset()

        # Reset ego tracking variables
        self.ego_stopped_time = 0.0
        self.prev_stopped = False
        self.finished_stopped = False
        self.prev_speed = 0.0  # Initialize previous speed for reward shaping

        # Flag to enable/disable logging for current episode (set externally)
        self.enable_episode_logging = getattr(self, 'enable_episode_logging', True)

        # Initialize log buffer for this episode (used when logging is disabled initially)
        self.log_buffer = []

        # DEBUG: Log all vehicles to understand spawning
        controlled_count = 0
        for i, vehicle in enumerate(self.env.unwrapped.road.vehicles):
            # vehicle_type = type(vehicle).__name__
            # logging.info(f"Vehicle {i}: Type={vehicle_type}, X={vehicle.position[0]:.1f}, Speed={vehicle.speed:.1f}")

            if isinstance(vehicle, ControlledVehicle):
                controlled_count += 1
                # FORCE LOWER SPEED for training stability - start slow to make braking easier to learn
                vehicle.speed = 20.0  # Start at 10 m/s (was 20 m/s)

                # Only set stop line for THE FIRST controlled vehicle
                if controlled_count == 1:
                    self.stop_line_x = vehicle.position[0] + self.STOP_DIST_OFFSET
                    self.approach_lane_index = vehicle.lane_index
                    self.prev_dist_s = self.STOP_DIST_OFFSET
                    if self.enable_episode_logging:
                        logging.info(f"EPISODE START: Ego X={vehicle.position[0]:.1f}, STOP LINE X={self.stop_line_x:.1f}")

        # Adding the RM state to the observation
        return self.get_observation(self.obs, self.current_u_id, False, self.STOP_DIST_OFFSET), info

    def step(self, action):
        # executing the action in the environment
        next_obs, _, env_done, truncated, info = self.env.step(action)

        # getting the output of the detectors and saving information
        phys_state = self.st4obs(next_obs)
        true_props = self.L(phys_state, info)

        # update the RM state
        # CRITICAL: Convert true_props to a frozenset for proper DNF evaluation
        # The evaluate_dnf function expects a set/collection, not a string
        # Use frozenset (not set) because it's hashable and can be used as dict key
        true_props_set = frozenset([true_props]) if true_props else frozenset()
        self.current_u_id, rm_rew, rm_done = self.rm.step(self.current_u_id, true_props_set, info)
        #rm_rew_before = rm_rew
        #print(f"u_state: {self.current_u_id}; rm_new: {rm_rew_before}")

        # --- REWARD SHAPING ---
        dist_s = phys_state["dist_to_stop_line"]
        v_e = phys_state["v_e"]
        d_inq = phys_state['d_i']    # unified obstacle/stop-line distance
        d_rss = phys_state['d_rss_safe']   # RSS minimum required following distance
        
        if self.current_u_id == 0 and (true_props in ['AB']): #true_props == ('AB' or 'C' or 'D'):
            err_AB = d_inq - d_rss           # ≥ 0 by phi_AB invariant; no clamp
            rm_rew = max(-1.0, 1.0 - err_AB / 2.0)
            c_prog = 0.5
            F_prog = self.gamma * (-c_prog * dist_s) - (-c_prog * self.prev_dist_s)
            rm_rew += (dist_s >= self.STOP_REGISTRATION_DISTANCE)*F_prog
        elif self.current_u_id == 1:
            if true_props in ['0']:
                violation = d_rss - d_inq
                rm_rew = -2.0 * violation / 10.0

        elif self.current_u_id == 3:
            if not (true_props in ['x', 'p']):
                rm_rew = rm_rew * v_e #(phys_state['v_e'] - self.v_min) / (self.v_max - self.v_min)
            #print(f"  rm_new: {rm_rew_before} -> {rm_rew}")
        # State 0 (Approach): Encourage BRAKING when approaching stop line
        # if self.current_u_id == 0 and self.prev_u_id == 0:
        #     dist_stop = phys_state['dist_to_stop_line']
        #     dist_car = phys_state['d_ego_to_front']
        #     speed = phys_state['v_e']

        #     target_v_stop = max(0.0, dist_stop / 5.0)
        #     target_v_car = max(0.0, (dist_car - 5.0) / 2.0)
        #     target_speed = min(target_v_stop, target_v_car)
        #     speed_error = abs(speed - target_speed)
        #     rm_rew -= speed_error * 0.1

            #  dist = min(phys_state['dist_to_stop_line'], phys_state['d_ego_to_front'])
            # speed = phys_state['speed']
            # prev_speed = self.prev_speed if hasattr(self, 'prev_speed') else speed

            # if 0.0 < dist < 20.0:
            #     rm_rew += (1.0 - (dist / 20.0)) * 0.05

            # # CRITICAL: Reward ANY speed reduction to encourage exploration of braking
            # speed_change = speed - prev_speed
            # if speed_change < -0.5:  # If agent slowed down by > 0.5 m/s
            #     rm_rew += 0.3  # Immediate reward for decelerating

            # # Multi-level reward structure:
            # # 1. Penalize high speed when approaching (within 50m)
            # if 0.0 < dist < 40.0:
            #     # Target speed should decrease linearly with distance
            #     # At 50m: target = 20 m/s, At 0m: target = 0 m/s
            #     target_speed = (dist / 40.0) * 20.0

            #     if speed > target_speed:
            #         # STRONG penalty for exceeding target speed
            #         speed_excess = speed - target_speed
            #         rm_rew -= speed_excess * 0.15  # Increased penalty

            # # 2. BIG reward for being very slow near the line
            # if 0.0 < dist < 10.0:
            #     if speed < 5.0:
            #         # Massive reward for slowing down close to line
            #         rm_rew += 0.8  # Increased reward
            #     elif speed < 10.0:
            #         rm_rew += 0.4  # Increased reward

            # # 3. HUGE bonus for actually stopping in the zone
            # if 0.0 < dist < 8.0 and speed < 1.0:
            #     rm_rew += 2.0  # Doubled the stopping reward

            # # Store speed for next step
            # self.prev_speed = speed

            # if 0.0 < dist < 50.0:
            #     # Desired speed decreases with distance
            #     # At 50m: desired = 10 m/s, At 0m: desired = 0 m/s
            #     desired_speed = max(0, (dist / 50.0) * 10.0)

            #     # Strong penalty for excess speed when close
            #     if speed > desired_speed + 2.0:
            #         rm_rew -= (speed - desired_speed) * 0.05  # Increased penalty
            #     elif speed <= desired_speed:
            #         # Reward for being at or below desired speed
            #         rm_rew += 0.02  # Increased reward

            #     # Extra reward for very low speed when very close
            #     if dist < 25.0 and speed < 2.0:
            #         rm_rew += 0.1  # Big reward for slowing down near line

        # State 1 (Proceeding): Encourage Speed AFTER stopping
        # We want the agent to accelerate through after completing stop
        #elif self.current_u_id == 1:
        # if self.current_u_id == 1:
        #     dist_car = phys_state['d_e2f']
        #     speed = phys_state['v_e']
            
        #     target_v_car = max(0.0, (dist_car - 5.0) / 2.0)
        #     target_speed = min(20.0, target_v_car)

        #     if speed < target_speed:
        #         # Reward for speeding up to the limit
        #         rm_rew += speed * 0.1
        #     else:
        #         # Penalize for tailgating / unsafe speed
        #         excess = speed - target_speed
        #         rm_rew -= excess * 0.2

            # Reward proportional to speed, but less aggressive
            # rm_rew += speed * 0.05

        self.prev_u_id = self.current_u_id
        self.prev_speed = phys_state["v_e"]

        # returning the result of this action
        done = rm_done or env_done

        # DEBUG: Log termination reasons
        if self.enable_episode_logging and (env_done or rm_done):
            termination_reason = []
            if env_done:
                termination_reason.append(f"env_done (crashed={info.get('crashed', False)}, off_road={info.get('off_road', False)})")
            if rm_done:
                termination_reason.append(f"rm_done (evt={true_props}, state={self.current_u_id})")
            logging.info(f"EPISODE TERMINATED: {', '.join(termination_reason)}")

        rm_obs = self.get_observation(next_obs, self.current_u_id, done, phys_state['dist_to_stop_line'])
        self.prev_dist_s = phys_state['dist_to_stop_line']

        # Expose the event so the testing loop does not need to re-invoke L(),
        # which would corrupt stateful flags like ego_stopped_time.
        info['event'] = true_props

        # Debug Logging - buffer log messages for potential writing
        log_message = f"T={self.env.unwrapped.time:.1f} | X: {phys_state['x_e']:.1f} | d_e2s: {phys_state['dist_to_stop_line']:.1f} | " \
                     f"d_e2f: {phys_state['d_ego_to_front']:.1f} | d_i: {phys_state['d_i']:.1f} | d_rss: {phys_state['d_rss_safe']:.1f} | v_i: {phys_state['v_i']} | v_f: {phys_state['v_front']:.2f} | v_e: {phys_state['v_e']:.2f} | Evt: {true_props} | RM State: {self.current_u_id} | r: {rm_rew:.2f} | env_done={env_done}"

        # If logging is enabled for this episode, write immediately
        if self.enable_episode_logging:
            logging.info(log_message)
        # Otherwise, buffer it for potential later writing
        elif hasattr(self, 'log_buffer'):
            self.log_buffer.append(log_message)

        return rm_obs, rm_rew, done, truncated, info

    def get_observation(self, next_obs, u_id, done, dist_val=0.0):
        # next_obs: shape (vehicles_count, features)
        # rm_feat: shape (num_rm_states,)


        dist_feat = np.array([dist_val/self.STOP_DIST_OFFSET], dtype=np.float32) #/150.0
        #dist_feat = dist_val

        # Safety check for u_id
        if u_id in self.rm_state_features:
            feat = self.rm_state_features[u_id]
        else:
            # Fallback to prevent crash if new state appears
            feat = np.zeros(self.num_rm_states)
        

        # Flatten obs
        rm_feat = self.rm_done_feat if done else feat
        aug_obs = np.concatenate([next_obs.flatten(), dist_feat, rm_feat], axis=0)

        return aug_obs

    def st4obs(self, obs):
        """
        Extract continuous state from intersection observations.

        Returns a dictionary with:
        - ego: dict with ego vehicle state (position, speed, dist_to_center, heading, etc.)
        - others: list of dicts with other vehicle states
        """
        ego_state = None
        d_e2f = 150  # Default: max sensing range
        v_s = None
        sv_i = None
        ego_vehicle = None
        dist_to_stop_line = 0.0
        d_i = 150
        v_i = 0.0
        a_min_f = self.a_min

        # FIRST PASS: Find ego vehicle and extract its state
        for vehicle in self.env.unwrapped.road.vehicles:
            if isinstance(vehicle, ControlledVehicle):
                ego_vehicle = vehicle
                ego_state = {
                    'position': vehicle.position,
                    'speed': vehicle.speed,
                    'heading': vehicle.heading,
                    'vehicle_obj': vehicle
                }
                #print(f"Ego lane_index: {vehicle.lane_index}")

                dist_to_stop_line = self.stop_line_x - vehicle.position[0]
                break  # Found ego, exit first pass

        # SECOND PASS: Find closest vehicle in front in the same lane
        if ego_vehicle is not None:
            ego_x = ego_vehicle.position[0]
            ego_lane = ego_vehicle.lane_index[2]

            for i, vehicle in enumerate(self.env.unwrapped.road.vehicles):
                # Skip ego vehicle itself
                if vehicle is ego_vehicle:
                    continue

                # Check if vehicle is in the same lane
                if vehicle.lane_index[2] == ego_lane:
                    veh_x = vehicle.position[0]
                    distance = veh_x - ego_x

                    # Only consider vehicles IN FRONT (positive distance)
                    if distance > 0 and distance < d_e2f:
                        d_e2f = distance
                        v_s = vehicle.speed
                        sv_i = i

        # Default front vehicle speed to 0 if none found
        if sv_i is None:
            v_s = 0

        approaching = dist_to_stop_line >= self.STOP_REGISTRATION_DISTANCE
        if approaching:
            if d_e2f < dist_to_stop_line:
                d_i = d_e2f
                v_i = v_s
            else:
                d_i = dist_to_stop_line
                v_i = 0.0
        else:
            if dist_to_stop_line >= 0:
                d_i = dist_to_stop_line
                v_i = 0.0
            else:
                d_i = d_e2f
                v_i = v_s

        d_rss_safe = ego_state['speed']*self.delta_t + 0.5*self.a_max*self.delta_t*self.delta_t + (ego_state['speed']+self.a_max*self.delta_t)*(ego_state['speed']+self.a_max*self.delta_t)/(2*abs(self.a_min_conf)) - v_i*v_i/(2*abs(a_min_f))
        return {
            'x_e': ego_state['position'][0] if ego_state else 0,
            'v_e': ego_state['speed'] if ego_state else 0,
            'dist_to_stop_line': dist_to_stop_line,
            'd_ego_to_front': d_e2f,
            'v_front': v_s,
            'sv_i': sv_i,
            'd_i': d_i,
            'v_i': v_i,
            'd_rss_safe': d_rss_safe
        }

    def L(self, state, info):
        """
        Labeling function: maps MDP observations to event propositions.

        Events defined in stop_hwy.txt:
        - x: ego crashed
        - g: ego reached goal
        - s: ego stopped in zone
        - p: ego passed stop sign (line)
        - f: ego should react to braking in order to stop before the stop line

        Returns: event string representing the current state
        """

        # Check for collision (x)
        if info.get('crashed', False): return 'x'

        
        # Extract Physics
        v_e = state['v_e']
        delta_t = self.delta_t
        dist_s = state['dist_to_stop_line'] # >0 approaching, <0 passed
        dist_f = state['d_ego_to_front']
        v_s = state['v_front']
        dist_i = state['d_i']
        v_i = state['v_i']
        a_e = self.a_min
        a_s = self.a_min
        a_e_max = self.a_max
        a_e_min_c = self.a_min_conf
        a_e_min = self.a_min

        A_bool = (dist_f > v_e*delta_t + v_e*v_e/(2*abs(a_e)))
        B_bool = ((v_s/abs(a_s)>delta_t) and (dist_f>(v_e-v_s)*delta_t+v_e*v_e/(2*abs(a_e))-0.5*a_s*delta_t*delta_t))
        C_bool = ((abs(a_e)>abs(a_s)) and (v_e>v_s+a_s*delta_t) and (v_e/abs(a_e)<(v_s+a_s*delta_t)/abs(a_s)) and (dist_f>(v_e-v_s)*delta_t-0.5*a_s*delta_t*delta_t+(v_e-v_s-a_s*delta_t)*(v_e-v_s-a_s*delta_t)/(2*(a_s-a_e))))
        D_bool = (dist_f>v_e*delta_t + v_e*v_e/(2*abs(a_e))-v_s*v_s/(2*abs(a_s)))
        # phi_AB_bool = A_bool or B_bool
        # phi_C_bool = (not phi_AB_bool) and C_bool
        # phi_D_bool = (not phi_AB_bool) and (not C_bool) and D_bool
        #phi_AB_bool = (dist_f > v_e*delta_t + 0.5*a_e_max*delta_t*delta_t + (v_e+a_e_max*delta_t)*(v_e+a_e_max*delta_t)/(2*abs(a_e_min_c)) - v_s*v_s/(2*abs(a_e_min)))
        phi_AB_bool = (dist_i > v_e*delta_t + 0.5*a_e_max*delta_t*delta_t + (v_e+a_e_max*delta_t)*(v_e+a_e_max*delta_t)/(2*abs(a_e_min_c)) - v_i*v_i/(2*abs(a_e_min)))
        
        # Stop Zone Logic
        # Stop Zone: 0 to 10 meters before line
        in_stop_zone = self.STOP_ZONE_START <= dist_s <= self.STOP_ZONE_END
        is_stopped = v_e < self.STOP_SPEED_THRESH

        # --- Prev. to 12/28 (PROBLEMATIC) ---
        # if dist_s < dist_f and dist_s > 0:
        #     if dist_s < self.REACT_TO_BRAKE and (not is_stopped):
        #         return 'f'
        # --- ---

        # --- Mod. 12/28 ---
        if in_stop_zone and is_stopped:
            if not (self.ego_stopped_time > 0):
                self.ego_stopped_time += self.dt
                return 's'
            else:
                return 'r'
        # if dist_s > 0: # (-CHECK-)
        #     if (dist_s < dist_f) and (not is_stopped): # () or dist_s < self.REACT_TO_BRAKE
        #         # If we are close (65m) but NOT in the success state ('s'),
        #         # # we MUST return 'f' to guide the agent, even if speed is 0.
        #         return 'f'
        # --- End of Mod. 12/28 ---

        # Passed Line Logic
        # We passed the end of the lane (dist < 0)
        if dist_s < 0:
            #logging.info(f"*** EVENT 'p' (PASSED): dist={dist:.2f}m, speed={speed:.3f} m/s")
            # --- (?) 0507 ---
            # return 'p'
            # --- (?) 0507 end ---
            if self.prev_dist_s >= 0:
                return 'p'
        
        # --- Prev. to 12/28 (PROBLEMATIC) ---
        # if in_stop_zone and is_stopped:
        #     #logging.info(f"*** EVENT 's' (STOPPED): dist={dist:.2f}m, speed={speed:.3f} m/s, in_zone={in_stop_zone}, stopped={is_stopped}")
        #     return 's'
        # --- ---
        
        # Check for goal (g)
        # highway-env sets 'is_success' to True when vehicle exits map
        #if info.get('is_success', False): return 'g'
        if dist_s <= -(self.GOAL - self.STOP_ZONE_START): return 'g' #state['dist_to_stop_line']

        # if dist_s >= dist_f: #and dist_s > 0:
        if phi_AB_bool:
            return 'AB'
            
        # if phi_C_bool:
        #     return 'C'
            
        # if phi_D_bool:
        #     return 'D'
        
        # if (not phi_AB_bool): #and (not phi_C_bool) and (not phi_D_bool):
        #     return '0'

        return '0'
        # 5. Normal driving (n)
        #return 'n'
    


if __name__ == "__main__":
    # Basic config for training IntersectionEnv with Four-Way Stop
    isPPO = True
    store_gif = False
    pre_train = True  # MUST retrain from scratch after changing reward machine structure
    learn_itrs = 2048*60  # 100k timesteps for PPO (needs more data than DQN)

    # Configure logging - only write to file if NOT pre_training
    # IMPORTANT: Must configure BEFORE creating the environment wrapper
    # Note: Episode-level filtering happens in the testing loop
    if not pre_train:
        logging.basicConfig(
            filename="log_hwy_stopandgo_rm_rss.txt",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s %(message)s"
        )
    else:
        # During pre-training, disable logging output
        logging.basicConfig(
            level=logging.CRITICAL,  # Set to CRITICAL to suppress INFO messages
            format="%(asctime)s %(message)s"
        )
    # logging.basicConfig(
    #     filename="log_hwy_stopandgo_rm.txt",
    #     filemode="w",
    #     level=logging.INFO,
    #     format="%(asctime)s %(message)s"
    # )

    # Create intersection environment
    config = {
        "lanes_count": 1,           # Single Lane
        "vehicles_count": 2,        # 2 IDM vehicle for consistent observation size
        "controlled_vehicles": 1,
        "initial_speed": 20,        # CRITICAL: Start ego at 20 m/s (matches target_speeds middle value)
        "initial_spacing": 1,       # Spacing between vehicles
        "duration": 20,             # Seconds (TOO LONG FOR 40)
        "policy_frequency": 5,      # CRITICAL: 10 Hz gives more control actions (was 1 Hz)
        "simulation_frequency": 15,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,    # CRITICAL: Fixed count ensures consistent observation size (pads with zeros if fewer vehicles)
            "features": ["x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": True        # Need absolute X to correlate with Stop Line X
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,       # Disable lane changes to focus on speed control
            "target_speeds": [0, 15, 30]  # CRITICAL: Limit max speed to 20 m/s to prevent overspeeding
            # SLOWER=0 m/s, IDLE=10 m/s, FASTER=20 m/s
        },
        "reward_speed_range": [0, 30], # default
        "collision_reward": -1,
    }
    env = gym.make("highway-v0", render_mode='rgb_array', config=config)
    #print(env.unwrapped.config.get(["action"]))

    # CRITICAL: Reset environment to apply observation config BEFORE wrapping
    # The observation_space is only updated after the first reset
    obs, info = env.reset()

    # Load FCFS reward machine
    rm = RewardMachine("rm_examples/stop_fcfs_rss_3.txt") #stop_hwy_rss

    # Wrap environment with reward machine
    hwy_env_rm = HighwayStopEnvRM(env, rm)

    # DEBUG: Print observation space info
    print("=" * 60)
    print("ENVIRONMENT OBSERVATION SPACE DEBUG:")
    print(f"Base env observation_space: {env.observation_space}")
    print(f"Wrapped env observation_space: {hwy_env_rm.observation_space}")

    # Check what the base env returns
    obs_base, _ = env.reset()
    print(f"Base env actual obs shape: {obs_base.shape}")
    print(f"Base env actual obs flattened: {obs_base.flatten().shape[0]}")

    obs_test, _ = hwy_env_rm.reset()
    print(f"Actual wrapped observation shape: {obs_test.shape}")
    print(f"Expected: {hwy_env_rm.observation_space.shape}")
    print(f"Match: {obs_test.shape == hwy_env_rm.observation_space.shape}")
    print("=" * 60)

    #obs, info = custom_reset_with_fixed_vehicles()
    done = False

    if pre_train:
        if isPPO:
            model_dir = "ppo_hwy_sg_rm_rss_" + str(int(learn_itrs))
            callback = TrainStatsCallback()

            # PPO Hyperparameters optimized for stop-and-go task:
            # - Higher n_steps for better long-horizon credit assignment
            # - Larger batch_size for stable policy updates
            # - Higher gamma for long-term rewards (stop -> go sequence)
            # - Entropy bonus for initial exploration
            # - Multiple epochs for better sample efficiency
            model = PPO("MlpPolicy", hwy_env_rm, verbose=1, device="cpu",
                        learning_rate=3e-4,        # Stable learning rate for PPO
                        n_steps=2048,              # Collect more steps before update (more experience per batch)
                        batch_size=512,            # Larger minibatches for stability (must divide n_steps evenly)
                        n_epochs=10,               # More gradient updates per batch (default is 10)
                        gamma=0.99,                # Higher discount for long-horizon task (stop->proceed sequence) (0.995?)
                        gae_lambda=0.95,           # GAE for advantage estimation
                        clip_range=0.2,            # Standard PPO clipping (default 0.2)
                        ent_coef=0.01,             # Entropy bonus for exploration (default 0.0, we boost it)
                        vf_coef=0.5,               # Value function loss coefficient
                        max_grad_norm=0.5,         # Gradient clipping for stability
                        tensorboard_log="./ppo_hwy_tensorboard/")
            model.learn(total_timesteps=int(learn_itrs), callback=callback)
            model.save(model_dir)
        else:
            model_dir = "dqn_hwy_sg_rm_rss_" + str(int(learn_itrs))
            model = DQN('MlpPolicy', hwy_env_rm,
                       policy_kwargs=dict(net_arch=[256, 256]),
                       learning_rate=5e-4,
                       buffer_size=30000,  # Increased buffer
                       learning_starts=1000,  # More initial exploration
                       batch_size=64,  # Larger batch for stability
                       gamma=0.99,  # Higher discount for long-term rewards
                       train_freq=1,
                       gradient_steps=1,
                       target_update_interval=100,
                       exploration_fraction=0.5,  # Explore for 50% of training
                       exploration_initial_eps=1.0,  # Start with full exploration
                       exploration_final_eps=0.05,  # End with 5% exploration
                       verbose=1,
                       device="cpu",
                       tensorboard_log="highway_dqn/")
            callback = TrainStatsCallback()
            model.learn(total_timesteps=int(learn_itrs), callback=callback)
            model.save(model_dir)
    else:
        if isPPO:
            model_dir = "ppo_hwy_sg_rm_rss_" + str(int(learn_itrs))
        else:
            model_dir = "dqn_hwy_sg_rm_rss_" + str(int(learn_itrs))

    # ----- Testing -----
    if isPPO:
        model = PPO.load(model_dir)
    else:
        model = DQN.load(model_dir)

    if store_gif:
        frames = []

    if store_gif:
        n_total = 5
    else:
        n_total = 100

    n_collision = 0
    n_goal = 0
    n_passed = 0
    n_stopped = 0
    sum_dist_s = 0.0

    # Testing trained IntersectionEnv with RM
    pbar = trange(n_total, desc='Episodes')
    #env.unwrapped.config["vehicles_count"] = 1
    #test_hwy_env_rm = HighwayStopEnvRM(env, rm)
    for episode in range(n_total):
        done = truncated = False

        # Episode-level tracking flags
        episode_collision = False
        episode_passed_without_stop = False
        episode_correct_stop = False
        episode_stopped = False  # Track if agent properly stopped before line
        episode_goal = False
        episode_passed = False
        #count_once = False
        stop_counter = False

        # Temporarily disable logging - we'll enable it after checking if episode is a failure
        hwy_env_rm.enable_episode_logging = False

        # IMPORTANT: Reset using the wrapped env, not the base env
        obs, info = hwy_env_rm.reset() #env.reset() #custom_reset_with_fixed_vehicles()

        t_i = 0
        episode_reward = 0.0
        curr_d_stop = 0.0

        while not (done or truncated):

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = hwy_env_rm.step(action)
            #obs, reward, done, truncated, info = test_hwy_env_rm.step(action)

            episode_reward += reward

            # Get current physical state for event detection.
            # Use info['event'] set by step() — do NOT call L() again here,
            # as a second call would corrupt ego_stopped_time and miss 's'.
            phys_state = hwy_env_rm.st4obs(obs)
            current_event = info.get('event', '')

            # Track collision events
            if current_event == 'x' or info.get('crashed', False):
                episode_collision = True

            # Track if agent stopped properly in the stop zone before the line
            if current_event == 's' and not episode_stopped: #stop_counter:
                episode_stopped = True
                n_stopped = n_stopped + 1
                # curr_d_stop = phys_state['dist_to_stop_line'] #obs[-(hwy_env_rm.num_rm_states + 1)]*hwy_env_rm.STOP_DIST_OFFSET
                #stop_counter = True

            if abs(phys_state['v_e']) < 0.1 and (not stop_counter):
                curr_d_stop = phys_state['dist_to_stop_line']
                stop_counter = True

            # Track if agent passed the stop line without stopping
            # This happens when 'p' event occurs WITHOUT having stopped first
            if current_event == 'p': #and not episode_stopped:
                episode_passed_without_stop = True
                episode_passed = True


            # Track if agent reached the goal
            if phys_state['dist_to_stop_line'] <= -50: #current_event == 'g':
                episode_goal = True

            if store_gif:
                frame = env.render()
                # Draw the stop sign line on the frame
                frame = draw_stop_line(frame, env, hwy_env_rm.stop_line_x, hwy_env_rm.env.unwrapped.vehicle)
                frames.append(frame)

            t_i += 1

        if curr_d_stop is not None:
            sum_dist_s = sum_dist_s + curr_d_stop
        # Determine if this episode is a failure that should be logged
        # Failure conditions: collision, passed without stopping, or didn't complete goal
        is_failure = (episode_collision or
                     (episode_passed_without_stop and not episode_correct_stop) or
                     not (episode_goal and episode_stopped and not episode_collision))
        # is_failure = True

        # If this is a failure episode and we're not pre-training, write buffered logs
        if is_failure and not pre_train and hasattr(hwy_env_rm, 'log_buffer'):
            logging.info(f"=== FAILURE EPISODE {episode+1}/{n_total} ===")
            for log_msg in hwy_env_rm.log_buffer:
                logging.info(log_msg)
            logging.info(f"=== END FAILURE EPISODE {episode+1} ===\n")

        # Update episode counters after episode completes
        if episode_collision:
            n_collision += 1

        if episode_passed and (not episode_stopped): #episode_passed_without_stop and (not episode_correct_stop):
            n_passed += 1

        # Only count as goal completion if agent stopped AND reached goal without collision
        if episode_goal and episode_stopped and not episode_collision:
            n_goal += 1

        #avg_dist_s = sum_dist_s / n_stopped if n_stopped > 0 else float('nan') # Strict avg_ds
        avg_dist_s = sum_dist_s / (episode+1-n_passed) if n_stopped > 0 else float('nan') # Relaxed avg_ds
        pbar.set_postfix({
            'comp': f"{(100*n_goal/(episode+1)):.1f}%",
            'safe': f"{(100*(1-n_collision/(episode+1))):.1f}%",
            'psed': f"{(100*n_passed/(episode+1)):.1f}%",
            # 'd_s': f"{(sum_dist_s/(episode+1-n_passed)):.2f}m"
            'd_s': f"{(avg_dist_s):.2f}m"
        })
        pbar.update(1)

    pbar.close()

    print("\n" + "=" * 60)
    print("RESULTS WITH FCFS REWARD MACHINE:")
    print(f"Completion (M_goal):     {(100*n_goal/n_total):.2f}%")
    print(f"Safety (M_safe):         {(100*(1 - n_collision/n_total)):.2f}%")
    #print(f"Stop Convention (M_stop): {(100*(1 - n_passed/n_total)):.2f}%")
    print(f"Stop Convention (M_stop): {(100*(n_stopped/n_total)):.2f}%")
    #print(f"Avg. d_stop: {(sum_dist_s/(n_total-n_passed)):.2f}m")
    # print(f"Avg. d_stop: {(sum_dist_s/n_stopped if n_stopped > 0 else float('nan')):.2f}m") # Strict avg_ds
    print(f"Avg. d_stop: {(sum_dist_s/(n_total-n_passed) if n_stopped > 0 else float('nan')):.2f}m") # Relaxed avg_ds
    print("-" * 60)
    print(f"Episodes with collision:  {n_collision}/{n_total}")
    print(f"Episodes passed w/o stop: {n_passed}/{n_total}")
    print(f"Episodes reached goal:    {n_goal}/{n_total}")
    print("=" * 60)

    env.close()

    if store_gif and len(frames) > 0:
        imageio.mimsave('stopandgo_rm_rss.gif', frames, fps=10)
        print("GIF saved as 'stopandgo_rm_rss.gif'")
    elif store_gif:
        print("Warning: No frames captured, GIF not saved")
