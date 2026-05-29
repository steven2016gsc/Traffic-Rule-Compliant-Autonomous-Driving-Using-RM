"""
Training and Testing: Single-Way Stop-signed Intersection
  — Reward Machine PPO  (HighwayStopEnvRM)
  — LSTM-PPO Baseline   (HighwayStopEnvLSTM)

HighwayStopEnvRM   : product-MDP wrapper; RM state is appended to the
                     observation and the RM gates reward delivery.
HighwayStopEnvLSTM : POMDP baseline; only raw kinematics + normalised
                     distance are observed.  The hidden env variable
                     `has_stopped` is never exposed, forcing the LSTM
                     hidden state to maintain an implicit belief over
                     task phases (approach → stop → proceed).
                     Trained with RecurrentPPO (sb3_contrib) on recurrent
                     rollout chunks of up to 64 policy steps.

Reference: arXiv 2308.05937
"""

import gymnasium as gym
from gymnasium import spaces
import highway_env  # side-effect: registers highway-v0 with Gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from torch import nn
from train_callback import TrainStatsCallback
from highway_env.vehicle.controller import ControlledVehicle
from reward_machine.reward_machine import RewardMachine

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False

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


class DenseRecurrentFeatureExtractor(BaseFeaturesExtractor):
    """
    Dense feature module used before the recurrent layer.

    SB3-contrib's default MlpLstmPolicy applies ``net_arch`` after the LSTM.
    This extractor implements the manuscript-style ordering
    s_t -> Dense(256) -> ReLU -> Dense(128) -> ReLU -> LSTM(128).
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations.float())


class HighwayStopEnvRM(gym.Wrapper):
    """
    Product MDP with Reward Machine Wrapping for highway-v0 to enforce a Stop-and-Go task at X = Start + 150m.

    Enforces: Approach -> Stop (Speed=0 at Line) -> Cross -> Goal
    """
    def __init__(self, env, rm, delta_t=None, a_max=5.0, a_min=-6.0, a_min_c=-5.0):
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
        self.STOP_ZONE_END = 20.0  # Stop zone: within 20m of stop line
        self.GOAL = 20.0
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
    



# ─────────────────────────────────────────────────────────────────────────────
# LSTM-PPO Baseline Environment
# ─────────────────────────────────────────────────────────────────────────────

class HighwayStopEnvLSTM(gym.Wrapper):
    """
    POMDP environment for the LSTM-PPO baseline (arXiv 2308.05937).

    Designed as the direct methodological foil to HighwayStopEnvRM:

    Observation  — raw kinematics (vehicles_count × features) flattened,
                   plus one scalar: normalised distance-to-stop-line.
                   The RM state one-hot is deliberately omitted so the policy
                   cannot read off the task phase from a privileged symbol;
                   it must maintain an implicit belief via h_t.

    Hidden state — `has_stopped` (bool) lives in the environment object and
                   is never included in the observation.  This creates the
                   POMDP: identical kinematic snapshots s_t can require
                   different actions depending on whether the ego has already
                   completed the stop, a distinction that is unobservable in
                   a single frame.  The LSTM must integrate past observations
                   to resolve this ambiguity.

                  Reward       — mirrors the RM-file scale (stop=+50, goal=+100,
                   collision/passed=−100) but delivered via physical-state
                   conditions rather than automaton-state gates, so the same
                   signal is available without privileged structure:

                   · Approach  (dist_s > 0): RSS shaping + potential-based
                     progress when the RSS gap is safe; RSS violation penalty
                     when the gap is unsafe.
                   · Stop zone & speed≈0 (first occurrence): +50.
                   · Crossed stop line without stopping: −100, episode ends.
                   · Past stop line (dist_s < 0, has_stopped): speed reward
                     proportional to v_e (identical to RM state-3 formula).
                   · Goal (dist_s ≤ −GOAL, has_stopped): +100, episode ends.
                   · Collision: −100, episode ends.

                  Policy       — RecurrentPPO (sb3_contrib) with MlpLstmPolicy:
                     Feature extractor : Dense(256)→ReLU→Dense(128)→ReLU
                     Recurrent layer   : LSTM(128)
                     Heads             : categorical action logits and V(h_t)
                   batch_size=64 caps recurrent training chunks at up to
                   64 policy steps, i.e. about 12.8 s at 5 Hz.
    """

    # ── Shared physical constants (keep in sync with HighwayStopEnvRM) ────────
    STOP_DIST_OFFSET        = 150.0
    STOP_SPEED_THRESH       = 0.1
    STOP_ZONE_START         = 0.0
    STOP_ZONE_END           = 20.0
    GOAL                    = 20.0
    STOP_REGISTRATION_DIST  = 5.0

    # ── Reward scale (mirrors RM file stop_fcfs_rss_3.txt) ────────────────────
    R_STOP      =  50.0
    R_GOAL      = 100.0
    R_COLLISION = -100.0
    R_PASSED    = -100.0

    def __init__(self, env, delta_t=None, a_max=5.0, a_min=-6.0, a_min_c=-5.0):
        super().__init__(env)

        self.a_max      = a_max
        self.a_min      = a_min
        self.a_min_conf = a_min_c
        self.gamma      = 0.99

        self.dt      = 1.0 / env.unwrapped.config.get("simulation_frequency", 15)
        self.kt      = 1.0 / env.unwrapped.config.get("policy_frequency", 5)
        self.delta_t = self.kt if delta_t is None else delta_t

        # Episode state — hidden from the policy (POMDP)
        self.has_stopped    = False
        self.stop_line_x    = None
        self.prev_dist_s    = self.STOP_DIST_OFFSET

        # Discover actual observation dimension via a silent reset
        _obs, _ = env.reset()
        _base_dim = _obs.flatten().shape[0]

        # Observation = kinematic flat + 1 normalised dist scalar
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (_base_dim + 1,),
            dtype = np.float32,
        )

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.has_stopped = False
        self.prev_dist_s = self.STOP_DIST_OFFSET

        for v in self.env.unwrapped.road.vehicles:
            if isinstance(v, ControlledVehicle):
                v.speed = 20.0
                self.stop_line_x = v.position[0] + self.STOP_DIST_OFFSET
                self.prev_dist_s = self.STOP_DIST_OFFSET
                break

        return self._make_obs(obs, self.STOP_DIST_OFFSET), info

    def step(self, action):
        next_obs, _, env_done, truncated, info = self.env.step(action)

        phys   = self._st4obs()
        dist_s = phys["dist_to_stop_line"]
        v_e    = phys["v_e"]
        d_i    = phys["d_i"]
        d_rss  = phys["d_rss_safe"]

        reward  = 0.0
        rm_done = False
        event   = ''   # mirrors info['event'] convention used by HighwayStopEnvRM

        in_zone   = self.STOP_ZONE_START <= dist_s <= self.STOP_ZONE_END
        is_halted = v_e < self.STOP_SPEED_THRESH

        # ── 1. Collision ──────────────────────────────────────────────────────
        if info.get("crashed", False):
            event   = 'x'
            reward  = self.R_COLLISION
            rm_done = True

        # ── 2. Stop-zone event: one-time reward for first full stop ───────────
        elif in_zone and is_halted and not self.has_stopped:
            self.has_stopped = True
            event   = 's'
            reward  = self.R_STOP

        # ── 3. Approach: RSS shaping or RSS violation penalty ─────────────────
        elif dist_s > 0:
            safe_gap = d_i > d_rss
            if safe_gap:
                err_AB = d_i - d_rss
                reward += max(-1.0, 1.0 - err_AB / 2.0)
                c_prog = 0.5
                reward += self.gamma * (-c_prog * dist_s) - (-c_prog * self.prev_dist_s)
            else:
                violation = d_rss - d_i
                reward -= 2.0 * violation / 10.0

        # ── 4. Crossed stop line without having stopped → severe penalty ──────
        elif dist_s < 0 and self.prev_dist_s >= 0 and not self.has_stopped:
            event   = 'p'
            reward  = self.R_PASSED
            rm_done = True

        # ── 5. Past stop line with prior stop: speed reward + goal check ──────
        elif dist_s < 0 and self.has_stopped:
            reward += 2.0 * v_e          # proceed reward (cf. RM state-3 formula)
            if dist_s <= -(self.GOAL - self.STOP_ZONE_START):
                event   = 'g'
                reward += self.R_GOAL
                rm_done = True

        self.prev_dist_s  = dist_s
        info['event']     = event
        info['has_stopped'] = self.has_stopped   # for eval metrics only; not in obs

        done = rm_done or env_done
        return self._make_obs(next_obs, dist_s), reward, done, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_obs(self, base_obs: np.ndarray, dist_s: float) -> np.ndarray:
        """Concatenate flat kinematics with normalised dist-to-stop-line."""
        dist_feat = np.array([dist_s / self.STOP_DIST_OFFSET], dtype=np.float32)
        return np.concatenate([base_obs.flatten(), dist_feat])

    def _st4obs(self) -> dict:
        """Extract continuous physical state from the live simulator.

        Vehicle state is read directly from road.vehicles; no observation
        array is needed because the simulator always holds the ground-truth
        positions and speeds.
        """
        ego_vehicle = None
        d_e2f       = 150.0
        v_s         = 0.0

        for v in self.env.unwrapped.road.vehicles:
            if isinstance(v, ControlledVehicle):
                ego_vehicle = v
                break

        dist_to_stop_line = (
            self.stop_line_x - ego_vehicle.position[0]
            if ego_vehicle is not None and self.stop_line_x is not None
            else 0.0
        )

        if ego_vehicle is not None:
            ego_x    = ego_vehicle.position[0]
            ego_lane = ego_vehicle.lane_index[2]
            for v in self.env.unwrapped.road.vehicles:
                if v is ego_vehicle:
                    continue
                if v.lane_index[2] == ego_lane:
                    d = v.position[0] - ego_x
                    if 0 < d < d_e2f:
                        d_e2f = d
                        v_s   = v.speed

        approaching = dist_to_stop_line >= self.STOP_REGISTRATION_DIST
        if approaching:
            d_i = d_e2f if d_e2f < dist_to_stop_line else dist_to_stop_line
            v_i = v_s   if d_e2f < dist_to_stop_line else 0.0
        else:
            d_i = dist_to_stop_line if dist_to_stop_line >= 0 else d_e2f
            v_i = 0.0               if dist_to_stop_line >= 0 else v_s

        v_e    = ego_vehicle.speed if ego_vehicle else 0.0
        d_rss  = (
            v_e * self.delta_t
            + 0.5 * self.a_max * self.delta_t ** 2
            + (v_e + self.a_max * self.delta_t) ** 2 / (2 * abs(self.a_min_conf))
            - v_i ** 2 / (2 * abs(self.a_min))
        )
        return {
            "v_e":              v_e,
            "dist_to_stop_line": dist_to_stop_line,
            "d_ego_to_front":   d_e2f,
            "v_front":          v_s,
            "d_i":              d_i,
            "v_i":              v_i,
            "d_rss_safe":       d_rss,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Mode selector ─────────────────────────────────────────────────────────
    # True  → LSTM-PPO baseline  (HighwayStopEnvLSTM + RecurrentPPO)
    # False → RM-PPO             (HighwayStopEnvRM   + standard PPO)
    USE_LSTM  = True
    store_gif = False
    pre_train = True
    learn_itrs = 2048 * 60   # 100 k timesteps

    logging.basicConfig(
        level=logging.CRITICAL if pre_train else logging.INFO,
        filename=None if pre_train else "log_hwy_stopandgo_lstm.txt",
        filemode="w",
        format="%(asctime)s %(message)s",
    )

    # ── Shared environment config ──────────────────────────────────────────────
    # 5 observed vehicle slots × 4 features = 20-dim kinematics.
    # HighwayStopEnvLSTM appends 1 normalised-distance scalar → 21-dim
    # observation for this toy highway-env baseline. Arrival/wait-time vectors
    # from the FCFS intersection manuscript are intentionally absent here.
    # HighwayStopEnvRM appends 1 distance + num_rm_states one-hot on top.
    env_config = {
        "lanes_count": 1,
        "vehicles_count": 2,
        "controlled_vehicles": 1,
        "initial_speed": 20,
        # highway-env samples the front-vehicle headway through vehicle density.
        # density=1 gives roughly 27-33 m in the installed highway-env version,
        # matching the intended [26.2, 34.9] m toy-case range closely.
        "vehicles_density": 1.0,
        "duration": 20,
        "policy_frequency": 5,       # 5 Hz → T=64 steps ≈ 12.8 s of context
        "simulation_frequency": 15,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": [0, 15, 30],   # SLOWER / IDLE / FASTER
        },
        "reward_speed_range": [0, 30],
        "collision_reward": -1,
    }
    env = gym.make("highway-v0", render_mode="rgb_array", config=env_config)
    env.reset()  # apply observation config before wrapping

    # ── Build wrapped environment and model ───────────────────────────────────
    if USE_LSTM:
        if not HAS_RECURRENT_PPO:
            raise ImportError(
                "sb3_contrib is required for the LSTM-PPO baseline.\n"
                "Install with:  pip install sb3-contrib"
            )

        wrapped_env = HighwayStopEnvLSTM(env)
        model_dir   = f"lstm_ppo_hwy_sg_dense_{int(learn_itrs)}"

        print("=" * 60)
        print("LSTM-PPO Baseline  |  HighwayStopEnvLSTM")
        print(f"Observation space: {wrapped_env.observation_space.shape}")
        print("=" * 60)

        if pre_train:
            # DenseRecurrentFeatureExtractor gives the intended pre-LSTM feature
            # module: obs -> Dense(256) -> ReLU -> Dense(128) -> ReLU -> LSTM.
            # With policy_frequency=5, batch_size=64 means up to 12.8 s of
            # policy-step context. The "64 frames = 4.27 s" statement only holds
            # if the recurrent policy is stepped at the 15 Hz simulator rate.
            policy_kwargs = dict(
                features_extractor_class = DenseRecurrentFeatureExtractor,
                features_extractor_kwargs = dict(features_dim=128),
                net_arch        = [],           # direct heads from h_t
                activation_fn   = nn.ReLU,
                lstm_hidden_size = 128,
                n_lstm_layers   = 1,
                enable_critic_lstm = True,      # separate LSTM for V(h_t)
            )
            model = RecurrentPPO(
                "MlpLstmPolicy",
                wrapped_env,
                policy_kwargs  = policy_kwargs,
                learning_rate  = 3e-4,
                n_steps        = 2048,
                batch_size     = 64,            # max recurrent chunk length
                n_epochs       = 10,
                gamma          = 0.99,
                gae_lambda     = 0.95,
                clip_range     = 0.2,
                ent_coef       = 0.01,
                vf_coef        = 0.5,
                max_grad_norm  = 0.5,
                verbose        = 1,
                device         = "cpu",
                tensorboard_log = "./lstm_ppo_hwy_tensorboard/",
            )
            model.learn(total_timesteps=int(learn_itrs))
            model.save(model_dir)

        model = RecurrentPPO.load(model_dir)

    else:
        # ── RM-PPO (original) ─────────────────────────────────────────────────
        rm          = RewardMachine("rm_examples/stop_fcfs_rss_3.txt")
        wrapped_env = HighwayStopEnvRM(env, rm)
        model_dir   = f"ppo_hwy_sg_rm_rss_{int(learn_itrs)}"

        print("=" * 60)
        print("RM-PPO  |  HighwayStopEnvRM")
        print(f"Observation space: {wrapped_env.observation_space.shape}")
        print("=" * 60)

        if pre_train:
            model = PPO(
                "MlpPolicy", wrapped_env,
                learning_rate  = 3e-4,
                n_steps        = 2048,
                batch_size     = 512,
                n_epochs       = 10,
                gamma          = 0.99,
                gae_lambda     = 0.95,
                clip_range     = 0.2,
                ent_coef       = 0.01,
                vf_coef        = 0.5,
                max_grad_norm  = 0.5,
                verbose        = 1,
                device         = "cpu",
                tensorboard_log = "./ppo_hwy_tensorboard/",
            )
            model.learn(total_timesteps=int(learn_itrs),
                        callback=TrainStatsCallback())
            model.save(model_dir)

        model = PPO.load(model_dir)

    # ── Evaluation (shared logic for both modes) ──────────────────────────────
    n_total     = 5 if store_gif else 100
    n_collision = n_goal = n_passed = n_stopped = 0
    sum_dist_s  = 0.0
    frames      = [] if store_gif else None

    pbar = trange(n_total, desc="Eval")
    for episode in range(n_total):
        episode_collision = episode_stopped = episode_goal = episode_passed = False
        stop_counter      = False
        curr_d_stop       = None
        cor_d_stop = None

        if USE_LSTM:
            wrapped_env.enable_episode_logging = False  # type: ignore[attr-defined]
        else:
            wrapped_env.enable_episode_logging = False

        obs, _     = wrapped_env.reset()
        done       = truncated = False
        lstm_state = None                         # LSTM carries h_t, c_t across steps
        ep_start   = np.ones((1,), dtype=bool)   # signals LSTM to reset at episode start

        while not (done or truncated):
            if USE_LSTM:
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=ep_start, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = wrapped_env.step(action)
            ep_start = np.array([done or truncated])

            event = info.get("event", "")

            if event == "x" or info.get("crashed", False):
                episode_collision = True
            if event == "s" and not episode_stopped:
                episode_stopped = True
                # n_stopped += 1
                # curr_d_stop = phys['dist_to_stop_line'] # Stopping precision for correct stops at the stop sign
                cor_d_stop = phys["dist_to_stop_line"]
            if event == "p":
                episode_passed = True

            # Record distance at first full stop (from physical state)
            if USE_LSTM:
                phys = wrapped_env._st4obs()
            else:
                phys = wrapped_env.st4obs(obs)
            if abs(phys["v_e"]) < 0.1 and not stop_counter:
                curr_d_stop  = phys["dist_to_stop_line"]
                stop_counter = True
            goal_dist = getattr(wrapped_env, "GOAL", 50.0) - getattr(wrapped_env, "STOP_ZONE_START", 0.0)
            if phys["dist_to_stop_line"] <= -goal_dist:
                episode_goal = True

            if frames is not None:
                frame = env.render()
                frame = draw_stop_line(
                    frame, env, wrapped_env.stop_line_x,
                    wrapped_env.env.unwrapped.vehicle)
                frames.append(frame)

        # if curr_d_stop is not None:
        #     sum_dist_s += curr_d_stop
        if episode_collision:
            n_collision += 1
        if episode_stopped:           n_stopped  += 1; sum_dist_s += cor_d_stop
        if episode_passed and not episode_stopped:
            n_passed += 1
        if episode_goal and episode_stopped and not episode_collision:
            n_goal += 1

        avg_dist_s = sum_dist_s / n_stopped if n_stopped > 0 else float('nan') # Strict avg_ds
        # avg_dist_s = sum_dist_s / (episode+1-n_passed) if n_stopped > 0 else float('nan') # Relaxed avg_ds
        pbar.set_postfix({
            "comp": f"{100*n_goal/(episode+1):.1f}%",
            "safe": f"{100*(1-n_collision/(episode+1)):.1f}%",
            "stop": f"{100*n_stopped/(episode+1):.1f}%",
            # "stop": f"{100*(1-n_passed/(episode+1)):.1f}%",
            # "d_s":  f"{sum_dist_s/max(episode+1-n_passed,1):.2f}m",
            "d_s": f"{(avg_dist_s):.2f}m"
        })
        pbar.update(1)

    pbar.close()
    label = "LSTM-PPO BASELINE" if USE_LSTM else "RM-PPO"
    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
    print(f"{'='*60}")
    print(f"M_goal  (completion):    {100*n_goal/n_total:.2f}%")
    print(f"M_safe  (no collision):  {100*(1-n_collision/n_total):.2f}%")
    print(f"M_stop  (stopped first): {100*n_stopped/n_total:.2f}%")
    print(f"Avg. d_stop: {(sum_dist_s/n_stopped if n_stopped > 0 else float('nan')):.2f}m") # Strict avg_ds
    # print(f"Avg. d_stop:             {(sum_dist_s/(n_total-n_passed) if n_stopped > 0 else float('nan')):.2f} m") # Relaxed avg_ds
    print(f"{'-'*60}")
    print(f"Collisions:              {n_collision}/{n_total}")
    print(f"Passed w/o stop:         {n_passed}/{n_total}")
    print(f"Goal completions:        {n_goal}/{n_total}")
    print(f"{'='*60}")

    env.close()

    if frames:
        tag = "lstm" if USE_LSTM else "rm"
        gif_path = f"stopandgo_{tag}.gif"
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"GIF saved → {gif_path}")
