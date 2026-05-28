"""
LSTM-PPO Baseline for the SUMO All-Way Stop FCFS Intersection Task.

Mirrors train_sumo_intersection_fcfs_rm.py but replaces the product-MDP
(AllWayStopEnvRM, RM state one-hot in observation) with a POMDP wrapper
(AllWayStopEnvLSTM, raw kinematics only) trained with RecurrentPPO.

Architecture — 100% identical to HighwayStopEnvLSTM in
    train_hwyenv_stopandgo_lstmppo_single.py:
        Feature extractor : Dense(256) → ReLU → Dense(128) → ReLU
        Recurrent layer   : LSTM(128)
        Actor head        : Linear(128, 3)    [categorical, Discrete(3)]
        Critic head       : Linear(128, 1)    [V(h_t)]
    BPTT truncation horizon T = 64 steps  (batch_size=64 in RecurrentPPO).

Trajectory-dependent rewards (POMDP) — the reward for the same kinematic
snapshot s_t differs based on trajectory history:

    Two trajectories both reaching state s* = (dist_s=−1 m, v_e=5 m/s):

    A  "ran stop sign"     has_stopped=False, prev_dist_s≥0  → R_PASSED = −100
    B  "stopped, cleared"  has_stopped=True, has_priority=True  → +2·v_e = +10

    Same observation, reward differs by +110.  A memoryless MLP cannot
    distinguish A from B; the LSTM h_t encodes whether the ego previously
    halted (v_e≈0 at the stop line) AND whether FCFS priority was established
    (by integrating the history of other-vehicle arrival times in the obs).

Hidden env variables (never in obs, pure POMDP):
    has_stopped      — ego halted (v_e < 0.1 m/s) at least once at the stop line
    ego_stopped_time — accumulated stopped time at stop line (for FCFS)
    ego_arrival_time — simulation time of ego's FCFS registration
    FCFSQueue        — all vehicle arrival/priority/exit tracking

Reference: arXiv 2308.05937
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

# ── local SUMO env imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HighwayEnv'))

from train_sumo_intersection_fcfs_rm import (
    ScenicIntersectionEnv,
    FCFSQueue,
    RewardMachine,
    HAS_SB3,
    HAS_TRACI,
    TrainStatsCallback,
    logger,
)

if HAS_SB3:
    from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
    print("Warning: sb3_contrib not installed. Install with: pip install sb3-contrib")


# =============================================================================
# Pre-LSTM feature extractor — Dense(256,ReLU) → Dense(128,ReLU)
# =============================================================================

class DenseRecurrentFeatureExtractor(BaseFeaturesExtractor):
    """
    Pre-LSTM feature extraction module: obs → Dense(256,ReLU) → Dense(128,ReLU).

    In sb3_contrib's MlpLstmPolicy, net_arch layers are placed AFTER the LSTM
    (the MlpExtractor receives lstm_output_dim as its input dimension).  To
    implement the paper's intended ordering

        obs → Dense(256,ReLU) → Dense(128,ReLU) → LSTM(128) → heads

    the Dense layers must be registered as the features_extractor (which feeds
    the LSTM) and net_arch must be set to [] (no post-LSTM MLP).
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


# =============================================================================
# AllWayStopEnvLSTM — POMDP Wrapper
# =============================================================================

class AllWayStopEnvLSTM(gym.Wrapper):
    """
    POMDP environment for the LSTM-PPO baseline of the SUMO all-way stop task.

    Observation  — raw kinematic features only:
                   (n_vehicles × n_features) = (15 × 8) = 120 features, flattened.
                   The RM state one-hot carried by ScenicIntersectionEnv is
                   stripped.  The LSTM h_t must encode all task-phase information
                   by integrating observations over time.

    Hidden state — the following variables live in the environment object and
                   are NEVER included in the observation:
                     has_stopped      : ego halted at stop line at least once
                     ego_stopped_time : time ego has been stopped at line
                     ego_arrival_time : FCFS registration time
                     FCFSQueue        : full inter-vehicle priority tracking
                   These create the POMDP:  identical kinematic snapshots s_t
                   carry different optimal actions depending on whether the ego
                   has stopped and whether FCFS priority has been established —
                   both facts are unobservable in a single frame.

    Reward scale — mirrors stop_fcfs_rss_3.txt:
                     R_STOP      = +50   (first full stop at stop line)
                     R_GOAL      = +100  (cleared intersection after stop)
                     R_COLLISION = −100  (crash, terminal)
                     R_PASSED    = −100  (crossed stop line without stopping, terminal)
                   Plus continuous shaping from AllWayStopEnvRM:
                     Approach safe (phi_AB):  RSS boundary-proximity + PBRS progress
                     Approach unsafe:         violation-magnitude penalty
                     Proceeding (has_priority): 2·v_e per step

    Policy       — RecurrentPPO (sb3_contrib) with MlpLstmPolicy:
                     net_arch=[256, 128], lstm_hidden_size=128, n_lstm_layers=1
                     batch_size=64 → BPTT truncation horizon T=64 steps
    """

    # ── Physics constants (keep in sync with AllWayStopEnvRM) ─────────────────
    INTERSECTION_CENTER     = np.array([100.0, 100.0])
    STOP_LINE_FROM_CENTER   = 7.2
    EDGE_LENGTH             = 92.8
    EGO_SPAWN_POS           = 10.0
    INITIAL_DIST_TO_STOP    = EDGE_LENGTH - EGO_SPAWN_POS - STOP_LINE_FROM_CENTER   # 75.6 m
    STOP_ZONE_START         = STOP_LINE_FROM_CENTER   # 7.2 m from center
    STOP_ZONE_END           = 52.0
    APPROACH_DIST           = EDGE_LENGTH
    FCFS_REGISTRATION_DIST  = 5.0     # m from stop line
    STOP_SPEED_THRESH       = 0.1     # m/s
    MIN_STOP_DURATION       = 0.1     # s (one SUMO timestep)
    DT                      = 0.1     # s (SUMO step_length)
    KT                      = 0.2     # s (policy frequency → 5 Hz)

    # ── Reward scale (mirrors RM file stop_fcfs_rss_3.txt) ────────────────────
    R_STOP      =  50.0
    R_GOAL      = 100.0
    R_COLLISION = -100.0
    R_PASSED    = -100.0

    def __init__(
        self,
        env: ScenicIntersectionEnv,
        a_max: float = 5.0,
        a_min: float = -5.0,
        a_min_c: float = -5.0,
        a_min_f: float = -4.5,
    ):
        super().__init__(env)

        self.a_max      = a_max
        self.a_min      = a_min
        self.a_min_conf = a_min_c
        self.a_min_f    = a_min_f
        self.gamma      = 0.99
        self.delta_t    = self.KT

        # ── FCFS tracking (hidden from policy) ────────────────────────────────
        self.fcfs_queue          = FCFSQueue()
        self.has_stopped         = False
        self.ego_stopped_time    = 0.0
        self.ego_arrival_time: Optional[float] = None
        self.step_count          = 0
        self.prev_dist_s         = self.INITIAL_DIST_TO_STOP
        self.prev_d_ego_to_front = self.INITIAL_DIST_TO_STOP
        self.prev_d_inquiry      = self.INITIAL_DIST_TO_STOP
        self.prev_leader_id      = None
        self.prev_leader_speed   = 0.0
        self.prev_Q              = False

        # ── Observation space: kinematic obs only (strip RM one-hot) ──────────
        # ScenicIntersectionEnv returns shape (n_vehicles*n_features + n_rm_states,)
        # = (15*8 + 9,) = (129,).  We expose only the kinematic 120 features.
        _raw_obs, _ = env.reset()
        _kin_dim = _raw_obs.flatten().shape[0] - env.n_rm_states   # 120
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_kin_dim,), dtype=np.float32
        )

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset environment and all hidden POMDP state variables.

        Includes the same spawn-collision retry guard as AllWayStopEnvRM.reset():
        SUMO occasionally places a traffic vehicle at the ego's spawn position,
        removing the ego on step 0.  We retry until the ego survives.
        """
        _MAX_RETRIES = 10
        for _attempt in range(_MAX_RETRIES):
            raw_obs, info = self.env.reset(**kwargs)
            if self.env.use_traci and not self.env._ensure_ego_vehicle():
                logger.warning(
                    f"[LSTM] Ego absent after reset (attempt {_attempt+1}/{_MAX_RETRIES})"
                    " — spawn collision, retrying."
                )
                continue
            break

        # Reset ALL hidden POMDP state
        self.fcfs_queue.reset()
        self.has_stopped         = False
        self.ego_stopped_time    = 0.0
        self.ego_arrival_time    = None
        self.step_count          = 0
        self.prev_dist_s         = self.INITIAL_DIST_TO_STOP
        self.prev_d_ego_to_front = self.INITIAL_DIST_TO_STOP
        self.prev_d_inquiry      = self.INITIAL_DIST_TO_STOP
        self.prev_leader_id      = None
        self.prev_leader_speed   = 0.0
        self.prev_Q              = False

        return self._make_obs(raw_obs), info

    def step(self, action: int):
        """Execute action; return kinematic-only obs + trajectory-dependent reward.

        The reward is conditioned on hidden POMDP variables (has_stopped,
        FCFS priority) that are NOT in the observation.  The LSTM h_t must
        maintain an implicit belief over these variables from kinematics history.
        """
        # ── Environment step ──────────────────────────────────────────────────
        self.env.traci.switch(self.env.traci_label)
        raw_obs, _, env_done, truncated, info = self.env.step(action)
        self.step_count += 1

        # ── Physical state ────────────────────────────────────────────────────
        phys      = self.st4obs(raw_obs)
        dist_s    = phys['dist_to_stop_line']
        v_e       = phys['v_e']
        d_inq     = phys['d_inquiry']
        d_rss     = phys['d_rss_safe']
        is_halted = phys['ego_stopped']
        at_line   = phys['ego_at_stop_line']

        # ── Update FCFS tracking (hidden) ─────────────────────────────────────
        self._update_fcfs_queue(info)
        self._update_ego_fcfs_queue(phys, info)

        # ── FCFS priority check (hidden) ──────────────────────────────────────
        # fcfs_queue.has_priority() already guards waiting_time >= MIN_STOP_DURATION
        # internally.  The extra ego_stopped_time guard was resetting to 0 every
        # step when dist_s < 0 (ego past stop line), silently killing the 2*v_e
        # proceed reward in Case 4 even when priority was correctly established.
        ego_in_queue = 'ego' in self.fcfs_queue.arrival_times
        has_priority = (
            ego_in_queue
            and not self.fcfs_queue.must_yield('ego')
        )

        # ── Goal / past-center detection ──────────────────────────────────────
        ego_pos     = np.array([phys['x_e'], phys['y_e']])
        past_center = self._is_past_center(ego_pos, phys['heading_e'])
        try:
            ego_len = self.env.traci.vehicle.getLength(self.env.ego_vehicle_id)
        except Exception:
            ego_len = 5.0
        past_intersection = (
            past_center
            and dist_s < -2 * self.STOP_LINE_FROM_CENTER - ego_len - 20
        )

        # ── phi_AB (RSS safe-braking condition) ───────────────────────────────
        v_i    = phys['v_inquiry']
        phi_AB = (
            d_inq > v_e * self.delta_t
            + 0.5 * self.a_max * self.delta_t ** 2
            + (v_e + self.a_max * self.delta_t) ** 2 / (2 * abs(self.a_min_conf))
            - v_i ** 2 / (2 * abs(self.a_min_f))
        )

        # ── Reward logic (trajectory-dependent) ───────────────────────────────
        reward  = 0.0
        rm_done = False
        event   = ''

        # 1. Collision
        if info.get('crashed', False):
            event   = 'x'
            reward  = self.R_COLLISION
            rm_done = True

        # 2. First full stop in FCFS zone → R_STOP, hidden has_stopped flip.
        #    Top-level case (before approach shaping) so the stop step gives
        #    exactly R_STOP, matching the RM u0→u2 / u1→u2 transition reward
        #    of +50 with no concurrent approach-shaping component.
        #    at_line = (0 ≤ dist_s ≤ FCFS_REGISTRATION_DIST = 5 m), covering
        #    dist_s = 0 (exactly at stop line) which dist_s > 0 would miss.
        elif at_line and is_halted and not self.has_stopped:
            self.has_stopped = True
            event  = 's'
            reward = self.R_STOP

        # 3. Approach phase (dist_s > 0, not the first halt in FCFS zone):
        #    RSS boundary-proximity reward + PBRS progress when safe (RM u0);
        #    RSS violation penalty when unsafe (RM u1).
        elif dist_s > 0:
            if phi_AB:
                # RSS boundary-proximity reward (identical to RM state-0 'AB' formula)
                err_AB  = d_inq - d_rss
                reward += max(-1.0, 1.0 - err_AB / 2.0)
                # PBRS forward-progress (Ng et al. 1999): F = γΦ(s') − Φ(s)
                c_prog  = 0.5
                reward += (
                    self.gamma * (-c_prog * dist_s) - (-c_prog * self.prev_dist_s)
                ) * (dist_s >= self.FCFS_REGISTRATION_DIST)
            else:
                # RSS violation penalty (identical to RM state-1 '0' formula)
                violation = d_rss - d_inq
                reward   += -2.0 * violation / 10.0

        # 4. Passed stop line without prior stop → R_PASSED, terminal
        elif dist_s < 0 and self.prev_dist_s >= 0 and not self.has_stopped:
            event   = 'p'
            reward  = self.R_PASSED
            rm_done = True

        # 5. Past stop line after proper stop: speed reward + goal check
        elif dist_s < 0 and self.has_stopped:
            if has_priority:
                # Proceed reward proportional to speed (identical to RM state-3 formula)
                reward += 2.0 * v_e
            if past_intersection:
                event   = 'g'
                reward += self.R_GOAL
                self.fcfs_queue.vehicle_exited_intersection('ego')
                rm_done = True

        # ── State bookkeeping ─────────────────────────────────────────────────
        self.prev_dist_s = dist_s
        self.prev_Q      = phys['d_ego_to_front'] < dist_s

        info['event']       = event
        info['has_stopped'] = self.has_stopped   # eval metrics only; NOT in obs

        done = rm_done or env_done
        return self._make_obs(raw_obs), reward, done, truncated, info

    def render(self, capture_frame: bool = False):
        return self.env.render(capture_frame=capture_frame)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Strip RM one-hot from base env observation → kinematic features only."""
        flat = raw_obs.flatten()
        return flat[: -self.env.n_rm_states].astype(np.float32)

    def st4obs(self, obs: np.ndarray) -> dict:
        """Extract continuous physical state (verbatim from AllWayStopEnvRM.st4obs)."""
        state = {
            'x_e': 0.0, 'y_e': 0.0, 'v_e': 0.0, 'heading_e': 0.0,
            'dist_to_intersection': self.APPROACH_DIST,
            'dist_to_stop_line':    self.INITIAL_DIST_TO_STOP,
            'd_ego_to_front':       self.INITIAL_DIST_TO_STOP,
            'd_inquiry':            self.INITIAL_DIST_TO_STOP,
            'v_front': 0.0, 'v_inquiry': 0.0,
            'ego_at_stop_line': False,
            'ego_stopped':      False,
            'd_rss_safe':       0.0,
        }
        if not (self.env.use_traci and hasattr(self.env, 'traci')):
            return state
        try:
            ego_id = self.env.ego_vehicle_id
            if not (ego_id and ego_id in self.env.traci.vehicle.getIDList()):
                return state

            pos     = self.env.traci.vehicle.getPosition(ego_id)
            speed   = self.env.traci.vehicle.getSpeed(ego_id)
            angle   = self.env.traci.vehicle.getAngle(ego_id)
            heading = np.radians(90 - angle)

            state.update({'x_e': pos[0], 'y_e': pos[1], 'v_e': speed, 'heading_e': heading})

            center         = self.INTERSECTION_CENTER
            dist_to_center = np.linalg.norm(np.array(pos) - center)
            state['dist_to_intersection'] = dist_to_center

            heading_vec = np.array([np.cos(heading), np.sin(heading)])
            to_center   = center - np.array(pos)
            state['dist_to_stop_line'] = np.dot(to_center, heading_vec) - self.STOP_LINE_FROM_CENTER

            dist_s = state['dist_to_stop_line']
            state['ego_at_stop_line'] = 0 <= dist_s <= self.FCFS_REGISTRATION_DIST
            state['ego_stopped']      = speed < self.STOP_SPEED_THRESH

            approaching = dist_s >= self.FCFS_REGISTRATION_DIST
            leader = self.env.traci.vehicle.getLeader(ego_id, 0)
            if leader is not None:
                leader_id, raw_gap = leader
                state['d_ego_to_front'] = max(0.0, raw_gap)
                state['v_front']        = self.env.traci.vehicle.getSpeed(leader_id)
                if approaching:
                    if raw_gap < dist_s:
                        state['d_inquiry']  = state['d_ego_to_front']
                        state['v_inquiry']  = state['v_front']
                        self.prev_d_inquiry = state['d_ego_to_front']
                    else:
                        state['d_inquiry']  = dist_s
                        state['v_inquiry']  = 0.0
                        self.prev_d_inquiry = dist_s
                else:
                    state['d_inquiry'] = dist_s if dist_s >= 0 else state['d_ego_to_front']
                    state['v_inquiry'] = 0.0    if dist_s >= 0 else state['v_front']
                self.prev_d_ego_to_front = state['d_ego_to_front']
                self.prev_leader_speed   = state['v_front']
                self.prev_leader_id      = leader_id
            else:
                if (
                    self.prev_leader_id is not None and self.prev_Q
                    and self.prev_leader_id in self.env.traci.vehicle.getIDList()
                ):
                    state['d_ego_to_front'] = self.prev_d_ego_to_front
                    state['v_front']        = self.prev_leader_speed
                else:
                    state['d_ego_to_front'] = self.INITIAL_DIST_TO_STOP
                    state['v_front']        = state['v_e']
                    self.prev_leader_id     = None
                    self.prev_leader_speed  = state['v_e']
                if approaching:
                    if state['d_ego_to_front'] < dist_s:
                        state['d_inquiry'] = state['d_ego_to_front']
                        state['v_inquiry'] = state['v_front']
                    else:
                        state['d_inquiry'] = dist_s
                        state['v_inquiry'] = 0.0
                else:
                    state['d_inquiry'] = dist_s if dist_s >= 0 else state['d_ego_to_front']
                    state['v_inquiry'] = 0.0    if dist_s >= 0 else state['v_front']
                self.prev_d_ego_to_front = state['d_ego_to_front']

        except Exception as e:
            logger.debug(f"[LSTM] st4obs error: {e}")

        v_e = state['v_e']
        v_i = state['v_inquiry']
        state['d_rss_safe'] = (
            v_e * self.delta_t
            + 0.5 * self.a_max * self.delta_t ** 2
            + (v_e + self.a_max * self.delta_t) ** 2 / (2 * abs(self.a_min_conf))
            - v_i ** 2 / (2 * abs(self.a_min_f))
        )
        return state

    def _update_fcfs_queue(self, info: dict):
        """Update FCFS queue for other vehicles (verbatim from AllWayStopEnvRM)."""
        if not self.env.use_traci:
            return
        try:
            sim_time = info.get('time', 0.0)
            self.fcfs_queue.simulation_time = sim_time
            for veh_id in self.env.traci.vehicle.getIDList():
                if veh_id == self.env.ego_vehicle_id:
                    continue
                pos     = np.array(self.env.traci.vehicle.getPosition(veh_id))
                speed   = self.env.traci.vehicle.getSpeed(veh_id)
                angle   = self.env.traci.vehicle.getAngle(veh_id)
                heading = np.radians(90 - angle)
                road_id = self.env.traci.vehicle.getRoadID(veh_id)

                dist_to_center    = np.linalg.norm(pos - self.INTERSECTION_CENTER)
                dist_to_stop_line = dist_to_center - self.STOP_LINE_FROM_CENTER
                in_fcfs_zone      = 0 <= dist_to_stop_line <= self.FCFS_REGISTRATION_DIST
                is_stopped        = speed < self.STOP_SPEED_THRESH

                if in_fcfs_zone:
                    if veh_id not in self.fcfs_queue.arrival_times:
                        try:
                            route      = self.env.traci.vehicle.getRoute(veh_id)
                            road_edges = [e for e in route if not e.startswith(':')]
                            incoming   = road_edges[0]  if road_edges else ''
                            outgoing   = road_edges[-1] if len(road_edges) > 1 else ''
                            is_left    = FCFSQueue._is_left_turn_route(incoming, outgoing)
                        except Exception:
                            is_left = False
                        self.fcfs_queue.register_arrival(veh_id, sim_time, pos, heading,
                                                         is_left_turn=is_left)
                    self.fcfs_queue.update_waiting_time(veh_id, is_stopped, self.DT)

                dx       = abs(pos[0] - self.INTERSECTION_CENTER[0])
                dy       = abs(pos[1] - self.INTERSECTION_CENTER[1])
                in_square = dx < self.STOP_ZONE_START and dy < self.STOP_ZONE_START
                if road_id.startswith(':') or in_square:
                    self.fcfs_queue.vehicle_entered_intersection(veh_id)
                if not road_id.startswith(':') and veh_id in self.fcfs_queue.in_intersection:
                    self.fcfs_queue.vehicle_exited_intersection(veh_id)

        except Exception as e:
            logger.debug(f"[LSTM] _update_fcfs_queue error: {e}")

    def _update_ego_fcfs_queue(self, state: dict, info: dict):
        """Update ego FCFS registration (verbatim from AllWayStopEnvRM)."""
        dist_s       = state['dist_to_stop_line']
        is_stopped   = state['ego_stopped']
        in_fcfs_zone = 0 <= dist_s <= self.FCFS_REGISTRATION_DIST

        if in_fcfs_zone:
            if self.ego_arrival_time is None:
                self.ego_arrival_time = info.get('time', 0.0)
                _edges = self.env.ego_route_edges
                self.fcfs_queue.register_arrival(
                    'ego', self.ego_arrival_time,
                    np.array([state['x_e'], state['y_e']]),
                    state['heading_e'],
                    is_left_turn=FCFSQueue._is_left_turn_route(_edges[0], _edges[-1]),
                )
            self.fcfs_queue.update_waiting_time('ego', is_stopped, self.DT)
            if is_stopped:
                self.ego_stopped_time += self.DT
        else:
            self.ego_stopped_time = 0.0

    def _is_past_center(self, pos: np.ndarray, heading: float) -> bool:
        """Check if vehicle has passed intersection center (verbatim from AllWayStopEnvRM)."""
        rel_pos     = pos - self.INTERSECTION_CENTER
        heading_vec = np.array([np.cos(heading), np.sin(heading)])
        return float(np.dot(rel_pos, heading_vec)) > 0


# =============================================================================
# Environment factory
# =============================================================================

def create_lstm_environment(
    use_sumo: bool = True,
    render_mode: str = None,
    traci_label: str = 'lstm_default',
    a_min_f: float = -4.5,
) -> AllWayStopEnvLSTM:
    """Create the LSTM-wrapped SUMO intersection environment.

    The base ScenicIntersectionEnv is identical to the one in create_environment()
    from train_sumo_intersection_fcfs_rm.py; only the wrapper differs.
    """
    base_env = ScenicIntersectionEnv(
        use_sumo=use_sumo,
        use_scenic_only=False,
        render_mode=render_mode,
        traci_label=traci_label,
    )

    # Resolve a_min_f from route file (same logic as create_environment)
    if hasattr(base_env, 'route_file') and os.path.exists(base_env.route_file):
        try:
            _root    = ET.parse(base_env.route_file).getroot()
            _s1_type = next(
                (v.get('type') for v in _root.findall('vehicle') if v.get('id') == 'v_S1'),
                'DEFAULT_VEHTYPE',
            )
            _decel = next(
                (float(t.get('decel', 4.5)) for t in _root.findall('vType')
                 if t.get('id') == _s1_type),
                4.5,
            )
            a_min_f = -_decel
        except Exception as _e:
            logger.warning(f"Could not resolve a_min_f: {_e}")

    return AllWayStopEnvLSTM(
        base_env,
        a_max=5.0, a_min=-5.0, a_min_c=-5.0, a_min_f=a_min_f,
    )


# =============================================================================
# Training
# =============================================================================

def train_lstm(
    total_timesteps: int = 122880,
    use_sumo: bool = True,
    save_path: str = None,
    seed: int = None,
):
    """Train the LSTM-PPO baseline on the SUMO all-way stop intersection.

    Architecture (100% identical to HighwayStopEnvLSTM):
        DenseRecurrentFeatureExtractor: obs → Dense(256,ReLU) → Dense(128,ReLU)
        lstm_hidden_size=128: LSTM(128→128)            (single layer)
        net_arch=[]:          no post-LSTM MLP; heads come directly from h_t
        enable_critic_lstm=True: separate LSTM for V(h_t)
        batch_size=64:        BPTT truncation horizon T=64 steps

    CRITICAL: in sb3_contrib, net_arch layers are inserted AFTER the LSTM.
    Using net_arch=[256,128] would produce obs→LSTM(raw)→Dense(Tanh)→heads,
    feeding raw 120-dim kinematics into the LSTM instead of 128-dim features.
    DenseRecurrentFeatureExtractor + net_arch=[] gives the correct pre-LSTM
    feature extraction ordering.
    """
    if not HAS_RECURRENT_PPO:
        raise ImportError("sb3_contrib required. Install with: pip install sb3-contrib")

    env = create_lstm_environment(use_sumo=use_sumo, traci_label='lstm_train')

    policy_kwargs = dict(
        features_extractor_class  = DenseRecurrentFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = [],          # no post-LSTM MLP; heads from h_t
        activation_fn             = nn.ReLU,    # matches hardcoded ReLU in extractor
        lstm_hidden_size          = 128,
        n_lstm_layers             = 1,
        enable_critic_lstm        = True,        # separate LSTM for V(h_t)
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs   = policy_kwargs,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 64,            # BPTT T=64 steps ≈ 12.8 s at 5 Hz
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        device          = "cpu",
        seed            = seed,
        tensorboard_log = "./lstm_ppo_sumo_tensorboard/",
    )

    if save_path is None:
        save_path = f"lstm_ppo_allway_stop_{total_timesteps}"

    logger.info(f"[LSTM] Training RecurrentPPO for {total_timesteps} timesteps …")
    model.learn(total_timesteps=total_timesteps, callback=TrainStatsCallback(),
                progress_bar=True)
    model.save(save_path)
    logger.info(f"[LSTM] Model saved → {save_path}")
    env.close()
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_lstm(
    model_path: str,
    n_episodes: int = 100,
    use_sumo: bool = True,
    render: bool = False,
):
    """Evaluate a trained LSTM-PPO model.

    The LSTM hidden state (h_t, c_t) is threaded between steps via the
    `state` and `episode_start` arguments of model.predict(), so the
    recurrent belief about task phase (has_stopped, FCFS priority) is
    maintained across the full episode.

    Returns (M_goal, M_safe, M_stop, M_yield, avg_d_stop) as percentages.
    """
    if not HAS_RECURRENT_PPO:
        raise ImportError("sb3_contrib required. Install with: pip install sb3-contrib")

    from tqdm import tqdm

    env   = create_lstm_environment(
        use_sumo=use_sumo,
        render_mode='human' if render else None,
        traci_label='lstm_eval',
    )
    model = RecurrentPPO.load(model_path, device='cpu')

    n_goal = n_col = n_stop = n_yv = n_passed = 0
    sum_d_stop = 0.0

    pbar = tqdm(range(n_episodes), desc="LSTM Eval")
    for episode in pbar:
        obs, _     = env.reset()
        lstm_state = None
        ep_start   = np.ones((1,), dtype=bool)
        done = trunc = False

        ep_col = ep_stopped = ep_goal = ep_passed = ep_entered = ep_yv = False
        stop_counter = False
        curr_d_stop  = None

        while not (done or trunc):
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=ep_start, deterministic=True
            )
            obs, _reward, done, trunc, info = env.step(action)
            ep_start = np.array([done or trunc])

            event = info.get('event', '')
            phys  = env.st4obs(obs)

            if (event == 'x' or info.get('crashed', False)) \
                    and not info.get('connection_lost', False):
                ep_col = True
            if event == 's' and not ep_stopped:
                ep_stopped  = True
                # curr_d_stop = phys['dist_to_stop_line'] # Stopping precision for correct stops at the stop sign
                cor_d_stop = phys['dist_to_stop_line']
            
            if abs(phys['v_e']) < 0.1 and not stop_counter:
                curr_d_stop = phys['dist_to_stop_line'] # Stopping precision for ordinary stops
                stop_counter = True

            if event == 'p':
                ep_passed = True
            if phys['dist_to_stop_line'] < 0 and not ep_entered:
                ep_entered = True
                if not env.fcfs_queue.has_priority('ego'):
                    ep_yv = True
            if event == 'g':
                ep_goal = True
            

        if ep_col:               n_col   += 1
        if ep_stopped:           n_stop  += 1; sum_d_stop += cor_d_stop
        if ep_passed and not ep_stopped: n_passed += 1
        if ep_yv:                n_yv    += 1
        if ep_goal and not ep_yv and ep_stopped and not ep_col:
            n_goal += 1
        # if curr_d_stop is not None:
        #     sum_d_stop += curr_d_stop
        
        avg_ds = sum_d_stop / n_stop if n_stop > 0 else float('nan') # Strict avg_ds
        # avg_ds = sum_d_stop / (episode+1-n_passed) if (n_stop > 0 or (episode+1-n_passed>0)) else float('nan') # Relaxed avg_ds

        pbar.set_postfix({
            'M_goal': f"{(100*n_goal/(episode+1)):.1f}%",
            'M_safe': f"{(100*(1-n_col/(episode+1))):.1f}%",
            # 'M_stop': f"{(100*(1-n_passed/(episode+1))):.1f}%",
            'M_stop': f"{(100*(n_stop/(episode+1))):.1f}%",
            'M_yield': f"{(100*(1-n_yv/(episode+1))):.1f}%",
            'avg_ds':  f"{avg_ds:.2f}m",
        })

    env.close()

    M_goal  = 100.0 * n_goal / n_episodes
    M_safe  = 100.0 * (1 - n_col / n_episodes)
    # M_stop  = 100.0 * (1 - n_passed / n_episodes)
    M_stop  = 100.0 * (1 - n_stop / n_episodes)
    M_yield = 100.0 * (1 - n_yv / n_episodes)
    avg_d   = sum_d_stop / n_stop if n_stop > 0 else float('nan') # Strict avg_ds
    # avg_d = sum_d_stop / (n_episodes - n_passed) if (n_stop > 0 or (n_episodes - n_passed > 0)) else float('nan') # Relaxed avg_ds

    print(f"\n{'='*60}")
    print("LSTM-PPO BASELINE — RESULTS")
    print(f"{'='*60}")
    print(f"M_goal  (goal reached)  : {M_goal:.2f}%")
    print(f"M_safe  (no collision)  : {M_safe:.2f}%")
    print(f"M_stop  (stopped first) : {M_stop:.2f}%")
    print(f"M_yield (FCFS compliant): {M_yield:.2f}%")
    print(f"Avg. d_stop             : {avg_d:.2f} m")
    print(f"{'-'*60}")
    print(f"Collisions              : {n_col}/{n_episodes}")
    print(f"Passed w/o stop         : {n_passed}/{n_episodes}")
    print(f"Goal completions        : {n_goal}/{n_episodes}")
    print(f"{'='*60}")

    return M_goal, M_safe, M_stop, M_yield, avg_d


# =============================================================================
# CLI  (mirrors train_sumo_intersection_fcfs_rm.py argument style)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(
        description="LSTM-PPO Baseline — SUMO All-Way Stop FCFS Intersection"
    )
    parser.add_argument('--mode',       choices=['train', 'eval'], default='train',
                        help="'train' to train; 'eval' to evaluate a saved model")
    parser.add_argument('--timesteps',  type=int,   default=122880,
                        help="Training timesteps (default: 122880)")
    parser.add_argument('--episodes',   type=int,   default=100,
                        help="Evaluation episodes (default: 100)")
    parser.add_argument('--model',      type=str,   default=None,
                        help="Model path for eval (default: auto from --timesteps)")
    parser.add_argument('--sumo',       action='store_true', default=True,
                        help="Use SUMO/TraCI backend (default: True)")
    parser.add_argument('--render',     action='store_true', default=False,
                        help="Render with SUMO-GUI during evaluation")
    parser.add_argument('--seed',       type=int,   default=None,
                        help="Random seed (default: None = random each run)")
    args = parser.parse_args()

    if args.mode == 'train':
        save = args.model or f"lstm_ppo_allway_stop_{args.timesteps}"
        print(f"\n{'='*60}")
        print("LSTM-PPO Training — SUMO All-Way Stop FCFS Intersection")
        print(f"Timesteps  : {args.timesteps}")
        print(f"Save path  : {save}")
        print(f"Seed       : {args.seed if args.seed is not None else 'random'}")
        print(f"HAS_TRACI  : {HAS_TRACI}")
        print(f"{'='*60}\n")
        train_lstm(
            total_timesteps=args.timesteps,
            use_sumo=args.sumo,
            save_path=save,
            seed=args.seed,
        )

    elif args.mode == 'eval':
        model_path = args.model or f"lstm_ppo_allway_stop_{args.timesteps}"
        print(f"\n{'='*60}")
        print("LSTM-PPO Evaluation — SUMO All-Way Stop FCFS Intersection")
        print(f"Model      : {model_path}")
        print(f"Episodes   : {args.episodes}")
        print(f"Render     : {args.render}")
        print(f"{'='*60}\n")
        evaluate_lstm(
            model_path=model_path,
            n_episodes=args.episodes,
            use_sumo=args.sumo,
            render=args.render,
        )
