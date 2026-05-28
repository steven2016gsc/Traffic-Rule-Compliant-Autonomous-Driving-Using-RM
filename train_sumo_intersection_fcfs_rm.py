"""
Training Script: All-Way Stop Intersection with Reward Machine + Scenic/SUMO

This script provides a complete training pipeline for an RL agent learning to
navigate all-way stop intersections using:
- Scenic for scenario generation
- SUMO for traffic simulation with FCFS (First-Come-First-Serve) logic
- Reward Machines for structured reward shaping

The approach follows the Reward Machine framework from:
https://proceedings.mlr.press/v80/icarte18a/icarte18a.pdf

Architecture:
    Scenic Scenario (allway_stop_scenario.scenic)
            |
            v
    SUMO Simulator (intersection.net.xml with allway_stop junction)
            |
            v
    ScenicIntersectionEnv (Gym Wrapper)
            |
            v
    RewardMachineWrapper (RM state tracking + reward shaping)
            |
            v
    RL Algorithm (PPO/DQN from Stable Baselines3)

Author: 
Date: 2025
"""

import os
import sys
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, Set, Tuple, Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add local HighwayEnv to path (fallback for highway-env based training)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HighwayEnv'))

# Import RL libraries
try:
    from stable_baselines3 import PPO, DQN
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable_baselines3 not installed. RL training disabled.")

# Import Scenic (optional - for Scenic-based scenarios)
try:
    import scenic
    # Scenic doesn't have a built-in SUMO simulator
    # Available simulators: carla, newtonian, metadrive, lgsvl, webots, xplane
    # For SUMO integration, we use traci (SUMO's TraCI API) directly
    HAS_SCENIC = True
except ImportError:
    HAS_SCENIC = False
    print("Warning: Scenic not installed. Using HighwayEnv fallback.")

# Import SUMO's TraCI for direct SUMO control (optional)
# SUMO tools path needs to be added if SUMO was built from source
HAS_TRACI = False
if os.environ.get('SUMO_HOME'):
    sumo_tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if os.path.isdir(sumo_tools) and sumo_tools not in sys.path:
        sys.path.insert(0, sumo_tools)

try:
    import traci
    HAS_TRACI = True
except ImportError:
    HAS_TRACI = False
    if not os.environ.get('SUMO_HOME'):
        print("Note: SUMO_HOME not set. Set it to enable SUMO/TraCI simulation.")

# Import local modules
from reward_machine.reward_machine import RewardMachine
from reward_machine.rm_environment import RewardMachineWrapper
from train_callback import TrainStatsCallback

# Configure logging
logging.basicConfig(
    filename="log_scenic_intersection_rm.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Reward Machine States for All-Way Stop (from stop_fcfs.txt)
# =============================================================================
# =============================================================================
# FCFS Priority Queue for All-Way Stop
# =============================================================================
class FCFSQueue:
    """
    SUMO-compliant all-way stop priority queue.

    Implements SUMO's LINKSTATE_ALLWAY_STOP logic from MSLink.cpp:
    - Primary priority: Longest waiting time (accumulated while speed < 0.1 m/s)
    - Tiebreaker: Earliest arrival time when waiting times are equal
    - Vehicles register when entering stop zone (not when stopped)
    - Waiting time accumulates continuously while stopped

    References:
    - MSLink.cpp: blockedByFoe() for all-way stop priority logic
    - MSVehicle.h: Waiting time definition (speed < 0.1 m/s threshold)
    """

    # (incoming_edge, outgoing_edge) pairs that constitute a left turn at a
    # standard 4-way intersection (N/S/E/W).  Used by has_priority() Rule 4.
    _LEFT_TURN_PAIRS: frozenset = frozenset({
        ('edge_south_in', 'edge_west_out'),   # northbound  → left  (west)
        ('edge_north_in', 'edge_east_out'),   # southbound  → left  (east)
        ('edge_east_in',  'edge_south_out'),  # westbound   → left  (south)
        ('edge_west_in',  'edge_north_out'),  # eastbound   → left  (north)
    })

    def __init__(self,
                 intersection_center: np.ndarray = np.array([100.0, 100.0]),
                 stop_line_from_center: float = 7.2,
                 fcfs_registration_distance: float = 5.0):
        self.intersection_center = np.array(intersection_center, dtype=float)
        self.stop_line_from_center = stop_line_from_center
        self.fcfs_registration_distance = fcfs_registration_distance
        self.reset()

    def reset(self):
        """Reset the queue for a new episode."""
        self.arrival_times = {}  # vehicle_id -> {time, position, heading, waiting_time, has_stopped}
        self.cleared_vehicles = set()  # vehicles that have been granted priority
        self.in_intersection = set()  # vehicles currently in intersection
        self.simulation_time = 0.0

    def register_arrival(self, vehicle_id: str, arrival_time: float,
                         position: np.ndarray, heading: float,
                         is_left_turn: bool = False):
        """
        Register a vehicle's arrival at the stop zone.

        Following SUMO's setApproaching() behavior, vehicles register when
        they enter the stop zone, not when they stop.

        Args:
            is_left_turn: True if the vehicle's route constitutes a left turn
                at the intersection.  Used by has_priority() Rule 4 (left-turn
                yields to oncoming traffic).
        """
        if vehicle_id not in self.arrival_times:
            self.arrival_times[vehicle_id] = {
                'time': arrival_time,
                'position': position.copy() if isinstance(position, np.ndarray) else np.array(position),
                'heading': heading,
                'waiting_time': 0.0,  # SUMO's myWaitingTime (accumulated while speed < 0.1)
                'has_stopped': False,
                'is_left_turn': is_left_turn,
            }
            logger.debug(f"FCFS: {vehicle_id} entered stop zone at t={arrival_time:.2f}")

    def update_waiting_time(self, vehicle_id: str, is_stopped: bool, dt: float):
        """
        Update waiting time for a vehicle (SUMO-compliant).

        Following MSVehicle.h: waiting time accumulates when speed < 0.1 m/s
        and resets when speed >= 0.1 m/s. This matches SUMO's updateWaitingTime().

        Args:
            vehicle_id: Vehicle identifier
            is_stopped: True if speed < 0.1 m/s
            dt: Time step duration
        """
        if vehicle_id in self.arrival_times:
            if is_stopped:
                self.arrival_times[vehicle_id]['waiting_time'] += dt
                self.arrival_times[vehicle_id]['has_stopped'] = True
            else:
                # SUMO behavior: waiting time resets when speed >= 0.1 m/s
                # However, for all-way stop, once stopped, keep the waiting time
                # until vehicle clears (to maintain FCFS priority)
                if not self.arrival_times[vehicle_id]['has_stopped']:
                    # Not yet stopped, reset waiting time
                    self.arrival_times[vehicle_id]['waiting_time'] = 0.0

    def vehicle_entered_intersection(self, vehicle_id: str):
        """Mark a vehicle as having entered the intersection."""
        self.in_intersection.add(vehicle_id)
        if vehicle_id in self.arrival_times:
            self.cleared_vehicles.add(vehicle_id)

    def vehicle_exited_intersection(self, vehicle_id: str):
        """Mark a vehicle as having exited the intersection."""
        self.in_intersection.discard(vehicle_id)
        # Remove from queue after exiting
        if vehicle_id in self.arrival_times:
            del self.arrival_times[vehicle_id]
        self.cleared_vehicles.discard(vehicle_id)

    def has_priority(self, vehicle_id: str, min_stop_duration: float = 0.1) -> bool:
        """
        Check if a vehicle has priority to proceed (SUMO-compliant).

        Implements SUMO's MSLink::blockedByFoe() logic for LINKSTATE_ALLWAY_STOP:

        From MSLink.cpp:
            if (waitingTime > avi.waitingTime + actionDelta) {
                return false;  // ego can proceed
            }
            if (waitingTime == (avi.waitingTime + actionDelta)
                && arrivalTime < avi.arrivalTime + actionDelta) {
                return false;  // ego arrived first at equal wait time
            }

        Priority rules:
        1. No other vehicle currently in intersection
        2. Longest waiting time wins (primary)
        3. Earliest arrival time wins if waiting times equal (tiebreaker)
        4. Must have stopped for at least min_stop_duration (default 0.1s = one timestep)

        Args:
            vehicle_id: Vehicle to check priority for
            min_stop_duration: Minimum stop duration (SUMO default: one timestep)

        Returns:
            True if vehicle has priority, False if must yield
        """
        if vehicle_id not in self.arrival_times:
            return False

        vehicle_info = self.arrival_times[vehicle_id]

        # SUMO requirement: Must have stopped for at least one timestep
        if vehicle_info['waiting_time'] < min_stop_duration:
            return False

        ## Condition 1: If intersection is occupied, no one has priority to enter [NOT IN SUMO]
        # if self.in_intersection and vehicle_id not in self.in_intersection:
        #     return False

        my_waiting_time = vehicle_info['waiting_time']
        my_arrival = vehicle_info['time']

        # Simultaneity tolerance: one simulation step (dt = 0.1 s).
        # Differences within this window are treated as simultaneous.
        EPSILON = 0.01 #0.1

        # Condition 2: Check priority against all other waiting vehicles
        for other_id, other_info in self.arrival_times.items():
            if other_id == vehicle_id:
                continue
            if other_id in self.cleared_vehicles:
                continue  # Already cleared, don't compete
            if not other_info['has_stopped']:
                continue  # Not yet stopped, don't compete

            other_waiting_time = other_info['waiting_time']
            other_arrival = other_info['time']

            # Rule 1 (primary): j waited longer than i by more than ε → i must yield.
            # Implements: t_wait,j > t_wait,i + ε
            #if other_waiting_time >= my_waiting_time + EPSILON:
            if other_waiting_time > my_waiting_time:
                return False

            # Rule 2 (secondary): waiting times are within ε (tied); earliest arrival wins.
            # Implements: |t_wait,i − t_wait,j| ≤ ε  ∧  t_arrival,j < t_arrival,i − ε → i yields.
            # Non-strict form "other_arrival + EPSILON <= my_arrival" avoids the dead zone
            # that would arise from "other_arrival < my_arrival − EPSILON" at the exact
            # ε = 0.1 s boundary (i.e., j arriving exactly one simulation step before i).
            if abs(other_waiting_time - my_waiting_time) < EPSILON:
                #if other_arrival + EPSILON <= my_arrival:
                if other_arrival < my_arrival - EPSILON:
                    return False

            # Rules 3 & 4 are tiebreakers applied only when both waiting time
            # AND arrival time are within ε (a true simultaneous-arrival tie).
            fully_tied = (abs(other_waiting_time - my_waiting_time) < EPSILON and
                          abs(other_arrival - my_arrival) < EPSILON)
            if fully_tied:
                my_pos     = vehicle_info['position']
                my_heading = vehicle_info['heading']
                other_pos  = other_info['position']

                # Rule 3 (tertiary): right-of-way.
                # At an uncontrolled intersection the vehicle to the right has
                # priority.  If other is on my right → I must yield.
                if self._is_on_right(my_pos, my_heading, other_pos):
                    return False

                # Rule 4 (quaternary): left-turn yields to oncoming traffic.
                # If I am making a left turn AND other is heading toward me
                # (oncoming) → I must yield.
                if vehicle_info.get('is_left_turn', False):
                    if self._is_oncoming(my_heading, other_info['heading']):
                        return False

        return True

    def _is_on_right(self, my_pos: np.ndarray, my_heading: float,
                     other_pos: np.ndarray) -> bool:
        """Check if other vehicle is on my right AND within the FCFS zone.

        Returns False immediately when the other vehicle is farther than
        fcfs_registration_distance from the stop line — the right-of-way rule
        only applies to vehicles that are already at the intersection.

        Geometry: the right-hand direction of a vehicle with heading θ (standard
        math convention, counter-clockwise from east) is the vector obtained by
        rotating θ by −90°, i.e. [cos(θ−π/2), sin(θ−π/2)].  'Other is on my
        right' iff the projection of (other_pos − my_pos) onto that vector is
        positive.
        """
        # Proximity guard: other must be within the FCFS registration zone.
        other_dist_to_stop = (np.linalg.norm(other_pos - self.intersection_center)
                              - self.stop_line_from_center)
        if other_dist_to_stop > self.fcfs_registration_distance:
            return False

        # Geometric check: is other to my right?
        to_other = other_pos - my_pos
        right_vec = np.array([np.cos(my_heading - np.pi / 2),
                              np.sin(my_heading - np.pi / 2)])
        return float(np.dot(to_other, right_vec)) > 0.0

    def _is_oncoming(self, my_heading: float, other_heading: float,
                     tol: float = np.pi / 4) -> bool:
        """Check if other vehicle is traveling in approximately the opposite direction.

        Computes the absolute angular difference between the two headings and
        returns True when that difference is within `tol` radians of π (180°).
        The default tolerance of π/4 (45°) handles registration-time heading
        noise and slight curvature on approach roads.

        Convention: heading is in standard math radians (counter-clockwise from
        east), matching the `heading = radians(90 − SUMO_angle)` conversion used
        elsewhere in this file.
        """
        diff = abs((my_heading - other_heading + np.pi) % (2 * np.pi) - np.pi)
        return diff > np.pi - tol

    @staticmethod
    def _is_left_turn_route(incoming_edge: str, outgoing_edge: str) -> bool:
        """Return True when the (incoming, outgoing) edge pair is a left turn."""
        return (incoming_edge, outgoing_edge) in FCFSQueue._LEFT_TURN_PAIRS

    def must_yield(self, vehicle_id: str, min_stop_duration: float = 0.1) -> bool:
        """
        Check if a vehicle must yield (opposite of has_priority).
        Returns True if the vehicle must wait ('y' event).
        """
        return not self.has_priority(vehicle_id, min_stop_duration)


# =============================================================================
# Event Detection for Reward Machine
# =============================================================================
class IntersectionEventDetector:
    """
    Labeling function L: maps MDP observations to RM event propositions.

    This implements the SUMO-style FCFS event detection logic.
    """

    def __init__(
        self,
        stop_line_distance: float = 10.0,
        intersection_half_size: float = 8.0,
        stop_speed_threshold: float = 0.1,
        min_stop_duration: float = 1.5,
        approach_distance: float = 30.0
    ):
        """
        Args:
            stop_line_distance: Distance from intersection center to stop line [m]
            intersection_half_size: Half-width of intersection area [m]
            stop_speed_threshold: Speed threshold for "stopped" [m/s]
            min_stop_duration: Minimum time to remain stopped [s]
            approach_distance: Distance at which "approaching" begins [m]
        """
        self.STOP_LINE_DISTANCE = stop_line_distance
        self.INTERSECTION_HALF_SIZE = intersection_half_size
        self.STOP_SPEED_THRESHOLD = stop_speed_threshold
        self.MIN_STOP_DURATION = min_stop_duration
        self.APPROACH_DISTANCE = approach_distance

        # Tracking variables
        self.stopped_time = 0.0
        self.has_stopped_at_line = False
        self.arrival_time = None
        self.waiting_time = 0.0
        self.clearance_granted = False

    def reset(self):
        """Reset detector state for new episode."""
        self.stopped_time = 0.0
        self.has_stopped_at_line = False
        self.arrival_time = None
        self.waiting_time = 0.0
        self.clearance_granted = False

    def label(self, obs: Dict, info: Dict, dt: float = 0.1) -> str:
        """
        Generate event label from observation.

        Args:
            obs: Observation dict with 'ego' and 'others' vehicle states
            info: Additional info from environment (crashed, etc.)
            dt: Timestep for time tracking

        Returns:
            Event label string for RM transition
        """
        events = set()

        ego = obs.get('ego', {})
        others = obs.get('others', [])
        intersection_center = obs.get('intersection_center', np.array([0.0, 0.0]))

        if not ego:
            return '!approaching&!at_stop&!stopped_early'

        # Extract ego state
        ego_pos = np.array(ego.get('position', [0, 0]))
        ego_speed = ego.get('speed', 0.0)
        dist_to_center = np.linalg.norm(ego_pos - intersection_center)

        # === Collision Detection ===
        if info.get('crashed', False) or ego.get('crashed', False):
            return 'collision'

        # === Position-based Events ===
        is_approaching = self.APPROACH_DISTANCE > dist_to_center > self.STOP_LINE_DISTANCE + 2.0
        is_at_stop_zone = abs(dist_to_center - self.STOP_LINE_DISTANCE) < 2.0
        is_in_intersection = dist_to_center < self.INTERSECTION_HALF_SIZE
        is_past_intersection = dist_to_center > self.STOP_LINE_DISTANCE and self._is_past_center(ego)

        # === Speed-based Events ===
        is_stopped = ego_speed < self.STOP_SPEED_THRESHOLD
        is_moving = ego_speed >= self.STOP_SPEED_THRESHOLD

        # Update stopped time tracking
        if is_stopped:
            self.stopped_time += dt
        else:
            self.stopped_time = 0.0

        min_stop_satisfied = self.stopped_time >= self.MIN_STOP_DURATION

        # === FCFS Priority Logic ===
        has_priority = self._check_fcfs_priority(ego, others, dt)

        # === Violation Detection ===
        # Ran stop sign: passed stop line while moving fast
        if dist_to_center < self.STOP_LINE_DISTANCE - 1.0 and not self.has_stopped_at_line and ego_speed > 2.0:
            return 'ran_stop'

        # Stopped too early (before stop zone)
        if is_stopped and dist_to_center > self.STOP_LINE_DISTANCE + 5.0:
            events.add('stopped_early')

        # FCFS violation: entered without priority
        if is_in_intersection and not self.clearance_granted:
            return 'violated'

        # === State-specific Event Generation ===

        # Approaching detection
        if is_approaching and not is_stopped:
            events.add('approaching')

        # At stop line and stopped
        if is_at_stop_zone and is_stopped:
            events.add('at_stop')
            events.add('stopped')
            self.has_stopped_at_line = True

            if self.arrival_time is None:
                self.arrival_time = 0.0
            self.waiting_time += dt

        # Minimum stop duration met
        if min_stop_satisfied and self.has_stopped_at_line:
            events.add('min_stop')
            events.add('stopped')

        # Has FCFS priority (clearance)
        if has_priority and self.has_stopped_at_line and min_stop_satisfied:
            events.add('has_priority')
            self.clearance_granted = True

        # Waiting (stopped at line, no priority yet)
        if self.has_stopped_at_line and not has_priority:
            events.add('waiting')

        # Entered intersection
        if is_in_intersection and self.clearance_granted:
            events.add('enter_int')
            events.add('in_intersection')

        # Exited intersection
        if is_past_intersection and self.clearance_granted:
            events.add('exit_int')

        # Cleared intersection (far past)
        if dist_to_center > self.STOP_LINE_DISTANCE + 10.0 and self._is_past_center(ego):
            events.add('clear_int')

        # Moving (for state transitions)
        if is_moving:
            events.add('moving')

        # Hesitation: has priority but not moving
        if self.clearance_granted and is_stopped and not is_in_intersection:
            if self.waiting_time > 5.0:  # Waited too long after clearance
                events.add('hesitation')

        # Build label string
        return self._events_to_label(events)

    def _check_fcfs_priority(self, ego: Dict, others: list, dt: float) -> bool:
        """
        Check if ego has FCFS priority based on SUMO rules.

        FCFS Rules:
        1. Vehicle with longest waiting time has priority
        2. If waiting times equal, earlier arrival has priority
        3. If arrival times equal, yield to vehicle on right
        """
        ego_waiting = self.waiting_time

        for other in others:
            other_state = other.get('intersection_state', None)

            # Only compare with vehicles that are also waiting
            if other_state not in ['waiting', 'WAITING']:
                continue

            other_waiting = other.get('waiting_time', 0.0)
            other_arrival = other.get('arrival_time', float('inf'))

            # Rule 1: Longer wait wins
            if other_waiting > ego_waiting + 0.1:
                return False

            # Rule 2: Earlier arrival wins (tie-breaker)
            if abs(other_waiting - ego_waiting) < 0.1:
                if self.arrival_time is None:
                    return False
                if other_arrival < self.arrival_time:
                    return False

                # Rule 3: Right-of-way (vehicle on right)
                if abs(other_arrival - (self.arrival_time or 0)) < 0.1:
                    if self._is_on_right(ego, other):
                        return False

        # Check if intersection is clear
        for other in others:
            other_state = other.get('intersection_state', None)
            if other_state in ['entering', 'inside', 'crossing', 'ENTERING', 'INSIDE', 'CROSSING']:
                return False

        return True

    def _is_on_right(self, ego: Dict, other: Dict) -> bool:
        """Check if other vehicle is on ego's right (geometric)."""
        ego_pos = np.array(ego.get('position', [0, 0]))
        ego_heading = ego.get('heading', 0)
        other_pos = np.array(other.get('position', [0, 0]))

        # Vector to other
        to_other = other_pos - ego_pos

        # Right direction (perpendicular to heading)
        right_vec = np.array([np.cos(ego_heading - np.pi/2), np.sin(ego_heading - np.pi/2)])

        # Dot product > 0 means other is on right
        return np.dot(to_other, right_vec) > 0

    def _is_past_center(self, ego: Dict) -> bool:
        """Check if ego has passed the intersection center."""
        ego_pos = np.array(ego.get('position', [0, 0]))
        ego_heading = ego.get('heading', 0)

        # Direction vector
        heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])

        # If position is in direction of heading from origin, we've passed
        return np.dot(ego_pos, heading_vec) > 0

    def _events_to_label(self, events: Set[str]) -> str:
        """Convert event set to DNF label string."""
        if not events:
            return '!approaching&!at_stop&!stopped_early'

        # Priority order for label generation
        priority_events = [
            'collision', 'ran_stop', 'violated', 'hesitation',
            'clear_int', 'exit_int', 'enter_int', 'has_priority',
            'min_stop', 'at_stop', 'stopped', 'approaching',
            'waiting', 'moving', 'stopped_early', 'in_intersection'
        ]

        label_parts = []
        for event in priority_events:
            if event in events:
                label_parts.append(event)

        if not label_parts:
            return '!approaching&!at_stop'

        return '&'.join(label_parts)


# =============================================================================
# Gymnasium Environment Wrapper
# =============================================================================
class ScenicIntersectionEnv(gym.Env):
    """
    Gymnasium environment wrapper for Scenic/SUMO all-way stop intersection.

    This environment:
    1. Loads a Scenic scenario with SUMO simulator
    2. Extracts observations for the ego vehicle
    3. Generates RM event labels
    4. Provides a standard Gym interface for RL training

    If Scenic/SUMO is not available, falls back to HighwayEnv's intersection.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    # Action mapping (discrete)
    ACTIONS = {
        0: 'SLOWER',     # Decelerate
        1: 'IDLE',       # Maintain speed
        2: 'FASTER',     # Accelerate
    }

    def __init__(
        self,
        scenic_file: str = None,
        network_file: str = None,
        use_sumo: bool = True,
        use_scenic_only: bool = False,
        render_mode: str = None,
        max_episode_steps: int = 500,
        traci_label: str = 'default',
        policy_dt: float = 0.2,
        **kwargs
    ):
        """
        Args:
            scenic_file: Path to Scenic scenario file
            network_file: Path to SUMO network file
            use_sumo: Whether to use SUMO/TraCI for simulation
            use_scenic_only: Whether to use Scenic with Newtonian simulator (not TraCI)
            render_mode: 'human' or 'rgb_array'
            max_episode_steps: Maximum steps per episode
            traci_label: TraCI connection label; use distinct labels (e.g. 'train'
                and 'eval') when two ScenicIntersectionEnv instances must coexist
                in the same Python process.
            policy_dt: Policy update period [s]. SUMO runs at step-length=0.1 s;
                each env.step() advances SUMO by round(policy_dt/0.1) physics steps
                so the agent decision frequency matches rho exactly. Default 0.2 s.
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.traci_label = traci_label
        self.current_step = 0

        # Policy update period: SUMO runs at 0.1 s; each env.step() advances
        # round(policy_dt / 0.1) SUMO physics steps so the agent decision
        # frequency matches AllWayStopEnvRM.delta_t (= rho) exactly.
        self.policy_dt = policy_dt
        self._n_sumo_per_policy = max(1, round(policy_dt / 0.1))

        # Ego route edges — set by _spawn_ego_vehicle() and used by
        # AllWayStopEnvRM._update_ego_fcfs_queue() to derive is_left_turn.
        self.ego_route_edges: list[str] = ["edge_south_in", "edge_north_out"]

        # Set up file paths
        base_dir = os.path.dirname(__file__)
        self.scenic_file = scenic_file or os.path.join(base_dir, 'allway_stop_scenario.scenic')
        self.network_file = network_file or os.path.join(base_dir, 'intersection.net.xml')

        # Initialize event detector for RM labeling
        self.event_detector = IntersectionEventDetector(
            stop_line_distance=10.0,
            intersection_half_size=8.0,
            stop_speed_threshold=0.1,
            min_stop_duration=1.5
        )

        # Try to initialize Scenic/SUMO, fall back to HighwayEnv
        self.use_scenic = False
        self.use_traci = False
        self.scenic_sim = None
        self.highway_env = None

        # Priority: TraCI > Scenic > HighwayEnv (when use_sumo=True)
        # If use_scenic_only=True, try Scenic first
        if use_scenic_only and HAS_SCENIC:
            try:
                self._init_scenic_sumo()
                self.use_scenic = True
                logger.info("Using Scenic/Newtonian simulation")
            except Exception as e:
                logger.warning(f"Failed to initialize Scenic: {e}")
                logger.info("Falling back to HighwayEnv")
                self._init_highway_env()
        elif use_sumo and HAS_TRACI:
            # Use SUMO directly via TraCI (without Scenic)
            try:
                self._init_traci_sumo()
                self.use_traci = True
                logger.info("Using SUMO/TraCI simulation (without Scenic)")
            except Exception as e:
                logger.warning(f"Failed to initialize TraCI: {e}")
                logger.info("Falling back to HighwayEnv")
                self._init_highway_env()
        else:
            self._init_highway_env()

        # Define observation and action spaces
        # Observation: [ego_features(8), other_vehicles(N * 8), rm_state_onehot]
        # Features: presence, x, y, vx, vy, cos_h, sin_h, arrival_time
        # NOTE: arrival_time is critical for FCFS priority determination at all-way stops.
        # Without it, this becomes a POMDP where identical observations can require
        # different optimal actions (the agent can't determine who arrived first).
        self.n_vehicles = 15
        self.n_features = 8  # Added arrival_time feature for FCFS priority
        self.n_rm_states = 9  # States 0-8

        # Track arrival times for all vehicles at the stop line
        # Key: vehicle_id, Value: simulation time when vehicle first stopped at stop line
        # This is essential for converting the POMDP into an MDP
        self.vehicle_arrival_times = {}
        self.STOP_LINE_REGION_MIN = 5.0   # Min distance to intersection center for stop line
        self.STOP_LINE_REGION_MAX = 15.0  # Max distance to intersection center for stop line
        self.STOPPED_SPEED_THRESHOLD = 0.5  # Speed below which vehicle is considered stopped

        obs_dim = self.n_vehicles * self.n_features + self.n_rm_states

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

        # RM state tracking
        self.rm_state = 0

    def _init_scenic_sumo(self):
        """Initialize Scenic scenario with newtonian simulator.

        Note: Scenic doesn't have a built-in SUMO simulator.
        We use the newtonian simulator for 2D vehicle dynamics.
        For full SUMO integration, use TraCI directly (see _init_traci_sumo).
        """
        from scenic.simulators.newtonian import NewtonianSimulator
        import pygame

        logger.info(f"Loading Scenic scenario: {self.scenic_file}")
        logger.info(f"Using network file: {self.network_file}")

        # Initialize pygame - required because Scenic's NewtonianSimulator
        # calls pygame.event.get() in step() even when render=False
        pygame.init()
        render_enabled = self.render_mode == 'human'
        if render_enabled:
            pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Scenic Intersection Simulation")
        else:
            # Set dummy video driver for headless operation
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
            pygame.display.set_mode((1, 1))

        # Compile Scenic scenario with render parameter
        self.scenic_scenario = scenic.scenarioFromFile(
            self.scenic_file,
            params={'map': self.network_file, 'render': render_enabled}
        )

        # Generate a scene from the scenario
        scene, _ = self.scenic_scenario.generate()

        # Create newtonian simulator for 2D physics
        self.scenic_sim = NewtonianSimulator(
            render=self.render_mode == 'human'
        )

        # Create simulation from scene
        self.simulation = self.scenic_sim.simulate(scene, maxSteps=self.max_episode_steps)

    def _init_traci_sumo(self):
        """Initialize SUMO simulation via TraCI (without Scenic).

        Uses our intersection.net.xml with allway_stop junction type.
        SUMO automatically handles FCFS priority for all vehicles.
        Traffic vehicles are spawned from intersection.rou.xml.
        Ego vehicle is spawned separately and controlled via TraCI.
        """
        import traci

        self.traci = traci
        self.sumo_running = False

        # SUMO binary selection
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        if self.render_mode == 'human':
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')

        # Check if network file exists
        if not os.path.exists(self.network_file):
            raise FileNotFoundError(f"SUMO network file not found: {self.network_file}")

        # Route file for traffic vehicles
        self.route_file = self.network_file.replace('.net.xml', '.rou.xml')

        # SUMO command line arguments
        self.sumo_cmd = [
            sumo_binary,
            "-n", self.network_file,
            "--no-step-log", "true",
            "--no-warnings", "true",  # Suppress warning messages (e.g., emergency braking)
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",  # Disable teleporting
            "--collision.action", "warn",
            "--step-length", "0.1",
            "--random",  # Add some randomness to traffic
        ]

        # Add route file for traffic vehicles (FCFS handled by allway_stop junction)
        if os.path.exists(self.route_file):
            self.sumo_cmd.extend(["-r", self.route_file])
            logger.info(f"Using route file for traffic: {self.route_file}")

        # Add GUI-specific options
        if self.render_mode == 'human':
            self.sumo_cmd.extend([
                "--start",
                "--quit-on-end", "false",
                "--delay", "50",
            ])

        # Store intersection center for distance calculations
        self.intersection_center = (0.0, 0.0)  # Center junction position

        logger.info(f"SUMO command: {' '.join(self.sumo_cmd)}")

    def _start_traci(self):
        """Start TraCI connection to SUMO."""
        if self.sumo_running:
            try:
                self.traci.switch(self.traci_label)
                self.traci.close()
            except Exception:
                pass
            self.sumo_running = False

        # On macOS, sumo-gui uses Homebrew Mesa (libGL) which needs indirect GLX
        # to render correctly under XQuartz.  Direct Mesa DRI rendering fails on
        # macOS and produces a gray/blank OpenGL window.
        # LIBGL_ALWAYS_INDIRECT=1  → Mesa sends all GL commands to XQuartz via
        #                             the GLX extension (no local DRI/SHM needed).
        # We also pre-launch XQuartz so it is ready before sumo-gui connects,
        # avoiding the "X Fatal error" that occurs when sumo-gui races the
        # launchd-activated XQuartz start-up.
        if self.render_mode == 'human' and sys.platform == 'darwin':
            # Mesa 26+ (Homebrew, updated Apr 2025) changed its direct-rendering
            # path for macOS/XQuartz; the hardware GLX path raises GLXBadContext.
            # Force Mesa software renderer (llvmpipe) so the OpenGL context is
            # created successfully and SUMO-GUI renders the road network.
            os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
            # XQuartz does not support the MIT-SHM extension properly; Qt's use of
            # MIT-SHM for pixmap blitting generates a flood of BadShmSeg X errors
            # which cause sumo-gui to crash before TraCI can connect.
            # QT_X11_NO_MITSHM=1 tells Qt to skip the MIT-SHM optimisation path.
            os.environ['QT_X11_NO_MITSHM'] = '1'
            try:
                import subprocess as _sp
                _sp.run(['open', '-a', 'XQuartz'], check=False,
                        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=5)
                import time as _time
                _time.sleep(2)  # Allow XQuartz to fully initialise
            except Exception:
                pass

        # Start SUMO with retries (SUMO-GUI can be slow to start)
        logger.info("Starting SUMO/TraCI connection...")
        try:
            self.traci.start(self.sumo_cmd, numRetries=20, label=self.traci_label)
        except Exception as e:
            logger.error(f"Failed to connect to SUMO: {e}")
            logger.info("Try running SUMO-GUI manually:")
            logger.info(f"  {' '.join(self.sumo_cmd)}")
            raise RuntimeError(f"Could not connect to SUMO/TraCI: {e}")

        logger.info("Connected to SUMO!")
        self.sumo_running = True
        self.ego_vehicle_id = None

        # Add ego vehicle
        self._spawn_ego_vehicle()

        # Perform one simulation step to ensure ego vehicle is loaded
        # (vehicles added with depart="now" aren't visible until after a step)
        self.traci.simulationStep()
        logger.debug("Initial simulation step performed to load ego vehicle")

        # Centre the GUI viewport on the intersection after connection.
        # Network bounds: (0,0)-(200,200); intersection centre at (100,100).
        # zoom=100 fits the full network in the window (SUMO default scale).
        if self.render_mode == 'human':
            try:
                self.traci.gui.setZoom('View #0', 100)
                self.traci.gui.setOffset('View #0', 100.0, 100.0)
                logger.debug("Set SUMO-GUI viewport: centre=(100,100), zoom=100")
            except Exception as _ve:
                logger.debug(f"Could not set GUI viewport (non-fatal): {_ve}")

    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle in SUMO on the south approach.

        Ego vehicle approaches from SOUTH and goes straight to NORTH.
        This route passes through the allway_stop intersection where
        SUMO's FCFS rules apply (but ego is controlled by RL agent).
        """
        self.ego_vehicle_id = "ego"

        # Define ego route: South to North (straight through intersection)
        ego_route_id = "route_ego_s2n"
        self.ego_route_edges = ["edge_south_in", "edge_north_out"]
        ego_route_edges = self.ego_route_edges

        try:
            # Check if ego vehicle already exists (prevent duplicate spawn)
            existing_vehicles = self.traci.vehicle.getIDList()
            if self.ego_vehicle_id in existing_vehicles:
                logger.debug(f"Ego vehicle '{self.ego_vehicle_id}' already exists, skipping spawn")
                return

            # Check if route already exists
            existing_routes = self.traci.route.getIDList()
            if ego_route_id not in existing_routes:
                self.traci.route.add(ego_route_id, ego_route_edges)

            # Spawn ego vehicle
            # Use ego_type if defined in route file, otherwise DEFAULT_VEHTYPE
            available_types = self.traci.vehicletype.getIDList()
            ego_type = "ego_type" if "ego_type" in available_types else "DEFAULT_VEHTYPE"

            self.traci.vehicle.add(
                self.ego_vehicle_id,
                ego_route_id,
                typeID=ego_type,
                depart="0", # "now"
                departPos=f"{np.random.uniform(3.0, 11.0):.2f}",  # Randomised start: U[3, 11] m from edge start
                departSpeed="10",  # Initial speed 8 m/s
                departLane="0"   # Use lane 0
            )

            # Set ego vehicle color to green for visibility
            self.traci.vehicle.setColor(self.ego_vehicle_id, (0, 255, 0, 255))

            # Disable SUMO's automatic control - RL agent will control
            self.traci.vehicle.setSpeedMode(self.ego_vehicle_id, 0)  # Disable all speed checks

            logger.info(f"Spawned ego vehicle '{self.ego_vehicle_id}' on south approach")

        except Exception as e:
            logger.error(f"Failed to spawn ego vehicle: {e}")
            # Fallback: try any available edge
            edges = [e for e in self.traci.edge.getIDList() if not e.startswith(':')]
            if edges:
                fallback_route = "route_ego_fallback"
                self.traci.route.add(fallback_route, [edges[0]])
                self.traci.vehicle.add(
                    self.ego_vehicle_id,
                    fallback_route,
                    typeID="DEFAULT_VEHTYPE",
                    depart="now"
                )
                logger.warning(f"Used fallback spawn on edge: {edges[0]}")

    def _is_traci_connected(self) -> bool:
        """Check if TraCI connection is valid and responsive.

        Returns:
            True if connection is valid and can communicate with SUMO.
        """
        if not self.sumo_running:
            return False
        try:
            self.traci.switch(self.traci_label)
            _ = self.traci.simulation.getTime()
            return True
        except Exception:
            return False

    def _ensure_traci_connection(self, max_retries: int = 3) -> bool:
        """Ensure TraCI connection is valid, reconnecting if needed.

        This method verifies the connection and attempts to reconnect
        if the connection has been lost. This ensures every timestep
        has a valid SUMO connection for reliable learning.

        Args:
            max_retries: Maximum number of reconnection attempts.

        Returns:
            True if connection is valid (or successfully reconnected).
        """
        if self._is_traci_connected():
            return True

        logger.warning("TraCI connection lost, attempting to reconnect...")

        for attempt in range(max_retries):
            try:
                # Close any existing connection
                if self.sumo_running:
                    try:
                        self.traci.switch(self.traci_label)
                        self.traci.close()
                    except Exception:
                        pass
                    self.sumo_running = False

                # Restart TraCI
                import time
                time.sleep(0.5 * (attempt + 1))  # Increasing backoff

                self.traci.start(self.sumo_cmd, numRetries=10, label=self.traci_label)
                self.sumo_running = True

                # Re-spawn ego vehicle after reconnection
                self._spawn_ego_vehicle()

                # Perform simulation step to load the ego vehicle
                self.traci.simulationStep()

                if self._is_traci_connected():
                    logger.info(f"TraCI reconnected successfully on attempt {attempt + 1}")
                    return True

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")

        logger.error(f"Failed to reconnect TraCI after {max_retries} attempts")
        return False

    def _safe_simulation_step(self) -> bool:
        """Execute SUMO simulation step with connection verification.

        This wrapper ensures the TraCI connection is valid before stepping,
        and handles any transient errors that may occur during simulation.

        Returns:
            True if step executed successfully, False otherwise.
        """
        # Verify connection before stepping
        if not self._ensure_traci_connection():
            logger.error("Cannot step simulation: TraCI connection unavailable")
            return False

        try:
            self.traci.simulationStep()
            return True
        except Exception as e:
            logger.warning(f"Simulation step failed: {e}")
            # Try to recover connection
            if self._ensure_traci_connection():
                try:
                    self.traci.simulationStep()
                    return True
                except Exception as e2:
                    logger.error(f"Simulation step failed after reconnection: {e2}")
            return False

    def _ensure_ego_vehicle(self) -> bool:
        """Check if ego vehicle exists in the simulation.

        Note: This method only checks existence, it does NOT attempt to respawn.
        Respawning during step() causes timing issues because newly added vehicles
        don't appear in getIDList() until after a simulation step.

        Returns:
            True if ego vehicle exists, False otherwise.
        """
        if not self._is_traci_connected():
            return False

        try:
            vehicle_ids = self.traci.vehicle.getIDList()
            if self.ego_vehicle_id and self.ego_vehicle_id in vehicle_ids:
                return True

            # Ego vehicle missing - this typically means it reached its destination
            # or was removed. Don't try to respawn mid-episode.
            logger.debug(f"Ego vehicle '{self.ego_vehicle_id}' not in vehicle list")
            return False

        except Exception as e:
            logger.warning(f"Error checking ego vehicle: {e}")
            return False

    def _update_vehicle_arrival_times(self):
        """Update arrival times for all vehicles at the stop line.

        This method tracks when each vehicle first stops at the stop line,
        which is essential for determining FCFS priority at all-way stops.

        A vehicle is considered "arrived" when:
        1. It is within the stop line region (STOP_LINE_REGION_MIN to STOP_LINE_REGION_MAX)
        2. Its speed is below STOPPED_SPEED_THRESHOLD

        This converts the POMDP into an MDP by making arrival order observable.
        """
        try:
            current_time = self.traci.simulation.getTime()
            vehicle_ids = self.traci.vehicle.getIDList()

            # Clean up arrival times for vehicles that have left the simulation
            self.vehicle_arrival_times = {
                veh_id: t for veh_id, t in self.vehicle_arrival_times.items()
                if veh_id in vehicle_ids
            }

            # Check each vehicle for arrival at stop line
            for veh_id in vehicle_ids:
                # Skip if already recorded arrival time
                if veh_id in self.vehicle_arrival_times:
                    continue

                try:
                    pos = self.traci.vehicle.getPosition(veh_id)
                    speed = self.traci.vehicle.getSpeed(veh_id)

                    # Distance to intersection center (0, 0) after netOffset
                    # Network center is at (100, 100) due to netOffset="100.00,100.00"
                    dist_to_center = np.sqrt((pos[0] - 100.0)**2 + (pos[1] - 100.0)**2)

                    # Check if vehicle is at stop line and stopped
                    at_stop_line = (self.STOP_LINE_REGION_MIN <= dist_to_center <= self.STOP_LINE_REGION_MAX)
                    is_stopped = speed < self.STOPPED_SPEED_THRESHOLD

                    if at_stop_line and is_stopped:
                        self.vehicle_arrival_times[veh_id] = current_time
                        logger.debug(f"Vehicle '{veh_id}' arrived at stop line at t={current_time:.2f}")

                except Exception as e:
                    logger.debug(f"Error checking vehicle {veh_id}: {e}")

        except Exception as e:
            logger.warning(f"Error updating arrival times: {e}")

    def _get_normalized_arrival_time(self, veh_id: str, current_time: float) -> float:
        """Get normalized arrival time for a vehicle.

        Returns a value in [0, 1] where:
        - 0.0 = vehicle arrived long ago (highest priority)
        - 1.0 = vehicle has not arrived yet (no priority)

        Normalization: (current_time - arrival_time) / MAX_WAIT_TIME, clamped to [0, 1]
        Then inverted so earlier arrivals have lower values.

        Args:
            veh_id: Vehicle ID
            current_time: Current simulation time

        Returns:
            Normalized arrival time feature in [0, 1]
        """
        MAX_WAIT_TIME = 30.0  # Maximum wait time for normalization (seconds)

        if veh_id not in self.vehicle_arrival_times:
            return 1.0  # Not arrived yet = lowest priority

        arrival_time = self.vehicle_arrival_times[veh_id]
        wait_time = current_time - arrival_time

        # Normalize and invert: earlier arrival → lower value → higher priority
        # This makes it easy for the agent to compare: lower value = arrived earlier
        normalized = 1.0 - min(wait_time / MAX_WAIT_TIME, 1.0)
        return normalized

    def _get_traci_observation(self):
        """Extract observation from SUMO via TraCI.

        Returns observation array with shape (n_vehicles, n_features).
        Features: [presence, x, y, vx, vy, cos_heading, sin_heading, arrival_time]

        The arrival_time feature is critical for FCFS priority determination:
        - Lower values indicate earlier arrival (higher priority)
        - Value of 1.0 indicates vehicle hasn't arrived at stop line yet

        All values are normalized to approximately [-1, 1] or [0, 1].
        """
        obs = np.zeros((self.n_vehicles, self.n_features), dtype=np.float32)

        try:
            vehicle_ids = self.traci.vehicle.getIDList()
            current_time = self.traci.simulation.getTime()
        except Exception as e:
            logger.warning(f"Failed to get vehicle list: {e}")
            return obs  # Return zeros if connection failed

        # Update arrival times for all vehicles
        self._update_vehicle_arrival_times()

        # Ego vehicle first
        if self.ego_vehicle_id and self.ego_vehicle_id in vehicle_ids:
            pos = self.traci.vehicle.getPosition(self.ego_vehicle_id)
            speed = self.traci.vehicle.getSpeed(self.ego_vehicle_id)
            angle = self.traci.vehicle.getAngle(self.ego_vehicle_id)
            heading = np.radians(90 - angle)  # Convert SUMO angle to heading
            arrival_time_norm = self._get_normalized_arrival_time(self.ego_vehicle_id, current_time)

            obs[0] = [
                1.0,  # presence
                pos[0] / 100.0,  # normalized x
                pos[1] / 100.0,  # normalized y
                speed * np.cos(heading) / 20.0,  # normalized vx
                speed * np.sin(heading) / 20.0,  # normalized vy
                np.cos(heading),
                np.sin(heading),
                arrival_time_norm  # arrival time (lower = earlier = higher priority)
            ]

            # Store ego state for event detection
            self._ego_pos = pos
            self._ego_speed = speed
            self._ego_heading = heading

        # Other vehicles (sorted by distance to ego for relevance)
        other_ids = [v for v in vehicle_ids if v != self.ego_vehicle_id]

        # Sort by distance to ego if ego exists
        if hasattr(self, '_ego_pos') and self._ego_pos is not None:
            ego_pos = np.array(self._ego_pos)
            other_ids_with_dist = []
            for veh_id in other_ids:
                try:
                    other_pos = np.array(self.traci.vehicle.getPosition(veh_id))
                    dist = np.linalg.norm(other_pos - ego_pos)
                    other_ids_with_dist.append((veh_id, dist))
                except:
                    pass
            other_ids_with_dist.sort(key=lambda x: x[1])
            other_ids = [v[0] for v in other_ids_with_dist]

        for i, veh_id in enumerate(other_ids[:self.n_vehicles - 1]):
            try:
                pos = self.traci.vehicle.getPosition(veh_id)
                speed = self.traci.vehicle.getSpeed(veh_id)
                angle = self.traci.vehicle.getAngle(veh_id)
                heading = np.radians(90 - angle)
                arrival_time_norm = self._get_normalized_arrival_time(veh_id, current_time)

                obs[i + 1] = [
                    1.0,
                    pos[0] / 100.0,
                    pos[1] / 100.0,
                    speed * np.cos(heading) / 20.0,
                    speed * np.sin(heading) / 20.0,
                    np.cos(heading),
                    np.sin(heading),
                    arrival_time_norm  # arrival time (lower = earlier = higher priority)
                ]
            except:
                pass  # Vehicle may have left

        return obs

    def _get_traci_info(self):
        """Get info dict from SUMO via TraCI.

        Returns info dict with intersection-specific data for RM event detection:
        - crashed: whether ego collided
        - time: simulation time
        - ego_speed: current speed
        - ego_distance_to_intersection: distance to center junction
        - ego_in_intersection: whether ego is in the intersection area
        - ego_at_stop_line: whether ego is near the stop line
        - ego_stopped: whether ego has stopped (speed < threshold)
        - other_vehicles_in_intersection: count of other vehicles in intersection
        """
        info = {
            'crashed': False,
            'time': 0,
            'ego_speed': 0.0,
            'ego_distance_to_intersection': 100.0,
            'ego_in_intersection': False,
            'ego_at_stop_line': False,
            'ego_stopped': False,
            'other_vehicles_in_intersection': 0,
        }

        try:
            info['time'] = self.traci.simulation.getTime()
            vehicle_ids = self.traci.vehicle.getIDList()
        except Exception as e:
            logger.warning(f"Failed to get simulation info: {e}")
            info['crashed'] = True  # Treat connection loss as crash
            return info

        if self.ego_vehicle_id and self.ego_vehicle_id in vehicle_ids:
            # Check for collisions
            collisions = self.traci.simulation.getCollidingVehiclesIDList()
            if self.ego_vehicle_id in collisions:
                info['crashed'] = True

            # Get ego state
            try:
                pos = self.traci.vehicle.getPosition(self.ego_vehicle_id)
                speed = self.traci.vehicle.getSpeed(self.ego_vehicle_id)
                road_id = self.traci.vehicle.getRoadID(self.ego_vehicle_id)

                info['ego_speed'] = speed

                # Distance to intersection center (0, 0)
                dist_to_center = np.sqrt(pos[0]**2 + pos[1]**2)
                info['ego_distance_to_intersection'] = dist_to_center

                # In intersection if on internal edge (starts with ':')
                info['ego_in_intersection'] = road_id.startswith(':')

                # At stop line: close to intersection but not yet in it
                # Stop line is approximately 5-10m from center for our network
                STOP_LINE_DIST_MIN = 5.0
                STOP_LINE_DIST_MAX = 15.0
                info['ego_at_stop_line'] = (
                    STOP_LINE_DIST_MIN <= dist_to_center <= STOP_LINE_DIST_MAX
                    and not info['ego_in_intersection']
                )

                # Stopped if speed < 0.5 m/s
                info['ego_stopped'] = speed < 0.5

            except Exception as e:
                logger.debug(f"Error getting ego info: {e}")

        # Count other vehicles in intersection
        for veh_id in vehicle_ids:
            if veh_id != self.ego_vehicle_id:
                try:
                    road_id = self.traci.vehicle.getRoadID(veh_id)
                    if road_id.startswith(':'):
                        info['other_vehicles_in_intersection'] += 1
                except:
                    pass

        return info

    def _apply_traci_action(self, action):
        """Apply discrete action to ego vehicle in SUMO.

        Actions:
        - SLOWER (0): Decelerate by 2 m/s
        - IDLE (1): Maintain current speed
        - FASTER (2): Accelerate by 2 m/s

        The ego vehicle's speed is directly controlled, bypassing SUMO's
        automatic car-following. This allows the RL agent to learn the
        proper behavior at the allway_stop intersection.
        """
        try:
            if not self.ego_vehicle_id or self.ego_vehicle_id not in self.traci.vehicle.getIDList():
                return

            # Convert numpy array to int if needed
            if hasattr(action, 'item'):
                action = action.item()
            elif isinstance(action, np.ndarray):
                action = int(action.flatten()[0])

            action_name = self.ACTIONS.get(action, 'IDLE')
            current_speed = self.traci.vehicle.getSpeed(self.ego_vehicle_id)

            # SUMO-compliant acceleration parameters (from intersection.rou.xml ego_type)
            # Units: m/s² (acceleration), consistent with SUMO's vehicle type definition
            ACCEL_MS2 = 5.0   # m/s² - matches ego_type accel parameter
            DECEL_MS2 = 5.0   # m/s² - matches ego_type decel parameter

            # Timestep for this environment (SUMO default)
            dt = 0.1  # seconds (self.dt)

            # SUMO formula: new_speed = current_speed + ACCEL2SPEED(accel)
            # where ACCEL2SPEED(x) = x * TS (timestep in seconds)
            # This matches MSCFModel::maxNextSpeed() from SUMO source
            ACCEL = ACCEL_MS2 * dt  # 3.0 × 0.1 = 0.3 m/s per step
            DECEL = DECEL_MS2 * dt  # 5.0 × 0.1 = 0.5 m/s per step

            MAX_SPEED = 20.0  # m/s - maximum allowed speed (matches SUMO vType maxSpeed)
            MIN_SPEED = 0.0   # m/s - minimum speed (can stop completely)

            if action_name == 'SLOWER':
                new_speed = max(MIN_SPEED, current_speed - DECEL)
            elif action_name == 'FASTER':
                new_speed = min(MAX_SPEED, current_speed + ACCEL)
            else:  # IDLE
                # SUMO-compliant: maintain current speed
                # When using TraCI setSpeed(), vehicles maintain commanded speed indefinitely
                # No natural deceleration in standard SUMO car-following models
                new_speed = current_speed

            self.traci.vehicle.setSpeed(self.ego_vehicle_id, new_speed)
        except Exception as e:
            logger.warning(f"Failed to apply action: {e}")

    def _init_highway_env(self):
        """Initialize HighwayEnv as fallback."""
        import highway_env

        self.highway_env = gym.make(
            "intersection-v0",
            render_mode=self.render_mode
        )

        # Configure for four-way stop
        self.highway_env.unwrapped.config.update({
            "four_way_stop": True,
            "other_vehicles_type": "highway_env.vehicle.realistic.RealisticVehicle",
            "initial_vehicle_count": 7,
            "spawn_probability": 0.0,
            "duration": 50,
            "stop_line_distance": 10.0,
        })

        logger.info("Using HighwayEnv intersection environment")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.rm_state = 0
        self.event_detector.reset()

        # Clear vehicle arrival times for new episode
        # This is essential for FCFS priority tracking
        self.vehicle_arrival_times = {}

        if self.use_scenic:
            # Reset Scenic simulation - regenerate scene
            scene, _ = self.scenic_scenario.generate()
            self.simulation = self.scenic_sim.simulate(scene, maxSteps=self.max_episode_steps)
            obs = self._get_scenic_observation()
        elif self.use_traci:
            # Reset SUMO via TraCI
            self._start_traci()
            obs = self._get_traci_observation()
        else:
            # Reset HighwayEnv
            obs, info = self.highway_env.reset(seed=seed)
            obs = self._process_highway_obs(obs)

        # Append RM state
        rm_onehot = np.zeros(self.n_rm_states, dtype=np.float32)
        rm_onehot[self.rm_state] = 1.0

        full_obs = np.concatenate([obs.flatten(), rm_onehot])

        return full_obs, {}

    def step(self, action: int):
        """Execute action and return new observation."""
        self.current_step += 1

        if self.use_scenic:
            # Apply action to Scenic simulation
            self._apply_scenic_action(action)

            # Step the simulation
            try:
                self.simulation.step()
            except StopIteration:
                pass  # Simulation ended

            obs = self._get_scenic_observation()
            info = self._get_scenic_info()
        elif self.use_traci:
            # Ensure our SUMO connection is active (two-env coexistence via labels)
            self.traci.switch(self.traci_label)
            # Apply action and step SUMO via TraCI
            # First ensure ego vehicle exists before applying action
            if not self._ensure_ego_vehicle():
                logger.warning("Ego vehicle unavailable, returning terminal state")
                obs = np.zeros((self.n_vehicles, self.n_features), dtype=np.float32)
                info = {'crashed': True, 'time': 0, 'connection_lost': True}
            else:
                self._apply_traci_action(action)
                # --- ??? 260525 ---
                # Advance SUMO by _n_sumo_per_policy steps so the elapsed
                # simulation time per env.step() equals policy_dt (= rho).
                # _step_ok = True
                # for _ in range(self._n_sumo_per_policy):
                #     if not self._safe_simulation_step():
                #         _step_ok = False
                #         break
                # --- end ---
                _step_ok = self._safe_simulation_step()
                if not _step_ok:
                    logger.warning("Simulation step failed, returning terminal state")
                    obs = np.zeros((self.n_vehicles, self.n_features), dtype=np.float32)
                    info = {'crashed': True, 'time': 0, 'connection_lost': True}
                else:
                    obs = self._get_traci_observation()
                    info = self._get_traci_info()
        else:
            # Step HighwayEnv
            obs, _, terminated, truncated, info = self.highway_env.step(action)
            obs = self._process_highway_obs(obs)

        # Get state dict for event detection
        state_dict = self._obs_to_state_dict(obs, info)

        # Generate event label
        label = self.event_detector.label(state_dict, info)

        # Store for RM wrapper
        self.last_label = label
        self.last_info = info

        # Check termination
        terminated = info.get('crashed', False) or self.rm_state in [-1] #[7, 8]
        truncated = self.current_step >= self.max_episode_steps

        # Append RM state to observation
        rm_onehot = np.zeros(self.n_rm_states, dtype=np.float32)
        rm_onehot[min(self.rm_state, self.n_rm_states - 1)] = 1.0

        full_obs = np.concatenate([obs.flatten(), rm_onehot])

        # Compute basic reward (RM wrapper will override)
        reward = self._compute_basic_reward(label, info)

        return full_obs, reward, terminated, truncated, info

    def _get_scenic_observation(self):
        """Extract observation from Scenic simulation.

        Features: [presence, x, y, vx, vy, cos_h, sin_h, arrival_time]
        """
        # Initialize observation array
        obs = np.zeros((self.n_vehicles, self.n_features), dtype=np.float32)

        if self.simulation is None:
            return obs

        # Get objects from simulation
        objects = list(self.simulation.objects)
        ego = objects[0] if objects else None
        current_time = getattr(self.simulation, 'currentTime', 0.0)

        # Update arrival times for Scenic vehicles
        self._update_scenic_arrival_times(objects, current_time)

        # Ego observation (first row)
        if ego:
            # Get velocity components from speed and heading
            speed = getattr(ego, 'speed', 0.0)
            heading = getattr(ego, 'heading', 0.0)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            arrival_time_norm = self._get_normalized_arrival_time('ego', current_time)

            obs[0] = [
                1.0,  # presence
                ego.position[0] / 100.0,  # normalized x
                ego.position[1] / 100.0,  # normalized y
                vx / 20.0,   # normalized vx
                vy / 20.0,   # normalized vy
                np.cos(heading),
                np.sin(heading),
                arrival_time_norm  # arrival time
            ]

        # Other vehicles
        for i, vehicle in enumerate(objects[1:self.n_vehicles]):
            speed = getattr(vehicle, 'speed', 0.0)
            heading = getattr(vehicle, 'heading', 0.0)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            veh_id = f'vehicle_{i}'
            arrival_time_norm = self._get_normalized_arrival_time(veh_id, current_time)

            obs[i + 1] = [
                1.0,
                vehicle.position[0] / 100.0,
                vehicle.position[1] / 100.0,
                vx / 20.0,
                vy / 20.0,
                np.cos(heading),
                np.sin(heading),
                arrival_time_norm  # arrival time
            ]

        return obs

    def _update_scenic_arrival_times(self, objects, current_time):
        """Update arrival times for Scenic vehicles."""
        for i, obj in enumerate(objects):
            veh_id = 'ego' if i == 0 else f'vehicle_{i-1}'

            # Skip if already recorded
            if veh_id in self.vehicle_arrival_times:
                continue

            try:
                pos = obj.position
                speed = getattr(obj, 'speed', 0.0)

                # Distance to intersection center
                dist_to_center = np.sqrt(pos[0]**2 + pos[1]**2)

                # Check if at stop line and stopped
                at_stop_line = (self.STOP_LINE_REGION_MIN <= dist_to_center <= self.STOP_LINE_REGION_MAX)
                is_stopped = speed < self.STOPPED_SPEED_THRESHOLD

                if at_stop_line and is_stopped:
                    self.vehicle_arrival_times[veh_id] = current_time
                    logger.debug(f"Scenic vehicle '{veh_id}' arrived at t={current_time:.2f}")
            except Exception as e:
                logger.debug(f"Error checking Scenic vehicle: {e}")

    def _get_scenic_info(self):
        """Get info dict from Scenic simulation."""
        if self.simulation is None:
            return {'crashed': False, 'time': 0}

        objects = list(self.simulation.objects)
        ego = objects[0] if objects else None

        return {
            'crashed': getattr(ego, 'crashed', False) if ego else False,
            'time': self.simulation.currentTime,
        }

    def _apply_scenic_action(self, action):
        """Apply discrete action to Scenic ego vehicle.

        Note: In Scenic's newtonian simulator, we control vehicles by
        setting their throttle/steering or by overriding behaviors.
        """
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()
        elif isinstance(action, np.ndarray):
            action = int(action.flatten()[0])

        action_name = self.ACTIONS.get(action, 'IDLE')

        if self.simulation is None:
            return

        objects = list(self.simulation.objects)
        ego = objects[0] if objects else None

        if ego is None:
            return

        # Map to acceleration command
        if action_name == 'SLOWER':
            accel = -3.0  # Decelerate
        elif action_name == 'FASTER':
            accel = 2.0   # Accelerate
        else:
            accel = 0.0   # Maintain

        # Set throttle based on acceleration
        # Newtonian simulator uses throttle in [-1, 1]
        if hasattr(ego, 'setThrottle'):
            throttle = np.clip(accel / 3.0, -1.0, 1.0)
            ego.setThrottle(throttle)
        elif hasattr(ego, 'throttle'):
            ego.throttle = np.clip(accel / 3.0, -1.0, 1.0)

    def _process_highway_obs(self, obs):
        """Process HighwayEnv observation to standard format.

        HighwayEnv provides 7 features per vehicle, but we need 8 (with arrival_time).
        We add arrival_time tracking for HighwayEnv vehicles as well.
        """
        # HighwayEnv returns (n_vehicles, 7) array - we need (n_vehicles, 8)
        highway_features = 7  # HighwayEnv provides 7 features

        if len(obs.shape) == 1:
            obs = obs.reshape(-1, highway_features)

        # Ensure correct shape with room for arrival_time
        processed = np.zeros((self.n_vehicles, self.n_features), dtype=np.float32)
        min_len = min(obs.shape[0], self.n_vehicles)
        min_feat = min(obs.shape[1], highway_features)

        # Copy the first 7 features from HighwayEnv
        processed[:min_len, :min_feat] = obs[:min_len, :min_feat]

        # Add arrival_time feature (8th feature)
        # For HighwayEnv, we track arrival times based on position and speed
        if hasattr(self, 'highway_env'):
            try:
                current_time = self.current_step * 0.1  # Approximate time based on steps
                self._update_highway_arrival_times(obs, current_time)

                # Set arrival_time for each vehicle
                for i in range(min_len):
                    if obs[i, 0] > 0.5:  # presence check
                        veh_id = 'ego' if i == 0 else f'highway_veh_{i}'
                        arrival_time_norm = self._get_normalized_arrival_time(veh_id, current_time)
                        processed[i, 7] = arrival_time_norm
                    else:
                        processed[i, 7] = 1.0  # Not present = no arrival
            except Exception as e:
                logger.debug(f"Error processing highway arrival times: {e}")
                # Default: set all arrival times to 1.0 (not arrived)
                processed[:, 7] = 1.0
        else:
            # Fallback: set arrival_time to 1.0 (not arrived)
            processed[:, 7] = 1.0

        return processed

    def _update_highway_arrival_times(self, obs, current_time):
        """Update arrival times for HighwayEnv vehicles based on observation."""
        for i in range(obs.shape[0]):
            if obs[i, 0] < 0.5:  # No vehicle present
                continue

            veh_id = 'ego' if i == 0 else f'highway_veh_{i}'

            # Skip if already recorded
            if veh_id in self.vehicle_arrival_times:
                continue

            # Extract position and speed from observation
            # obs format: [presence, x, y, vx, vy, cos_h, sin_h]
            x = obs[i, 1] * 100.0  # Denormalize
            y = obs[i, 2] * 100.0
            vx = obs[i, 3] * 20.0
            vy = obs[i, 4] * 20.0
            speed = np.sqrt(vx**2 + vy**2)

            # Distance to intersection center (assumed at origin for HighwayEnv)
            dist_to_center = np.sqrt(x**2 + y**2)

            # Check if at stop line and stopped
            at_stop_line = (self.STOP_LINE_REGION_MIN <= dist_to_center <= self.STOP_LINE_REGION_MAX)
            is_stopped = speed < self.STOPPED_SPEED_THRESHOLD

            if at_stop_line and is_stopped:
                self.vehicle_arrival_times[veh_id] = current_time
                logger.debug(f"Highway vehicle '{veh_id}' arrived at t={current_time:.2f}")

    def _obs_to_state_dict(self, obs, info):
        """Convert observation array to state dict for event detection."""
        ego_obs = obs[0]

        ego_state = {
            'position': np.array([ego_obs[1] * 100.0, ego_obs[2] * 100.0]),
            'speed': np.sqrt((ego_obs[3] * 20.0)**2 + (ego_obs[4] * 20.0)**2),
            'heading': np.arctan2(ego_obs[6], ego_obs[5]),
            'crashed': info.get('crashed', False),
        }

        others = []
        for i in range(1, self.n_vehicles):
            if obs[i, 0] > 0.5:  # presence
                others.append({
                    'position': np.array([obs[i, 1] * 100.0, obs[i, 2] * 100.0]),
                    'speed': np.sqrt((obs[i, 3] * 20.0)**2 + (obs[i, 4] * 20.0)**2),
                    'heading': np.arctan2(obs[i, 6], obs[i, 5]),
                    'waiting_time': 0.0,  # Would need to track
                    'intersection_state': 'unknown',
                })

        return {
            'ego': ego_state,
            'others': others,
            'intersection_center': np.array([0.0, 0.0]),
        }

    def _compute_basic_reward(self, label: str, info: Dict) -> float:
        """Compute basic reward (RM wrapper will enhance this)."""
        if 'collision' in label:
            return -200.0
        elif 'ran_stop' in label or 'violated' in label:
            return -100.0
        elif 'clear_int' in label:
            return 100.0
        elif 'exit_int' in label:
            return 20.0
        elif 'enter_int' in label:
            return 15.0
        elif 'has_priority' in label:
            return 10.0
        elif 'min_stop' in label:
            return 5.0
        elif 'at_stop' in label and 'stopped' in label:
            return 2.0
        elif 'stopped_early' in label:
            return -5.0
        else:
            return -0.01  # Small time penalty

    def render(self, capture_frame=False):
        """Render the environment and optionally return frame for GIF saving.

        Args:
            capture_frame: If True, capture and return the frame (for GIF saving).
                          If False, just let the GUI render without capturing.

        Returns:
            numpy array of RGB frame if capture_frame=True, else None
        """
        # Debug: log once
        if not hasattr(self, '_render_call_count'):
            self._render_call_count = 0
        self._render_call_count += 1
        if self._render_call_count == 1:
            logger.info(f"render() called: use_scenic={self.use_scenic}, use_traci={self.use_traci}, render_mode={self.render_mode}, capture={capture_frame}")

        if self.use_scenic:
            # Newtonian simulator renders automatically if render=True
            # For rgb_array mode, try to capture pygame screen
            if self.render_mode == 'rgb_array':
                try:
                    import pygame
                    if pygame.display.get_surface():
                        surface = pygame.display.get_surface()
                        frame = pygame.surfarray.array3d(surface)
                        # Transpose from (width, height, 3) to (height, width, 3)
                        return np.transpose(frame, (1, 0, 2))
                except:
                    pass
            return None
        elif self.use_traci:
            # SUMO-GUI renders automatically in 'human' mode
            # Only capture frames when explicitly requested (for GIF saving)
            if self.render_mode == 'human' and capture_frame:
                try:
                    import tempfile
                    import time
                    from PIL import Image

                    # Use TraCI's screenshot function - captures SUMO-GUI window directly
                    # Create a temporary file for the screenshot
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name

                    # Capture screenshot from SUMO-GUI
                    # View ID is typically "View #0" for the default view
                    self.traci.gui.screenshot("View #0", tmp_path)

                    # IMPORTANT: traci.gui.screenshot() is asynchronous
                    # We need to wait for the file to be written
                    max_wait = 0.5  # Max wait time in seconds
                    poll_interval = 0.01  # Check every 10ms
                    waited = 0
                    while waited < max_wait:
                        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                            break
                        time.sleep(poll_interval)
                        waited += poll_interval

                    # Load the screenshot as numpy array
                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                        img = Image.open(tmp_path)
                        frame = np.array(img)

                        # Clean up temp file
                        try:
                            os.remove(tmp_path)
                        except:
                            pass

                        return frame
                    else:
                        if self._render_call_count <= 3:
                            logger.warning(f"Screenshot file not written after {max_wait}s")
                        return None

                except ImportError:
                    if self._render_call_count == 1:
                        logger.warning("PIL not installed. Install with: pip install Pillow")
                except Exception as e:
                    if self._render_call_count <= 3:
                        logger.warning(f"Could not capture SUMO screenshot: {e}")
            # For SUMO-GUI, rendering happens automatically - just return None
            return None
        else:
            return self.highway_env.render()

    def close(self):
        """Clean up resources."""
        if self.use_scenic:
            if self.simulation is not None:
                try:
                    self.simulation.terminate()
                except Exception:
                    pass
            self.simulation = None
            self.scenic_sim = None
            # Quit pygame if it was initialized for rendering
            if self.render_mode == 'human':
                try:
                    import pygame
                    pygame.quit()
                except Exception:
                    pass
        if self.use_traci:
            try:
                if hasattr(self, 'sumo_running') and self.sumo_running:
                    self.traci.switch(self.traci_label)
                    self.traci.close()
                    self.sumo_running = False
            except Exception:
                pass
        if self.highway_env:
            self.highway_env.close()


# =============================================================================
# RM-Wrapped Environment with FCFS Queue Logic
# =============================================================================
class AllWayStopEnvRM(gym.Wrapper):
    """
    Product MDP with Reward Machine Wrapping for All-Way Stop Intersection.

    Based on train_hwyenv_stopandgo_rm_single_sas.py, this wrapper:
    1. Tracks FCFS queue for priority determination
    2. Extracts physical state via st4obs()
    3. Generates event labels via L() with 'y' (yielding) detection
    4. Applies reward shaping in step()
    5. Returns product MDP observations via get_observation()

    The RM is loaded from rm_examples/stop_fcfs.txt with states:
    - 0: Initial/safe approach
    - 1: Approaching (unsafe)
    - 2: Forced stop mode
    - 3: Yielding (stopped, waiting for priority)
    - 4: Cleared (has priority, proceeding)
    - 5: Goal (terminal success)
    - -1: Terminal (collision/violation)
    """

    def __init__(self, env: ScenicIntersectionEnv, rm: RewardMachine, delta_t=None, a_max=6.0, a_min=-6.0, a_min_c=-5.0, a_min_f=-6.0, enable_logging=True):
        super().__init__(env)

        self.rm = rm
        # Count only non-terminal states (exclude -1) for observation encoding
        self.num_rm_states = len([s for s in rm.get_states() if s >= 0])

        # Episode logging control
        self.enable_episode_logging = enable_logging

        # FCFS Queue for priority tracking
        self.fcfs_queue = FCFSQueue()

        # =============================================================================
        # Intersection Geometry (from SUMO network: intersection.net.xml)
        # =============================================================================
        # Network structure (after netOffset="100,100"):
        #   - Center junction at (100, 100) with type="allway_stop"
        #   - Junction box: ~14.4m x 14.4m (from y=92.8 to y=107.2) [1-lane network]
        #   - Approach edges are 92.8m long (e.g., edge_south_in: y=0 to y=92.8)
        #   - Stop line is at edge of junction (y=92.8 for south approach)
        #
        # Reference: https://sumo.dlr.de/docs/Networks/PlainXML.html#node_types
        # "allway_stop: Vehicles have to stop at the stop line, then pass the
        #  intersection in first-come-first-served order."
        # =============================================================================

        # Intersection center in SUMO coordinates (after netOffset)
        self.INTERSECTION_CENTER = np.array([100.0, 100.0])

        # Distance from intersection center to stop line
        # Junction shape shows stop line at 7.2m from center (100 - 92.8 = 7.2)
        # [1-lane network: junction box spans 92.8–107.2 on each axis]
        self.STOP_LINE_FROM_CENTER = 7.2

        # Stop zone: where vehicle should stop (distance from center)
        # Stop line is at 7.2m, allow margin of +~5m
        self.STOP_ZONE_START = self.STOP_LINE_FROM_CENTER   # Inner edge (closer to center)
        self.STOP_ZONE_END = 52    # Outer edge (further from center)

        # Total edge length (from intersection.net.xml: length="92.80") [1-lane network]
        self.EDGE_LENGTH = 92.8

        # Ego spawn position (from _spawn_ego_vehicle: departPos="10")
        self.EGO_SPAWN_POS = 10.0

        # Initial distance from spawn to stop line
        # = EDGE_LENGTH - EGO_SPAWN_POS = 92.8 - 10 = 82.8m
        self.INITIAL_DIST_TO_STOP = self.EDGE_LENGTH - self.EGO_SPAWN_POS - self.STOP_LINE_FROM_CENTER  # 82.8m

        # Distance where forced braking should start (gives time to brake from ~10m/s)
        # At 10 m/s with -5 m/s² decel: stopping distance = v²/(2a) = 100/10 = 10m
        # Add safety margin: start braking at ~40m from stop line
        self.REACT_TO_BRAKE = self.STOP_ZONE_END #40.0

        # Approach distance (for normalization in observations)
        self.APPROACH_DIST = self.EDGE_LENGTH  # 92.8m

        # Speed threshold for "stopped" (m/s)
        self.STOP_SPEED_THRESH = 0.1 #0.5

        # Minimum stop duration before can proceed (seconds)
        # Match SUMO's default jmAllwayStopWait behavior: one simulation step
        self.MIN_STOP_DURATION = 0.1  # One timestep, same as SUMO's default

        # SUMO-compliant FCFS registration and waiting time tracking
        # SUMO doesn't specify an all-way stop registration distance explicitly.
        # Based on SUMO's default visibilityDistance (4.5m for priority junctions)
        # and JM_CROSSING_GAP_DEFAULT (10m), we use 15m as a conservative threshold.
        # This is smaller than the forced-stop zone but large enough for early detection.
        self.FCFS_REGISTRATION_DISTANCE = 5   # meters from stop line — register only when
        # the ego is genuinely at the stop line (within the pull-up zone), so that FCFS
        # arrival-time comparisons are physically meaningful and YIELD labels are not
        # assigned spuriously during the 40-m approach phase. (0223 -> 40)

        # Physics parameters (from reference)
        self.dt = 0.1  # SUMO step length
        self.kt = 0.2  # Policy frequency
        self.delta_t = self.kt if not delta_t else delta_t
        self.a_min = a_min  # Minimum deceleration (m/s^2)
        self.a_max = a_max
        self.a_min_conf = a_min_c
        self.a_min_f = a_min_f
        self.r_a = 1  # Speed reward scaling: 0.4 (0223);
        self.v_max = 20  # Max speed (SUMO lane speed limit from intersection.net.xml) 13.89?
        self.v_min = 0.0

        # State tracking
        self.prev_u_id = 0
        self.gamma = 0.99
        self.prev_dist_s = self.INITIAL_DIST_TO_STOP
        self.prev_phi_s = max(0, 1.0 - (self.prev_dist_s / self.EDGE_LENGTH))
        self.prev_speed = 0.0
        self.ego_stopped_time = 0.0
        self.ego_arrival_time = None
        self.step_count = 0  # Track steps to avoid premature termination on first step
        self.prev_Q = False             # Queue condition from previous step (for cleared bonus)
        self.prev_leader_id = None
        self.prev_leader_speed = 0.0
        self.prev_d_inquiry = self.INITIAL_DIST_TO_STOP

        # Observation space: augment base env obs with RM state one-hot and distance
        assert isinstance(env.observation_space, gym.spaces.Box)

        # Get actual observation size from base env
        actual_obs, _ = env.reset()
        actual_obs_size = actual_obs.flatten().shape[0] - env.n_rm_states  # Remove existing RM states

        low = env.observation_space.low.flatten()[0] if len(env.observation_space.low.flatten()) > 0 else -np.inf
        high = env.observation_space.high.flatten()[0] if len(env.observation_space.high.flatten()) > 0 else np.inf

        # Observation: [base_obs, dist_to_stop_normalized, rm_state_onehot]
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(actual_obs_size + 1 + self.num_rm_states,),
            dtype=np.float32
        )

        # Computing one-hot encodings for non-terminal RM states
        self.rm_state_features = {}
        non_terminal_states = sorted([s for s in rm.get_states() if s >= 0])
        state_to_index = {u_id: i for i, u_id in enumerate(non_terminal_states)}

        for u_id in non_terminal_states:
            self.rm_state_features[u_id] = np.eye(self.num_rm_states)[state_to_index[u_id]]

        # Terminal state feature is all zeros
        self.rm_done_feat = np.zeros(self.num_rm_states)

        # Current RM state
        self.current_u_id = None

        # Episode logging buffer
        self.log_buffer = []

    def reset(self, **kwargs):
        """Reset environment, RM, and FCFS queue.

        Spawn-collision guard: with SUMO's '--collision.action warn' and
        '--random', a traffic vehicle occasionally occupies the ego's random
        departure position (U[3,11] m on edge_south_in at t=0).  Despite the
        name 'warn', SUMO removes colliding vehicles, so the ego can be absent
        from the simulation before the first step() call — causing an immediate
        spurious terminal episode.  We retry env.reset() (which fully restarts
        SUMO with a fresh random seed each time) until _ensure_ego_vehicle()
        confirms the ego survived initialisation.  Spawn collisions are rare
        (~5–10% of seeds), so 10 retries gives a failure probability < 10^-10.
        """
        _MAX_RESET_RETRIES = 10
        for _attempt in range(_MAX_RESET_RETRIES):
            self.obs, info = self.env.reset(**kwargs)
            # Only check when using SUMO/TraCI; other backends are unaffected.
            if self.env.use_traci and not self.env._ensure_ego_vehicle():
                logger.warning(
                    f"Ego vehicle absent after reset "
                    f"(attempt {_attempt + 1}/{_MAX_RESET_RETRIES}) — "
                    "spawn collision with traffic, retrying with new SUMO seed."
                )
                continue
            break   # ego present (or not using TraCI) — proceed normally
        self.current_u_id = self.rm.reset()

        # Reset FCFS queue
        self.fcfs_queue.reset()

        # Reset tracking variables
        self.ego_stopped_time = 0.0
        self.ego_arrival_time = None
        #self.prev_dist_s = self.INITIAL_DIST_TO_STOP
        #self.prev_phi_s = self.get_potential(phys_state)
        self.prev_stopped = False
        self.step_count = 0
        self.prev_speed = 0.0
        self.prev_u_id = 0
        self.prev_Q = False             # Queue condition from previous step (for cleared bonus)
        self.prev_leader_id = None
        self.prev_leader_speed = 0.0
        self.prev_d_ego_to_front = self.INITIAL_DIST_TO_STOP
        self.prev_d_inquiry = self.INITIAL_DIST_TO_STOP


        # Get initial physical state
        phys_state = self.st4obs(self.obs)
        self.prev_dist_s = self.INITIAL_DIST_TO_STOP #phys_state['dist_to_stop_line']
        self.prev_phi_s = self.get_potential(phys_state)

        if self.enable_episode_logging:
            logger.info(f"EPISODE START: dist_to_int={phys_state['dist_to_intersection']:.1f}, speed={phys_state['v_e']:.1f}, pos=({phys_state['x_e']:.1f}, {phys_state['y_e']:.1f})")

        # Return augmented observation
        return self.get_observation(self.obs, self.current_u_id, False, phys_state['dist_to_intersection']), info

    def step(self, action):
        """
        Step environment and update RM state with reward shaping.

        Following the pattern from train_hwyenv_stopandgo_rm_single_sas.py:
        1. Execute action in environment
        2. Extract physical state via st4obs()
        3. Generate event label via L()
        4. Update RM state
        5. Apply reward shaping based on RM state
        6. Return augmented observation
        """
        # Execute action in environment
        next_obs, _, env_done, truncated, info = self.env.step(action)

        # Increment step counter
        self.step_count += 1

        # Extract physical state
        phys_state = self.st4obs(next_obs)

        # Update FCFS queue with other vehicles
        self._update_fcfs_queue(info)

        # Update ego FCFS registration and waiting time (simulation-clock-synchronized)
        # Must precede L() so must_yield('ego') sees a fully current queue for t_N
        self._update_ego_fcfs_queue(phys_state, info)

        # Generate event label
        true_props = self.L(phys_state, info)

        # Update RM state
        # Convert true_props to frozenset for DNF evaluation
        true_props_set = frozenset([true_props]) if true_props else frozenset()
        self.current_u_id, rm_rew, rm_done = self.rm.step(self.current_u_id, true_props_set, info)

        # === REWARD SHAPING ===
        #dist_s = phys_state['dist_to_intersection']
        #dist_to_stop_line = phys_state['dist_to_stop_line']
        dist_s = phys_state['dist_to_stop_line']
        dist_f = phys_state['d_ego_to_front']
        v_e = phys_state['v_e']
        v_f = phys_state['v_front']
        d_inq = phys_state['d_inquiry']    # unified obstacle/stop-line distance
        d_rss = phys_state['d_rss_safe']   # RSS minimum required following distance
        # at_stop_line = phys_state['ego_at_stop_line']
        # is_stopped = phys_state['ego_stopped']

        phi_s = self.get_potential(phys_state) # Potential of (s)
        F_shape = self.gamma * phi_s - self.prev_phi_s
        # shaping_i = 50 * F_shape # 100 *

        # Queue condition Q: a lead vehicle lies geometrically between the ego
        # vehicle and the stop line (i.e., d_ego_to_front < dist_to_stop_line).
        #   True  → a predecessor is queued ahead; ego following is correct car-following.
        #   False → path to stop line is clear; ego should be pulling up to stop line.
        # Computed once here so it is in scope for both the State-2 in-state block
        # and the 2→3 transition block, and can be stored for the next step.
        Q_now = phys_state['d_ego_to_front'] < dist_s

        # # State 2 (Forced Stop): Encourage approaching stop zone correctly
        # if self.current_u_id == 2 and (not true_props in ['s','x','p']):
            
        #     theta_5 = 1 #1; self.r_a
        #     theta_6 = 2 #4; self.r_a
        #     #rm_rew = -theta_5 * dist_s * dist_s - theta_6 * v_e * v_e
        #     #rm_rew = theta_3 * (1.0 - (dist_s / (self.REACT_TO_BRAKE - self.STOP_ZONE_START)))
        #     rm_rew = 1.0 - (dist_s / (self.REACT_TO_BRAKE - self.STOP_ZONE_START))


        # State 0 (Safe Approach): reward for speed + forward-progress toward stop line.
        # The pure speed reward rm_rew = 4*(v_e/v_max) creates a zero-reward stable
        # attractor whenever the ego is stopped (v_e=0 → rm_rew=0 every step).  The
        # policy then parks arbitrarily far from the stop line because staying stopped
        # is costless.  Two additions break this equilibrium:
        #   (a) PBRS forward-progress: ~0.5 reward-unit advantage per metre of advance
        #       vs standing still, providing a continuous gradient toward the stop line.
        #   (b) Wrong-position-stop penalty: −1.0 when stopped outside the FCFS zone
        #       with no leader blocking, making the stopped equilibrium strictly negative.
        # Both are conditioned on Q_now=False (clear path, no leader) and
        # dist_s > FCFS_REGISTRATION_DISTANCE (ego has not yet reached the stop zone).
        if self.current_u_id == 0:
            if true_props in ['AB']:
                # Direct boundary-proximity reward/penalty — analogue of state 1 '0'.
                #
                # err_AB = d_inq − d_rss  (un-clamped, ≥ 0 by phi_AB invariant).
                # With a_max=a_min=5 m/s², d_rss can be negative when the leader
                # is fast; un-clamped so braking amplifies err and the signal is 2×
                # stronger than the clamped form.
                #
                # Linear form:  rm_rew = clamp(1 − err/2, min=−1)
                #   err = 0 m → +1.0  (peak: ego exactly at RSS boundary)
                #   err = 2 m →  0.0  (zero crossing)
                #   err ≥ 4 m → −1.0  (floor: far from boundary → negative gradient)
                #
                # Speed incentive is implicit: higher v_e → larger d_rss → smaller
                # err for the same physical gap → higher rm_rew.  This avoids the
                # need for an explicit speed term while preserving boundary-first
                # priority.
                err_AB = d_inq - d_rss           # ≥ 0 by phi_AB invariant; no clamp
                rm_rew = max(-1.0, 1.0 - err_AB / 2.0) # [best]

                # if not Q_now and dist_s >= self.FCFS_REGISTRATION_DISTANCE:
                #     # (3) PBRS forward-progress: reward closing distance to stop line.
                #     # F_prog = γ·Φ(s') − Φ(s), Φ(s) = −c·dist_s.
                #     # Advancing 1 m yields ~0.5 extra reward vs standing still.
                #     c_prog = 0.5
                #     F_prog = self.gamma * (-c_prog * dist_s) - (-c_prog * self.prev_dist_s)
                #     rm_rew += F_prog

                #     # (4) Wrong-position-stop penalty: ego is stopped with a clear
                #     # path but has not yet reached the FCFS registration zone.
                #     if v_e < self.STOP_SPEED_THRESH:
                #         rm_rew -= 1.0
                # rm_rew = 1.0 - err_AB / 2.0 # [not good]
                # rm_rew = -np.tanh(err_AB - 2.0) # [good]
                # rm_rew = 2**(1 - 0.5 * err_AB) - 1 # [not good]
                c_prog = 0.5
                F_prog = self.gamma * (-c_prog * dist_s) - (-c_prog * self.prev_dist_s)
                rm_rew += (dist_s >= self.FCFS_REGISTRATION_DISTANCE)*F_prog
                
                # # --- Try 0226 ---
                # # CBF boundary-proximity reward (state-grounded: uses h≡d_inq, v_e, v_s from s).
                # # err_AB ≥ 0 by phi_AB invariant. Tanh: smooth, bounded (−1,+1), steepest gradient
                # # at the critical zero-crossing err=2m, no clamp needed.
                # err_AB = d_inq - d_rss
                # #rm_rew = -np.tanh(err_AB - 2.0)
                # rm_rew = max(-1.0, 1.0 - err_AB / 2.0)

                # # Unconditional PBRS: Φ(s) = −c·dist_s, c=0.5.
                # # Provides the forward-progress gradient when the CBF saturates (dist_s >> d_rss(v_e)).
                # # Unconditional removes the Q_now discontinuity and the FCFS domain threshold.
                # # Theoretically justified: Ng et al. (1999) guarantees policy-invariance for any
                # # potential-based shaping F = γΦ(s')−Φ(s) over the full state space.
                # c_prog = 0.5
                # rm_rew += self.gamma * (-c_prog * dist_s) - (-c_prog * self.prev_dist_s)
                # # --- End of try 0226 ---
                
                

            # elif true_props in ['f']:
            #     theta_3 = self.r_a #10.0; self.r_a
            #     a_opt = self.a_min #v_e*v_e/(2*dist_s)
                #rm_rew = theta_3 * (1.0 - (abs(v_e*v_e-2*a_opt*dist_s) / (self.v_max*self.v_max)))
            # rm_rew = self.r_a * (v_e - self.v_min) / (self.v_max - self.v_min) #+ shaping_i
            # # PBRS forward-progress shaping: identical formula to State 2 but at
            # # half strength (c_dist_0 = 0.5 vs c_dist = 1.0).  Every metre of
            # # approach yields ~0.50/step at 10 m/s, making it costly to park
            # # mid-approach just outside the 'f' zone boundary.
            # c_dist_0 = 0.5
            # F_prec_0 = self.gamma * (-c_dist_0 * dist_to_stop_line) - (-c_dist_0 * self.prev_dist_s)
            # rm_rew += F_prec_0
            # # Speed-following incentive: when a preceding vehicle is ahead and moving
            # # faster than the ego (Q_now=True, v_f > v_e), reward closing the speed
            # # gap.  This counters the early-deceleration artifact (v_e dropping from
            # # 10→8 m/s at d_e2s=58m even though RSS is satisfied and the front vehicle
            # # is pulling away at 13+ m/s).
            # #
            # # Coefficient 0.2 is modest (~0.04/step at v_gap=4 m/s) — just enough to
            # # break the old training equilibrium without conflicting with RSS or PBRS.
            # # Conditioned on dist_to_stop_line > 2*FCFS_REG_DIST so it does not fight
            # # the stop-zone deceleration when ego is within 10m of the stop line.
            # if Q_now and dist_to_stop_line > 2.0 * self.FCFS_REGISTRATION_DISTANCE:
            #     v_front = phys_state.get('v_front', 0.0)
            #     if v_front > v_e:
            #         speed_gap_norm = (v_front - v_e) / self.v_max
            #         rm_rew += 0.2 * speed_gap_norm
            # # Penalise unnecessary mid-approach stop (only if not yet at stop line
            # # and a predecessor is not forcing the pause).
            # if is_stopped and dist_to_stop_line > self.FCFS_REGISTRATION_DISTANCE:
            #     v_front = phys_state.get('v_front', 0.0)
            #     if not Q_now:
            #         # Path to stop line is clear — same pull-up force as State 2.
            #         rm_rew -= 1.5
            #     elif v_front > self.STOP_SPEED_THRESH:
            #         # Predecessor is moving again — ego must resume following.
            #         rm_rew -= 0.5
            # # One-time queue-cleared bonus to initiate pull-up (mirrors State 2).
            # if self.prev_Q and not Q_now and dist_to_stop_line > self.FCFS_REGISTRATION_DISTANCE:
            #     rm_rew += 3.0

        # State 2 (Yielding): Small time penalty to encourage efficiency
        # elif self.current_u_id == 2:
            # --- (?) 0210 ---
            # if true_props == 'y':
            #     rm_rew = -0.05  # Small penalty for waiting (but not too punishing)
            # else:
            #     rm_rew = rm_rew + shaping_i
            # --- (?) 0210 ---
            # if not (true_props == 'y'):
            #     rm_rew = rm_rew * v_e

        elif self.current_u_id == 1:
            if true_props in ['0']:
                # Violation-magnitude penalty: proportional to the RSS deficit
                # (how far inside the unsafe zone we are), not to speed.
                # When d_inq == d_rss the violation is 0 → no penalty at
                # the exact boundary (transition to 'AB' will fire instead).
                # σ=10 m normalises the deficit so a 10 m shortfall gives −2.0.
                #violation = max(0.0, d_rss - d_inq)
                violation = d_rss - d_inq
                rm_rew = -2.0 * violation / 10.0
                # rm_rew = -np.tanh(d_rss - d_inq - 2.0)

        # State 3 (Cleared): Strong reward for proceeding through intersection
        elif self.current_u_id == 3:
            if not (true_props in ['x', 'p']):
                delta_8 = 2.0
                #rm_rew = delta_8 * (v_e/self.v_max)
                rm_rew = rm_rew * v_e

        # elif self.current_u_id == 2 and true_props in ['q']:
        #     # Speed maintenance: don't slow down just because state 1 looms
        #     speed_reward = 4.0 * (v_e - self.v_min) / (self.v_max - self.v_min)

        #     if v_f > 1.0:
        #         # Leader is moving: reward matching its speed
        #         # Maximized when v_e == v_f, zero when v_e == 0
        #         speed_match = 0.5 * max(0.0, 1.0 - abs(v_e - v_f) / self.v_max)
        #     else:
        #         # Leader stopped (waiting at stop line): reward ego for also braking
        #         # This is necessary behavior, not penalize it
        #         speed_match = 0.3 * (1.0 - v_e / self.v_max)

        #     rm_rew = speed_reward + speed_match
        
        # # --- Transition reward override: State 2 → State 3 ('s' event) ---
        # # Fires exactly once on the step that the 2→3 arc is consumed.
        # # Replaces the constant RM reward (R_base = 20) with a context-aware formulation.
        # #
        # # Queue condition Q: a lead vehicle lies geometrically between the ego
        # # vehicle and the stop line, i.e. it is closer to the stop line than ego is.
        # #   Q = True  → premature stop in queue   → severe penalty (overrides 's' reward)
        # #   Q = False → clean stop at stop line   → R_base + Gaussian precision bonus
        # #
        # # When Q is True, the policy foresees -P_premature and learns to maintain
        # # a marginal creep velocity (v > STOP_SPEED_THRESH) behind the lead vehicle,
        # # holding the RM in State 2 until the queue clears, then pulling up to the
        # # stop line and triggering the unpenalised transition.
        # if self.prev_u_id == 2 and self.current_u_id == 3:
        #     if not Q_now:
        #         # Gaussian precision bonus peaked at d_stop = 0.
        #         # sigma=4.0 m extends the 3-sigma influence radius to ~12 m, covering the
        #         # agent's current typical stopping position (~10 m) and producing a
        #         # non-zero learning gradient there: bonus at 0 m = 30.0, at 4 m ~18.2,
        #         # at 8 m ~4.1, at 10 m ~1.3, at 12 m ~0.3.
        #         # A sigma of 1.5 m was too narrow: at 10 m the bonus collapsed to ~0,
        #         # giving the policy no gradient signal to pull up closer.
        #         omega_prec = 30.0
        #         sigma_prec = 4.0   # (m) — 3-sigma radius ≈ 12 m covers current stop zone
        #         rm_rew = 20.0 + omega_prec * np.exp(
        #             -(dist_to_stop_line ** 2) / (2.0 * sigma_prec ** 2)
        #         )
        #     else:
        #         # Premature stop behind lead vehicle: override entire transition reward.
        #         rm_rew = -50.0   # P_premature

        # Store for next step
        self.prev_u_id = self.current_u_id
        self.prev_speed = phys_state['v_e']
        self.prev_dist_s = dist_s  # FIXED: Use dist_to_stop_line, not dist_to_intersection
        self.prev_phi_s = self.get_potential(phys_state)
        self.prev_Q = Q_now                   # Track queue condition for cleared bonus

        # Termination
        done = rm_done or env_done

        # Debug logging (format matching train_hwyenv_stopandgo_rm_single_sas.py)
        if self.enable_episode_logging:
            # Log all steps when enabled (for eval/demo/debug modes)
            # Previously: Log on first 5 steps, every 50 steps, or when done
            # if done or self.step_count <= 5 or (self.env.current_step % 50 == 0):

            # Get FCFS priority status for ego
            has_priority = self.fcfs_queue.has_priority('ego') if 'ego' in self.fcfs_queue.arrival_times else None
            fcfs_status = 'PRIORITY' if has_priority else ('YIELD' if has_priority is False else 'N/A')

            # Get FCFS info for ego (SUMO-compliant: show waiting_time and arrival_time)
            if 'ego' in self.fcfs_queue.arrival_times:
                ego_info = self.fcfs_queue.arrival_times['ego']
                ego_arrival = ego_info['time']
                ego_wait = ego_info['waiting_time']
                arrival_info = f"arr={ego_arrival:.1f},wait={ego_wait:.1f}"
            else:
                arrival_info = "arr=N/A,wait=N/A"

            # Get info for all other vehicles in FCFS queue (show waiting_time)
            other_vehicles_info = []
            for veh_id, veh_info in self.fcfs_queue.arrival_times.items():
                if veh_id == 'ego':
                    continue
                in_int = 'INT' if veh_id in self.fcfs_queue.in_intersection else 'WAIT'
                cleared = 'CLR' if veh_id in self.fcfs_queue.cleared_vehicles else ''
                wait_time = veh_info['waiting_time']
                other_vehicles_info.append(
                    f"{veh_id}(arr={veh_info['time']:.1f},wait={wait_time:.1f},{in_int}{cleared})"
                )
            other_vehs_str = ', '.join(other_vehicles_info) if other_vehicles_info else 'None'

            log_message = (
                f"T={info.get('time', 0):.1f} | X: {phys_state['x_e']:.1f} | "
                f"d_e2s: {phys_state['dist_to_stop_line']:.1f} | "
                f"d_e2f: {phys_state['d_ego_to_front']:.1f} | "
                f"v_f: {phys_state['v_front']:.2f} | v_e: {phys_state['v_e']:.2f} | "
                f"a: {action} | "
                f"Evt: {true_props} | RM State: {self.current_u_id} | "
                f"FCFS: {fcfs_status} ({arrival_info}) | "
                f"Others: [{other_vehs_str}] | "
                f"r: {rm_rew:.2f} | env_done={env_done}"
            )
            logger.info(log_message)

        # Expose the event so callers (evaluate, _evaluate_inline) do not need to
        # re-invoke L(), which would see stale prev_dist_s and miss 'p', and would
        # call vehicle_exited_intersection('ego') twice on the 'g' step.
        info['event'] = true_props

        # Return augmented observation
        rm_obs = self.get_observation(next_obs, self.current_u_id, done, dist_s)

        return rm_obs, rm_rew, done, truncated, info

    def get_observation(self, next_obs, u_id, done, dist_val=0.0):
        """
        Create product MDP observation.

        Combines:
        - Base environment observation (flattened)
        - Normalized distance to intersection
        - RM state one-hot encoding
        """
        # Normalize distance (0 at intersection, 1 at approach distance)
        dist_feat = np.array([dist_val / self.APPROACH_DIST], dtype=np.float32)

        # Get RM state features
        if u_id in self.rm_state_features:
            feat = self.rm_state_features[u_id]
        else:
            feat = np.zeros(self.num_rm_states)

        rm_feat = self.rm_done_feat if done else feat

        # Remove existing RM states from base observation and concatenate new ones
        base_obs = next_obs.flatten()
        # The base env already adds RM states - we need to replace them
        base_obs_without_rm = base_obs[:-self.env.n_rm_states]

        aug_obs = np.concatenate([base_obs_without_rm, dist_feat, rm_feat], axis=0)

        return aug_obs.astype(np.float32)

    def get_potential(self, state):
        # Higher value = closer to goal. 
        # Max distance to consider ~100m. 
        # Max Potential = 1.0 (at 0m). Min Potential = 0.0 (at 100m).
        d = state['dist_to_intersection']
        return max(0, 1.0 - (d / self.EDGE_LENGTH))

    def st4obs(self, obs):
        """
        Extract continuous physical state from observations.

        Returns a dictionary with:
        - x_e, y_e: ego position
        - v_e: ego speed
        - heading_e: ego heading
        - dist_to_intersection: distance from ego to intersection center
        - dist_to_stop_line: signed distance to stop line (positive=before, negative=after)
        - d_ego_to_front: distance to nearest vehicle in front
        - v_front: speed of nearest vehicle in front
        - ego_at_stop_line: boolean, whether ego is in stop zone [0, 5m from stop line]
        - ego_stopped: boolean, whether ego has stopped (speed < 0.1 m/s)
        """
        state = {
            'x_e': 0.0,
            'y_e': 0.0,
            'v_e': 0.0,
            'heading_e': 0.0,
            'dist_to_intersection': self.APPROACH_DIST,
            'dist_to_stop_line': self.INITIAL_DIST_TO_STOP,
            'd_ego_to_front': self.INITIAL_DIST_TO_STOP,
            'd_inquiry': self.INITIAL_DIST_TO_STOP,
            'v_front': 0.0,
            'v_inquiry': 0.0,
            'ego_at_stop_line': False,
            'ego_stopped': False,
            'd_rss_safe': 0.0
        }

        # If using TraCI, get direct data
        if self.env.use_traci and hasattr(self.env, 'traci'):
            try:
                ego_id = self.env.ego_vehicle_id
                if ego_id and ego_id in self.env.traci.vehicle.getIDList():
                    pos = self.env.traci.vehicle.getPosition(ego_id)
                    speed = self.env.traci.vehicle.getSpeed(ego_id)
                    angle = self.env.traci.vehicle.getAngle(ego_id)
                    heading = np.radians(90 - angle)

                    state['x_e'] = pos[0]
                    state['y_e'] = pos[1]
                    state['v_e'] = speed
                    state['heading_e'] = heading

                    # Distance to intersection center (100, 100) after netOffset
                    # The intersection center is at INTERSECTION_CENTER
                    center = self.INTERSECTION_CENTER

                    # Euclidean distance for dist_to_intersection
                    dist_to_center = np.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2)
                    state['dist_to_intersection'] = dist_to_center

                    # SIGNED distance to stop line (along approach direction):
                    # Use projection onto heading vector to get signed distance
                    # - Positive: vehicle before stop line (approaching)
                    # - Zero: vehicle at stop line
                    # - Negative: vehicle past stop line (and becomes MORE negative as vehicle moves away)
                    heading_vec = np.array([np.cos(heading), np.sin(heading)])
                    to_center = center - pos  # Vector from vehicle to center
                    signed_dist_to_center = np.dot(to_center, heading_vec)  # Positive if heading toward center
                    state['dist_to_stop_line'] = signed_dist_to_center - self.STOP_LINE_FROM_CENTER

                    # Check if at stop line (within margin of stop zone)
                    state['ego_at_stop_line'] = 0 <= state['dist_to_stop_line'] <= self.FCFS_REGISTRATION_DISTANCE
                    #state['ego_at_stop_line'] = 0 <= state['dist_to_stop_line'] <= (self.STOP_ZONE_END-self.STOP_ZONE_START)
                    # state['ego_at_stop_line'] = self.STOP_ZONE_START <= state['dist_to_stop_line'] <= self.STOP_ZONE_END # self.STOP_ZONE_START <= dist_s <= self.STOP_ZONE_END

                    # Check if stopped
                    state['ego_stopped'] = state['v_e'] < self.STOP_SPEED_THRESH # speed

                    # Find closest vehicle in front using SUMO's topology-aware leader
                    # detection.  traci.vehicle.getLeader() follows ego's planned route
                    # (edge_south_in → :center_7 internal lane → edge_north_out), so it
                    # correctly tracks a preceding car even after it has crossed the stop
                    # line and entered the junction.
                    #
                    # Previous approach initialised min_dist = dist_to_stop_line, which
                    # silently dropped any vehicle whose bumper gap exceeded that value.
                    # When the preceding vehicle just crossed the stop line (gap growing
                    # from 61.0 to 60.75 m while d_e2s shrank from 61.2 to 60.5 m),
                    # gap > d_e2s by 0.25 m → vehicle lost → d_e2f snapped to d_e2s,
                    # v_front snapped to 0, Q_now falsely flipped to False, and the
                    # one-time +3.0 queue-cleared bonus fired prematurely.
                    # Mode gate: the stop-line sentinel (d_ego_to_front = dist_to_stop_line,
                    # v_front = 0) is only valid while the ego is still approaching from
                    # outside the FCFS registration zone.  Once within FCFS_REGISTRATION_DISTANCE
                    # (or past the stop line, where dist_to_stop_line goes negative), using
                    # dist_to_stop_line as the sentinel produces zero or negative d_ego_to_front,
                    # which causes phi_AB to evaluate False and emits spurious '0' labels while
                    # the ego is stopped at or proceeding through the intersection.
                    dist_s = state['dist_to_stop_line']
                    approaching = dist_s >= self.FCFS_REGISTRATION_DISTANCE

                    leader = self.env.traci.vehicle.getLeader(ego_id, 0)
                    if leader is not None:
                        leader_id, raw_gap = leader
                        # raw_gap is already bumper-to-bumper (SUMO subtracts leader length)
                        state['d_ego_to_front'] = max(0.0, raw_gap)
                        state['v_front'] = self.env.traci.vehicle.getSpeed(leader_id)
                        if approaching:
                            # Approaching mode: gate on whether the leader is still between
                            # the ego and the stop line.
                            
                            if raw_gap < dist_s:
                                # Leader is between ego and stop line → car-following
                                # state['d_ego_to_front'] = max(0.0, raw_gap)
                                # state['v_front'] = self.env.traci.vehicle.getSpeed(leader_id)
                                state['d_inquiry'] = state['d_ego_to_front']
                                state['v_inquiry'] = state['v_front']
                                # Persist for one-step holdover
                                # self.prev_leader_id      = leader_id
                                # self.prev_leader_speed   = state['v_front']
                                self.prev_d_inquiry = state['d_ego_to_front']
                            else:
                                # Leader is at or past the stop line → use stop line as sentinel
                                #state['d_ego_to_front'] = dist_s
                                #state['v_front'] = 0.0
                                state['d_inquiry'] = dist_s
                                state['v_inquiry'] = 0.0
                                # self.prev_leader_id      = None
                                # self.prev_leader_speed   = state['v_e'] #0.0
                                self.prev_d_inquiry = dist_s
                        else:
                            # At/past stop line mode: use actual bumper-to-bumper headway
                            # unconditionally.  The dist_to_stop_line gate is meaningless here
                            # (dist_s may be negative) and must not be applied.
                            #state['d_ego_to_front'] = max(0.0, raw_gap)
                            #state['v_front'] = self.env.traci.vehicle.getSpeed(leader_id)
                            if dist_s >= 0:
                                state['d_inquiry'] = dist_s
                                state['v_inquiry'] = 0.0
                            else:
                                state['d_inquiry'] = state['d_ego_to_front'] # max(0.0, raw_gap)
                                state['v_inquiry'] = state['v_front'] #self.env.traci.vehicle.getSpeed(leader_id)

                        self.prev_d_ego_to_front = state['d_ego_to_front']
                        self.prev_leader_speed = state['v_front']
                        self.prev_leader_id = leader_id

                    else:
                        # getLeader() returned None.
                        # Check whether this is a one-step junction boundary gap
                        # (leader just crossed onto internal lane :center_7).
                        # Condition: a leader was tracked last step (prev_Q) AND the
                        # previous leader is still active in the simulation.
                        if (self.prev_leader_id is not None) and self.prev_Q and \
                                self.prev_leader_id in self.env.traci.vehicle.getIDList():
                            # Hold the stale gap for this single step; next step
                            # getLeader() will reattach to the leader normally.
                            state['d_ego_to_front'] = self.prev_d_ego_to_front
                            state['v_front'] = self.prev_leader_speed
                        else:
                            # Genuinely no leader on approach: stop line is the sentinel
                            # state['d_ego_to_front'] = dist_s
                            # state['v_front'] = 0.0
                            state['d_ego_to_front'] = self.INITIAL_DIST_TO_STOP
                            state['v_front'] = state['v_e']
                            # Reset holdover state; approaching mode will not recur this episode
                            self.prev_leader_id      = None
                            self.prev_leader_speed   = state['v_e']

                        if approaching:
                            if state['d_ego_to_front'] < dist_s:
                                state['d_inquiry'] = state['d_ego_to_front'] #self.prev_d_inquiry
                                state['v_inquiry'] = state['v_front'] #self.prev_leader_speed
                            else:
                                state['d_inquiry'] = dist_s
                                state['v_inquiry'] = 0.0
                                # self.prev_leader_id      = None
                                # self.prev_leader_speed   = 0.0
                        else:
                            # At/past stop line, no leader: ego is in free flow through the
                            # intersection.  Assign a large safe headway and treat the virtual
                            # leader as moving at ego speed (no obstacle ahead).  This keeps
                            # phi_AB = True and produces label 'AB' (not '0') so the ego is
                            # not penalised for correctly proceeding through the intersection.
                            # state['d_ego_to_front'] = self.INITIAL_DIST_TO_STOP
                            # state['v_front'] = state['v_e']
                            if dist_s >= 0:
                                state['d_inquiry'] = dist_s
                                state['v_inquiry'] = 0.0
                            else:
                                state['d_inquiry'] = state['d_ego_to_front']
                                state['v_inquiry'] = state['v_front']
                            # state['d_inquiry'] = self.INITIAL_DIST_TO_STOP
                            # state['v_inquiry'] = state['v_e']
                        
                        self.prev_d_ego_to_front = state['d_ego_to_front']
                            

            except Exception as e:
                logger.debug(f"Error in st4obs (TraCI): {e}")
        else:
            # Parse from observation array (HighwayEnv fallback)
            # Note: HighwayEnv uses different coordinate system, center may be at (0,0)
            ego_obs = obs.flatten()[:7] if len(obs.flatten()) >= 7 else np.zeros(7)

            if ego_obs[0] > 0.5:  # presence
                # HighwayEnv observations are normalized differently
                state['x_e'] = ego_obs[1] * 100.0
                state['y_e'] = ego_obs[2] * 100.0
                vx = ego_obs[3] * 20.0
                vy = ego_obs[4] * 20.0
                state['v_e'] = np.sqrt(vx**2 + vy**2)
                state['heading_e'] = np.arctan2(ego_obs[6], ego_obs[5])

                # For HighwayEnv, assume intersection center at origin
                dist = np.sqrt(state['x_e']**2 + state['y_e']**2)
                state['dist_to_intersection'] = dist

                # SIGNED distance to stop line (along approach direction)
                center = np.array([0.0, 0.0])  # HighwayEnv uses origin as center
                pos = np.array([state['x_e'], state['y_e']])
                heading_vec = np.array([np.cos(state['heading_e']), np.sin(state['heading_e'])])
                to_center = center - pos
                signed_dist_to_center = np.dot(to_center, heading_vec)
                state['dist_to_stop_line'] = signed_dist_to_center - self.STOP_LINE_FROM_CENTER
                state['ego_at_stop_line'] = 0 <= state['dist_to_stop_line'] <= self.FCFS_REGISTRATION_DISTANCE
                #state['ego_at_stop_line'] = 0 <= state['dist_to_stop_line'] <= (self.STOP_ZONE_END-self.STOP_ZONE_START)
                # state['ego_at_stop_line'] = self.STOP_ZONE_START <= state['dist_to_stop_line'] <= self.STOP_ZONE_END # dist
                state['ego_stopped'] = state['v_e'] < self.STOP_SPEED_THRESH

        state['d_rss_safe'] = state['v_e']*self.delta_t + 0.5*self.a_max*self.delta_t*self.delta_t + (state['v_e']+self.a_max*self.delta_t)*(state['v_e']+self.a_max*self.delta_t)/(2*abs(self.a_min_conf)) - state['v_inquiry']*state['v_inquiry']/(2*abs(self.a_min_f))
        return state

    def L(self, state, info):
        """
        Labeling function: maps physical state to event propositions.

        Events (from rm_examples/stop_fcfs.txt):
        - x: collision
        - p: passed stop line without stopping
        - g: reached goal (exited intersection)
        - s: stopped at stop line properly
        - f: forced stop zone (should brake)
        - y: yielding (must wait for FCFS priority)
        - !y: has FCFS priority (can proceed)
        - AB, C, D: safe braking conditions (from RSS formulas)
        - 0: unsafe (needs to brake)

        Returns: event string for RM transition
        """
        # Check collision first
        if info.get('crashed', False):
            return 'x'

        # Extract state variables
        v_e = state['v_e']
        dist_s = state['dist_to_stop_line']
        dist_int = state['dist_to_intersection']
        dist_f = state['d_ego_to_front']
        v_s = state['v_front']
        dist_i = state['d_inquiry']
        v_i = state['v_inquiry']
        a_e = self.a_min
        a_s = self.a_min
        a_e_max = self.a_max
        a_e_min_c = self.a_min_conf
        a_e_min = self.a_min
        a_f_min = self.a_min_f
        delta_t = self.delta_t

        # === Position-based state ===
        #in_intersection = dist_int < self.STOP_ZONE_START
        # Check if past center using heading-based detection
        ego_pos = np.array([state['x_e'], state['y_e']])
        past_center = self._is_past_center(ego_pos, state['heading_e'])
        past_intersection = past_center and dist_s < -2 * self.STOP_LINE_FROM_CENTER-self.env.traci.vehicle.getLength(self.env.ego_vehicle_id)-20 #dist_int > self.STOP_LINE_FROM_CENTER
        at_stop_line = state['ego_at_stop_line']
        is_stopped = state['ego_stopped']
        #in_stop_zone = state['ego_at_stop_line'] #self.STOP_ZONE_START <= dist_s <= self.STOP_ZONE_END

        # === RSS-style safe braking conditions ===
        # From the reference implementation
        # A_bool = (dist_f > v_e * delta_t + v_e * v_e / (2 * abs(a_e)))
        # B_bool = ((v_s / abs(a_s) > delta_t) and
        #           (dist_f > (v_e - v_s) * delta_t + v_e * v_e / (2 * abs(a_e)) - 0.5 * a_s * delta_t * delta_t))
        # C_bool = ((abs(a_e) > abs(a_s)) and
        #           (v_e > v_s + a_s * delta_t) and
        #           (v_e / abs(a_e) < (v_s + a_s * delta_t) / abs(a_s)) and
        #           (dist_f > (v_e - v_s) * delta_t - 0.5 * a_s * delta_t * delta_t +
        #            (v_e - v_s - a_s * delta_t) ** 2 / (2 * (a_s - a_e))))
        # D_bool = (dist_f > v_e * delta_t + v_e * v_e / (2 * abs(a_e)) - v_s * v_s / (2 * abs(a_s)))

        #phi_AB = A_bool or B_bool
        phi_AB = (dist_i > v_e*delta_t + 0.5*a_e_max*delta_t*delta_t + (v_e+a_e_max*delta_t)*(v_e+a_e_max*delta_t)/(2*abs(a_e_min_c)) - v_i*v_i/(2*abs(a_f_min)))
        # phi_AB = (dist_f > v_e*delta_t + 0.5*a_e_max*delta_t*delta_t + (v_e+a_e_max*delta_t)*(v_e+a_e_max*delta_t)/(2*abs(a_e_min_c)) - v_s*v_s/(2*abs(a_e_min)))
        # phi_C = (not phi_AB) and C_bool
        # phi_D = (not phi_AB) and (not C_bool) and D_bool


        # === Stopped at stop line (SUMO-aligned: one timestep stop) ===
        # Once stopped for one timestep (MIN_STOP_DURATION = 0.1s), check FCFS priority:
        # 1. No other vehicle is currently traversing the intersection
        # 2. Arrived before all other waiting vehicles, or earlier arrivals have cleared
        if at_stop_line and is_stopped:
            if self.ego_stopped_time > self.MIN_STOP_DURATION:
                # Check FCFS priority (min_stop_duration param kept for backward compat)
                if self.fcfs_queue.must_yield('ego'):
                    return 'y'  # Must yield - other vehicle has priority
                else:
                    return 'r' # '!y'  # Has priority, can proceed through intersection
            else:
                return 's'  # Just stopped, will check priority next timestep

        # === Forced stop zone ===
        # Fire 'f' when ego is within the forced-deceleration corridor AND no preceding
        # vehicle is between ego and the stop line (dist_f >= dist_s means the front
        # vehicle is at or beyond the stop line, i.e., the path ahead is clear).
        # Previously used min(threshold, dist_f) which collapsed to dist_s < dist_s
        # (always False) because st4obs defaults dist_f = dist_s when no front vehicle.
        # if (dist_s > 0) and (not is_stopped):
        #     if dist_s < (self.REACT_TO_BRAKE - self.STOP_ZONE_START): #and dist_f >= dist_s:
        #         return 'f'

        # === Passed without stopping ===
        # Skip this check on first step (step_count <= 1) since we can't reliably
        # detect "passing without stopping" before the vehicle has had time to move
        if self.step_count > 1:
            # --- (?) 0210 ---
            # if in_intersection and self.ego_stopped_time < self.MIN_STOP_DURATION and not is_stopped:
            #     if self.prev_dist_s >= self.STOP_ZONE_START:  # Was outside, now inside
            #         return 'p'  # Ran stop sign
            # --- (?) 0210 end ---
            # --- (?) 0507 ---
            # if dist_s < 0:
            #     return 'p'
            # --- (?) 0507 end ---
            if dist_s < 0 and self.prev_dist_s >= 0:
                return 'p'
            
            
        # === Goal Detection ===
        if past_intersection: #and dist_int > self.STOP_LINE_FROM_CENTER:
            self.fcfs_queue.vehicle_exited_intersection('ego')
            return 'g'

        
        # === Safe braking conditions ===
        if phi_AB:
            # --- 0225 Try ---
            # if dist_f < dist_s:   # Q_now: leader is between ego and stop line
            #     return 'q'        # Queue-following mode: RSS safe while following leader
            # else:
            #     return 'AB'       # Direct approach: RSS safe toward stop line (no leader)
            return 'AB'
            # --- 0225 Try end ---
        # if phi_C:
        #     return 'C'
        # if phi_D:
        #     return 'D'

        # # === Unsafe state ===
        # if (not phi_AB) and (not phi_C) and (not phi_D):
        #     return '0'

        # Default
        return '0'

    def _update_fcfs_queue(self, info):
        """
        Update FCFS queue with other vehicle positions from TraCI (SUMO-compliant).

        Implements SUMO's all-way stop behavior:
        1. Register vehicles when they ENTER the stop zone (not when stopped)
        2. Accumulate waiting time continuously while speed < 0.1 m/s
        3. Track intersection entry/exit for conflict resolution

        This matches SUMO's MSLink::setApproaching() and waiting time accumulation.
        """
        if not self.env.use_traci:
            return

        try:
            sim_time = info.get('time', 0.0)
            self.fcfs_queue.simulation_time = sim_time

            for veh_id in self.env.traci.vehicle.getIDList():
                if veh_id == self.env.ego_vehicle_id:
                    continue

                pos = np.array(self.env.traci.vehicle.getPosition(veh_id))
                speed = self.env.traci.vehicle.getSpeed(veh_id)
                angle = self.env.traci.vehicle.getAngle(veh_id)
                heading = np.radians(90 - angle)
                road_id = self.env.traci.vehicle.getRoadID(veh_id)

                # Distance to intersection center (100, 100)
                dist_to_center = np.linalg.norm(pos - self.INTERSECTION_CENTER)
                dist_to_stop_line = dist_to_center - self.STOP_LINE_FROM_CENTER

                in_fcfs_zone = 0 <= dist_to_stop_line <= self.FCFS_REGISTRATION_DISTANCE
                is_stopped = speed < self.STOP_SPEED_THRESH

                # Register when entering FCFS zone (matches SUMO's setApproaching() concept)
                if in_fcfs_zone:
                    if veh_id not in self.fcfs_queue.arrival_times:
                        # Determine left-turn intent from the vehicle's route edges.
                        try:
                            route = self.env.traci.vehicle.getRoute(veh_id)
                            road_edges = [e for e in route if not e.startswith(':')]
                            incoming = road_edges[0] if road_edges else ''
                            outgoing = road_edges[-1] if len(road_edges) > 1 else ''
                            is_left_turn = FCFSQueue._is_left_turn_route(incoming, outgoing)
                        except Exception:
                            is_left_turn = False
                        self.fcfs_queue.register_arrival(veh_id, sim_time, pos, heading,
                                                         is_left_turn=is_left_turn)

                    # Update waiting time continuously (SUMO's updateWaitingTime())
                    self.fcfs_queue.update_waiting_time(veh_id, is_stopped, self.dt)

                # Check if in intersection (internal lane or within square center zone).
                # Square half-side = STOP_ZONE_START so all four stop lines form the boundary.
                # A circle (Euclidean dist) misses turning vehicles in the corner regions
                # (up to sqrt(2)*STOP_ZONE_START away), letting them slip past Condition 1.
                dx = abs(pos[0] - self.INTERSECTION_CENTER[0])
                dy = abs(pos[1] - self.INTERSECTION_CENTER[1])
                in_square = dx < self.STOP_ZONE_START and dy < self.STOP_ZONE_START
                if road_id.startswith(':') or in_square:
                    self.fcfs_queue.vehicle_entered_intersection(veh_id)

                # # Check if exited (outside square zone and past center in target lane).
                # # Guard on in_intersection (not arrival_times) so vehicles that entered
                # # via the ':' branch without FCFS registration are also cleared.
                # if not in_square and veh_id in self.fcfs_queue.in_intersection:
                #     if self._is_past_center(pos, heading):
                #         self.fcfs_queue.vehicle_exited_intersection(veh_id)

                # MATCHING with SUMO
                if not road_id.startswith(':') and veh_id in self.fcfs_queue.in_intersection:
                    self.fcfs_queue.vehicle_exited_intersection(veh_id)

        except Exception as e:
            logger.debug(f"Error updating FCFS queue: {e}")

    def _update_ego_fcfs_queue(self, state, info):
        """Update ego vehicle FCFS registration and waiting time.

        Extracted from L() so that all FCFS queue writes (ego + surrounding)
        occur at the same point in step() — once per simulationStep(), keeping
        the queue synchronized with the simulation clock (dt=0.1 s).

        L() must be called after this method so that must_yield('ego') sees a
        fully up-to-date queue for the current timestep.
        """
        dist_s = state['dist_to_stop_line']
        is_stopped = state['ego_stopped']
        in_fcfs_zone = 0 <= dist_s <= self.FCFS_REGISTRATION_DISTANCE

        if in_fcfs_zone:
            if self.ego_arrival_time is None:
                self.ego_arrival_time = info.get('time', 0.0)  # traci.simulation.getTime() at t_N
                _ego_edges = self.env.ego_route_edges
                self.fcfs_queue.register_arrival(
                    'ego',
                    self.ego_arrival_time,
                    np.array([state['x_e'], state['y_e']]),
                    state['heading_e'],
                    is_left_turn=FCFSQueue._is_left_turn_route(
                        _ego_edges[0], _ego_edges[-1]
                    ),
                )

            # Update waiting time continuously — matches SUMO's updateWaitingTime()
            self.fcfs_queue.update_waiting_time('ego', is_stopped, self.dt)  # dt=0.1 s per step

            # Track stopped time for reward shaping
            if is_stopped:
                self.ego_stopped_time += self.dt
        else:
            self.ego_stopped_time = 0.0

    def _is_past_center(self, pos, heading):
        """Check if vehicle has passed intersection center.

        Uses the heading vector to determine if the vehicle's position
        relative to the intersection center is "in front" (past center).
        """
        # Position relative to intersection center
        center = self.INTERSECTION_CENTER
        rel_pos = np.array(pos) - center
        heading_vec = np.array([np.cos(heading), np.sin(heading)])
        # Positive dot product means position is in direction of heading from center
        return np.dot(rel_pos, heading_vec) > 0

    def render(self, capture_frame=False):
        """Render the environment and optionally return frame for GIF saving."""
        return self.env.render(capture_frame=capture_frame)


# =============================================================================
# AllWayStop Simulator
# =============================================================================
class AllWayStopSimulator:
    """
    Simulator for data collection on the AllWayStopEnvRM environment.

    Analogous to ReacherDiscreteSimulator in the learning_reward_machines repo,
    adapted for the SUMO all-way-stop intersection.

    State discretization (equivalent to ReacherDiscretizerUniform):
      - Spatial fields (x_e, y_e, dist_to_stop_line, d_ego_to_front, d_inquiry,
        d_rss_safe, heading_e): rounded to the nearest 0.01 (2 decimal places).
      - Velocity fields (v_e, v_front, v_inquiry): rounded to the nearest 0.1
        (1 decimal place).
      - Boolean fields (ego_at_stop_line, ego_stopped): unchanged.

    Values are rounded directly (no bin maps), so the state key is a hashable
    tuple of rounded floats — equivalent to a discrete state index but without
    the memory overhead of a full Cartesian-product table.

    The state key in state_action_counts / state_action_probs is this physical
    discrete state tuple (not the RM state integer).  This allows trajectories
    that visit the same physical region to share statistical evidence, and keeps
    the number of unique label histories per key tractably small.

    Action space:
      Discrete(3) — actions are integers {0, 1, 2} directly; no action
      discretizer is needed (unlike the Reacher / Highway cases).

    Policy input:
      Raw gymnasium observations (the policy was trained without discretization);
      only the state key and label extraction use the discretized state dict.

    Label extraction:
      env.L(discrete_state, info) is called with the discretized state dict.
      For terminal events ('g', 'x', 'p') that carry irreversible side effects
      inside env.L() (e.g. vehicle_exited_intersection), info['event'] is used
      directly instead of re-calling env.L().  For the FCFS fully_tied
      tiebreaker, vehicle positions and headings in the FCFS queue are
      temporarily rounded to 0.01 before the geometric right-of-way / oncoming
      checks, then restored.

    Justification — why waiting_time and time need no discretization:
      SUMO advances the simulation clock by exactly step_length = 0.1 s per
      step, so both the arrival time ('time', set once when a vehicle registers)
      and the waiting time (accumulated as n_steps × 0.1 s) are inherently
      multiples of 0.1 s.  The EPSILON window in has_priority() is 0.01 s,
      which is strictly smaller than one step, so simultaneous-arrival ties can
      only occur when two vehicles registered in the exact same simulation step.
      No additional discretization is required.
    """

    # Discretization increments (matching the user specification)
    _SPATIAL_INCR: float = 0.01   # x_e, y_e, dist_to_stop_line, d_ego_to_front,
                                   # d_inquiry, d_rss_safe, heading_e
    _VEL_INCR:     float = 0.1    # v_e, v_front, v_inquiry

    def __init__(self, env: 'AllWayStopEnvRM', policy):
        self.env = env
        self.policy = policy
        self.state_action_counts = {}   # {discrete_state_key: [(compressed_label, Counter)]}
        self.state_action_probs = {}    # {discrete_state_key: {compressed_label: np.array}}
        self.state_label_counts = {}    # {discrete_state_key: {compressed_label: int}}
        self.n_actions = env.action_space.n     # 3 (Discrete(3))
        self.n_states = env.num_rm_states       # GT-RM state count (metadata only)
        self.history = []

    # ------------------------------------------------------------------
    # Discretization helpers
    # ------------------------------------------------------------------

    def discretize_state(self, state: dict) -> dict:
        """Return a copy of *state* with each field rounded to its grid increment.

        Spatial fields (x_e, y_e, dist_to_stop_line, d_ego_to_front, d_inquiry,
        d_rss_safe, heading_e) → rounded to 2 decimal places (0.01 increment).
        Velocity fields (v_e, v_front, v_inquiry) → rounded to 1 decimal place
        (0.1 increment).  Boolean and derived fields are passed through unchanged.
        """
        s = dict(state)
        for k in ('x_e', 'y_e', 'dist_to_stop_line', 'd_ego_to_front',
                  'd_inquiry', 'd_rss_safe', 'heading_e'):
            s[k] = round(s[k], 2)
        for k in ('v_e', 'v_front', 'v_inquiry'):
            s[k] = round(s[k], 1)
        # ego_at_stop_line, ego_stopped, dist_to_intersection: unchanged
        return s

    def state_to_key(self, discrete_state: dict) -> tuple:
        """Convert a discretized state dict to a hashable tuple state key.

        Equivalent to the integer discrete_state_idx used in
        ReacherDiscreteSimulator but built without a full Cartesian-product
        index table — the rounded values themselves serve as the unique key.
        """
        return (
            discrete_state['x_e'],
            discrete_state['y_e'],
            discrete_state['v_e'],
            discrete_state['heading_e'],
            discrete_state['dist_to_stop_line'],
            discrete_state['d_ego_to_front'],
            discrete_state['d_inquiry'],
            discrete_state['d_rss_safe'],
            discrete_state['v_front'],
            discrete_state['v_inquiry'],
            discrete_state['ego_at_stop_line'],
            discrete_state['ego_stopped'],
        )

    def midpoint_from_state(self, discrete_state: dict) -> dict:
        """Return the 'midpoint' representative of a discrete state cell.

        With nearest-neighbour rounding the discretized value IS the bin
        centre (the bin spans [v − δ/2, v + δ/2) and its midpoint is v),
        so this is the identity on already-discretized state dicts.
        Provided for API symmetry with ReacherDiscretizerUniform.
        """
        return discrete_state

    # ------------------------------------------------------------------
    # Discretized labelling function
    # ------------------------------------------------------------------

    def L_discretized(self, discrete_state: dict, info: dict) -> str:
        """Call env.L() with the discretized state and rounded FCFS queue positions.

        Terminal events ('g', 'x', 'p') have irreversible side effects inside
        env.L() (e.g. vehicle_exited_intersection).  These are already handled
        by env.step(), so we return info['event'] directly for those cases.

        For all other events ('s', 'y', 'r', 'AB', '0') we call env.L() with:
          - discrete_state  (rounded physical state values)
          - FCFS queue positions rounded to 0.01 m and headings rounded to
            0.01 rad for the fully_tied geometric tiebreaker, then restored.
        """
        event = info.get('event', '0')
        # Terminal / side-effect events: trust info['event'] directly.
        if event in ('g', 'x', 'p'):
            return event

        # Temporarily patch fcfs_queue arrival_times with discretized geometry.
        original: dict = {}
        for vid, vinfo in self.env.fcfs_queue.arrival_times.items():
            original[vid] = (vinfo['position'].copy(), vinfo['heading'])
            vinfo['position'] = np.round(vinfo['position'], 2)   # 0.01 m
            vinfo['heading']  = round(vinfo['heading'], 2)        # 0.01 rad

        try:
            label = self.env.L(discrete_state, info)
        finally:
            # Always restore original geometry regardless of exceptions.
            for vid, (orig_pos, orig_heading) in original.items():
                if vid in self.env.fcfs_queue.arrival_times:
                    self.env.fcfs_queue.arrival_times[vid]['position'] = orig_pos
                    self.env.fcfs_queue.arrival_times[vid]['heading']  = orig_heading

        return label

    # ------------------------------------------------------------------
    # Shared utility
    # ------------------------------------------------------------------

    def remove_consecutive_duplicates(self, s: str) -> str:
        """Compress consecutive identical labels in a comma-separated string.

        Labels like 'AB' or 'y' repeat across many consecutive steps (approach
        phase, yield-wait phase, etc.).  Without compression every trajectory
        produces a unique label-history key, making empirical action
        distributions degenerate (count = 1 per key).  Compression maps
        semantically equivalent trajectories — same qualitative label sequence,
        differing only in repetition count — to the same key.
        """
        elements = s.split(',')
        if not elements:
            return s
        result = [elements[0]]
        for i in range(1, len(elements)):
            if elements[i] != elements[i - 1]:
                result.append(elements[i])
        return ','.join(result)

    # ------------------------------------------------------------------
    # Trajectory sampling
    # ------------------------------------------------------------------

    def sample_trajectory(self, len_traj: int, render: bool = False) -> bool:
        """Run one episode; commit data only if the ego reaches the goal ('g').

        Only successful trajectories (ending with event 'g', i.e. ego cleared
        the intersection) contribute to state_action_counts.  Failed episodes
        (collision 'x', ran stop 'p', or timeout truncated) are silently
        discarded — their buffered data is never committed.

        Returns:
            True  if the trajectory ended with event 'g' (goal reached).
            False otherwise (collision, ran stop, or timeout).

        Justification for 'g' as the success criterion:
          - 'g' is emitted by L() only when ego_cleared_intersection=True, which
            simultaneously sets terminated=True.  It is therefore the final event
            of a successful episode and cannot appear mid-trajectory.
          - Failed endings always carry a different final event ('x', 'p') or
            set truncated=True, so `terminated and event == 'g'` is both
            necessary and sufficient.

        Mirrors ReacherDiscreteSimulator.sample_trajectory() with the following
        correspondences:
          st4obs()            → env.st4obs(obs)     (gets physical state dict)
          discretize_state()  → rounds each field to its grid increment
          state_to_key()      → produces a hashable tuple (≡ discrete_state_idx)
          midpoint_from_state()  → identity (rounded value = bin centre)
          L_discretized()     → env.L(rounded_state, info) with rounded queue
          policy.predict(obs) → raw obs (policy was trained without discretisation)
        """
        self.history = []
        obs, _ = self.env.reset()

        # Compute initial discrete state key from the reset observation.
        state_dict = self.env.st4obs(obs)
        disc_state = self.discretize_state(state_dict)
        state_key  = self.state_to_key(disc_state)
        label      = ''
        event      = ''

        # Per-trajectory buffer: committed to state_action_counts only on success.
        traj_buffer = []  # list of (state_key, action, compressed_label)

        for _ in range(len_traj):
            # Policy prediction uses raw obs (trained without discretisation).
            action, _ = self.policy.predict(obs, deterministic=False)
            self.history.append((state_key, int(action)))

            obs, reward, terminated, truncated, info = self.env.step(action)
            if render:
                self.env.render()

            # Compute next discrete state after the step.
            next_state_dict = self.env.st4obs(obs)
            next_disc_state = self.discretize_state(next_state_dict)

            # Label from the discretized next state (with rounded FCFS geometry).
            event = self.L_discretized(next_disc_state, info)
            label = (label + ',' + event) if label else event

            compressed_label = self.remove_consecutive_duplicates(label)

            # Buffer this step — not yet written to state_action_counts.
            traj_buffer.append((state_key, int(action), compressed_label))

            # Advance to the next discrete state key.
            state_key = self.state_to_key(next_disc_state)

            if terminated or truncated:
                self.history = []
                break

        # Commit buffer only for successful trajectories (goal reached).
        goal_reached = terminated and event == 'g'
        if goal_reached:
            for sk, act, comp_lbl in traj_buffer:
                if sk not in self.state_action_counts:
                    self.state_action_counts[sk] = []

                label_exists = False
                for _, (existing_label, counter) in enumerate(
                        self.state_action_counts[sk]):
                    if existing_label == comp_lbl:
                        counter[act] += 1
                        label_exists = True
                        break

                if not label_exists:
                    self.state_action_counts[sk].append(
                        (comp_lbl, Counter({act: 1}))
                    )

        return goal_reached

    def sample_dataset(self, n_trajectories: int, max_trajectory_length: int):
        """Collect data from exactly n_trajectories successful (goal-reaching) episodes."""
        success_count = 0
        while success_count < n_trajectories:
            if self.sample_trajectory(len_traj=max_trajectory_length):
                success_count += 1

    # ------------------------------------------------------------------
    # Distribution computation
    # ------------------------------------------------------------------

    def compute_action_distributions(self):
        """Normalise action Counters into probability distributions.

        Populates:
            state_action_probs  : {discrete_state_key: {label: np.array(n_actions)}}
            state_label_counts  : {discrete_state_key: {label: int total samples}}
        """
        for state, label_counts in self.state_action_counts.items():
            self.state_action_probs[state] = {}
            self.state_label_counts[state] = {}

            for label, action_counter in label_counts:
                total = sum(action_counter.values())
                action_probs = np.zeros(self.n_actions)
                if total > 0:
                    for action, count in action_counter.items():
                        action_probs[action] = count / total
                self.state_action_probs[state][label] = action_probs
                self.state_label_counts[state][label] = total


# =============================================================================
# Training Functions
# =============================================================================
def create_environment(
    use_scenic: bool = False,
    use_sumo: bool = False,
    render_mode: str = None,
    enable_logging: bool = False,
    traci_label: str = 'default',
    rho: float = 0.2,
) -> AllWayStopEnvRM:
    """
    Create the complete RM-wrapped environment.

    Args:
        use_scenic: Whether to use Scenic with Newtonian simulator
        use_sumo: Whether to use SUMO/TraCI directly (recommended for FCFS intersection)
        render_mode: 'human' for visualization, None for training
        enable_logging: Whether to enable episode logging (True for eval/demo/debug, False for training)
        traci_label: TraCI connection label; pass distinct labels (e.g. 'train' and
            'eval') when two environments must coexist in the same Python process.
        rho: Policy update period [s] (= AllWayStopEnvRM.delta_t). Default 0.2 s
            (5 Hz), matching SUMO's default jmAllwayStopWait and the kt constant
            hard-coded in AllWayStopEnvRM.__init__.

    Returns:
        AllWayStopEnvRM environment ready for training
    """
    # Create base environment
    base_env = ScenicIntersectionEnv(
        use_sumo=use_sumo or use_scenic,  # Either flag enables SUMO
        use_scenic_only=use_scenic and not use_sumo,  # Scenic takes precedence only if explicitly requested
        render_mode=render_mode,
        traci_label=traci_label,
        policy_dt=rho,
    )

    # Load reward machine from stop_fcfs.txt
    # This RM has states for FCFS intersection with 'y' (yielding) transitions
    rm_path = os.path.join(os.path.dirname(__file__), 'rm_examples', 'stop_fcfs_rss_3.txt')

    if not os.path.exists(rm_path):
        # Fall back to allway_stop_fcfs.txt or fcfs.txt
        alt_paths = [
            os.path.join(os.path.dirname(__file__), 'rm_examples', 'allway_stop_fcfs.txt'),
            os.path.join(os.path.dirname(__file__), 'rm_examples', 'fcfs.txt'),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                rm_path = alt_path
                break

    rm = RewardMachine(rm_path)
    logger.info(f"Loaded RM from: {rm_path}")
    logger.info(f"RM states: {rm.get_states()}")

    # Resolve a_min_f from the route file.
    # v_S1 (type=DEFAULT_VEHTYPE, route=route_s2n, departPos=30) is the only
    # vehicle spawned on the same lane as the ego (departPos=10), making it the
    # guaranteed initially-closest vehicle in front.  Its 'decel' attribute is
    # the maximum braking deceleration the vehicle can apply, which maps directly
    # to |a_min_f| in the RSS formula:
    #   d_rss = ... − v_f² / (2·|a_min_f|)
    # (front-vehicle stopping distance credit; larger |a_min_f| → shorter credit
    # → larger required safety gap, reflecting a vehicle that can stop suddenly).
    _a_min_f = -4.5  # fallback: DEFAULT_VEHTYPE decel from intersection.rou.xml
    if hasattr(base_env, 'route_file') and os.path.exists(base_env.route_file):
        try:
            _rou_root = ET.parse(base_env.route_file).getroot()
            _v_s1_type = next(
                (v.get('type') for v in _rou_root.findall('vehicle')
                 if v.get('id') == 'v_S1'),
                'DEFAULT_VEHTYPE'
            )
            _type_decel = next(
                (float(t.get('decel', 4.5)) for t in _rou_root.findall('vType')
                 if t.get('id') == _v_s1_type),
                4.5
            )
            _a_min_f = -_type_decel
            logger.info(f"a_min_f: v_S1 type='{_v_s1_type}' decel={_type_decel} m/s² → a_min_f={_a_min_f}")
        except Exception as _e:
            logger.warning(f"Could not resolve a_min_f from route file: {_e}")

    # Wrap with RM
    env = AllWayStopEnvRM(base_env, rm, delta_t=rho, enable_logging=enable_logging, a_max=5.0, a_min=-5.0, a_min_c=-5.0, a_min_f=_a_min_f)

    return env


def train(
    algorithm: str = 'DQN',
    total_timesteps: int = 51200,
    use_scenic: bool = False,
    use_sumo: bool = False,
    save_path: str = None,
    tensorboard_log: str = None,
    seed: int = 0,
    rho: float = 0.2,
):
    """
    Train an RL agent on the all-way stop intersection.

    Args:
        algorithm: 'DQN' or 'PPO'
        total_timesteps: Number of training steps
        use_scenic: Whether to use Scenic with Newtonian simulator
        use_sumo: Whether to use SUMO/TraCI (recommended for FCFS)
        save_path: Path to save trained model
        tensorboard_log: Path for tensorboard logs
        seed: Random seed for reproducibility
    """
    if not HAS_SB3:
        logger.error("stable_baselines3 required for training!")
        return None

    # Create environment (disable logging during training)
    env = create_environment(use_scenic=use_scenic, use_sumo=use_sumo, render_mode=None, enable_logging=False, rho=rho)

    # Set up tensorboard logging
    if tensorboard_log is None:
        tensorboard_log = "./scenic_intersection_tensorboard/"

    # Create model
    if algorithm.upper() == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log=tensorboard_log,
            device="cpu",
            seed=seed,
        )
    else:  # DQN
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.5,
            target_update_interval=500,
            tensorboard_log=tensorboard_log,
            device="cpu",
            seed=seed,
        )

    # Training callback
    callback = TrainStatsCallback()

    # Train
    logger.info(f"Starting {algorithm} training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save model
    if save_path is None:
        save_path = f"{algorithm.lower()}_allway_stop_rm_{total_timesteps}"

    model.save(save_path)
    logger.info(f"Model saved to: {save_path}")

    env.close()

    return model


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    use_scenic: bool = False,
    use_sumo: bool = False,
    save_gif: bool = False,
    gif_duration: float = 12.0,
    log_fail_only: bool = False,
    no_log: bool = False,
    rho: float = 0.2,
):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes (default False — headless for batch eval)
        use_scenic: Whether to use Scenic with Newtonian simulator
        use_sumo: Whether to use SUMO/TraCI (recommended for FCFS)
        save_gif: Whether to save episode recordings as GIF files
        gif_duration: Duration in seconds to capture for GIF (default: 12.0)
        log_fail_only: When True, write log records ONLY for failed episodes
            (those ending without the goal event 'g').  Mutually exclusive with
            no_log.
        no_log: When True, suppress ALL log output during evaluation — no new
            records are written to log_scenic_intersection_rm.txt.  Mutually
            exclusive with log_fail_only.
    """
    if not HAS_SB3:
        logger.error("stable_baselines3 required for evaluation!")
        return

    # ── Logging mode setup ────────────────────────────────────────────────────
    # Three mutually exclusive modes (enforced by CLI mutual-exclusion group):
    #
    #   default       — FileHandler active; all records written to disk as usual.
    #   no_log        — FileHandler removed for the duration; no records written.
    #   log_fail_only — FileHandler replaced by a MemoryHandler that buffers
    #                   records in RAM and forwards them to the FileHandler only
    #                   on explicit flush() calls.  At each episode boundary we
    #                   either flush (failure) or clear (success) the buffer.
    #
    # In all cases the FileHandler is restored before this function returns.
    _mem_handler = None
    _file_handler_removed = None
    _root = logging.getLogger()

    if no_log or log_fail_only:
        # Step 1: locate and detach the FileHandler set up by basicConfig.
        _file_handler_removed = next(
            (h for h in _root.handlers if isinstance(h, logging.FileHandler)), None
        )
        if _file_handler_removed is not None:
            _root.removeHandler(_file_handler_removed)

        if log_fail_only and _file_handler_removed is not None:
            # Step 2 (log_fail_only only): install MemoryHandler in its place.
            import logging.handlers as _lh
            # capacity=10**6 → never auto-flushes on size;
            # flushLevel above CRITICAL → never auto-flushes on severity.
            _mem_handler = _lh.MemoryHandler(
                capacity=10 ** 6,
                flushLevel=logging.CRITICAL + 1,
                target=_file_handler_removed,
                flushOnClose=False,
            )
            _root.addHandler(_mem_handler)
        # For no_log: FileHandler simply stays detached — nothing added back.

    # Determine render mode
    # For SUMO: GIF saving requires 'human' mode (SUMO-GUI) to capture screenshots
    # For HighwayEnv: GIF saving works with 'rgb_array' mode
    if use_sumo or use_scenic:
        # SUMO/Scenic need GUI for screenshot capture
        if save_gif or render:
            render_mode = 'human'
        else:
            render_mode = None
    else:
        # HighwayEnv supports rgb_array for GIF capture
        if save_gif:
            render_mode = 'rgb_array'
        elif render:
            render_mode = 'human'
        else:
            render_mode = None

    env = create_environment(use_scenic=use_scenic, use_sumo=use_sumo, render_mode=render_mode, enable_logging=True, rho=rho)

    # Load model
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    else:
        model = DQN.load(model_path)

    from tqdm import tqdm

    # Unified evaluation counters (episode-level)
    results = {
        'n_total':      n_episodes,
        'n_collision':  0,   # episodes where ego collides
        'n_goal':       0,   # episodes: stopped + priority + goal + no collision
        'n_passed':     0,   # episodes where ego crossed stop line without stopping
        'n_stopped':    0,   # episodes where ego properly stopped before the line
        'n_yield_viol': 0,   # episodes where ego entered intersection without FCFS priority
        'sum_d_stop':   0.0, # sum of d_stop at first full stop (for avg computation)
        'total_rewards':    [],
        'episode_lengths':  [],
        # Metrics (populated after all episodes)
        'M_safe':  0.0,
        'M_stop':  0.0,
        'M_yield': 0.0,
        'M_goal':  0.0,
        'avg_d_stop': 0.0,
    }

    # Create output directory for GIFs
    if save_gif:
        gif_dir = os.path.join(os.path.dirname(__file__), 'eval_gifs')
        os.makedirs(gif_dir, exist_ok=True)
        logger.info(f"Saving GIFs to: {gif_dir}")
        logger.info(f"Capturing first {gif_duration} seconds of each episode")

    pbar = tqdm(range(n_episodes), desc="Evaluating")

    for episode in pbar:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        frames = []  # For GIF saving
        episode_start_time = 0.0  # Track simulation time for GIF capture

        # ── Clear log buffer at the start of every episode ────────────────────
        # Any records left from a previous failure have already been flushed;
        # start fresh so only THIS episode's records are in the buffer.
        if log_fail_only and _mem_handler is not None:
            _mem_handler.buffer.clear()

        # ── Episode-level tracking flags ──────────────────────────────────────
        episode_collision       = False  # any 'x' event or crashed flag
        episode_stopped         = False  # 's' event fired (stopped before stop line)
        episode_goal            = False  # 'g' event fired (cleared intersection)
        episode_passed          = False  # 'p' event fired (ran stop sign)
        # Priority / yield tracking
        episode_entered    = False  # ego crossed stop line (dist_s < 0) for first time
        episode_yield_viol = False  # entered intersection without FCFS priority
        # Distance-at-first-stop tracking
        stop_counter  = False   # latch: first full stop already recorded
        curr_d_stop   = None    # dist_to_stop_line at first valid stop; None = no stop occurred
        cor_d_stop = None
        d_stop_inc = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Extract state for metrics.
            # Use info['event'] set by step() — do NOT call L() again here.
            # A second L() call would see stale prev_dist_s (already updated) and
            # silently drop 'p'; it would also call vehicle_exited_intersection
            # twice on the 'g' step.
            state = env.st4obs(obs)
            event = info.get('event', '')

            # ── 1. Collision (M_safe) ─────────────────────────────────────────
            # Label 'x' or TraCI crashed flag — latch for the episode.
            if (event == 'x' or info.get('crashed', False)) \
                and not info.get('connection_lost', False):
                episode_collision = True

            # ── 2. Proper stop before stop line (M_stop) ──────────────────────
            # Label 's' fires the first time ego is at the stop line, is_stopped,
            # and has held the stop for at least MIN_STOP_DURATION (0.1 s).
            if event == 's' and not episode_stopped:
                episode_stopped = True
                #curr_d_stop = state['dist_to_stop_line'] # Stopping precision for correct stops at the stop sign
                cor_d_stop = state['dist_to_stop_line']

            # ── 3. Distance to stop line at first full stop ───────────────────
            # Latch the first step where speed falls below the stop threshold
            # (0.1 m/s, matching STOP_SPEED_THRESH) to record where the ego
            # first came to a halt relative to the stop line.
            if abs(state['v_e']) < 0.1 and not stop_counter:
                curr_d_stop = state['dist_to_stop_line'] # Stopping precision for ordinary stops
                stop_counter = True

            # ── 4. Passed stop line without stopping (M_stop numerator) ───────
            # Label 'p' fires when dist_to_stop_line crosses from ≥ 0 to < 0
            # without the ego having fully stopped first.
            if event == 'p':
                episode_passed = True

            # ── 5. Yield violation (M_yield) — public FCFSQueue check ─────────
            # At the exact step the ego first crosses the stop line
            # (dist_to_stop_line goes negative), query has_priority('ego')
            # directly from the FCFSQueue.  This is an independent check that
            # does not rely on the 'r' label returned by L(), avoiding any
            # circular dependency on the labeling function itself.
            # At this crossing step the ego's FCFS record still holds the
            # waiting-time accumulated while it was in-zone, so the priority
            # decision reflects the correct queue state at the moment of entry.
            if state['dist_to_stop_line'] < 0 and not episode_entered:
                episode_entered = True
                if not env.fcfs_queue.has_priority('ego'):
                    episode_yield_viol = True

            # ── 6. Goal reached (M_goal) ──────────────────────────────────────
            # Label 'g' fires when ego has fully cleared the intersection.
            if event == 'g':
                episode_goal = True

            # Get current simulation time
            sim_time = info.get('time', step_count * 0.1)  # Fallback to step-based time

            if render or save_gif:
                # Only capture frames for the first gif_duration seconds when saving GIF
                should_capture = save_gif and (sim_time - episode_start_time) <= gif_duration
                frame = env.render(capture_frame=should_capture)
                if should_capture and frame is not None:
                    frames.append(frame)

            done = done or truncated

        # ── Episode-end: accumulate into unified results dict ─────────────────
        results['total_rewards'].append(total_reward)
        results['episode_lengths'].append(step_count)
        # Only accumulate stop distance when a valid stop actually occurred.
        # curr_d_stop is None for episodes that ended via collision, timeout, or
        # ran the stop sign before stopping — excluding them keeps avg_d_stop
        # honest (it answers "how accurate was the stop?" not "what happened to
        # vehicles that never stopped?").  The failure itself is captured by
        # M_safe (collision) and M_stop (no stop before stop line).
        if curr_d_stop is not None:
            if cor_d_stop is not None:
                d_stop_inc = cor_d_stop
            else:
                d_stop_inc = curr_d_stop
            # results['sum_d_stop'] += d_stop_inc

        if episode_collision:
            results['n_collision'] += 1

        if episode_stopped:
            results['n_stopped'] += 1
            results['sum_d_stop'] += cor_d_stop

        # M_stop numerator: passed the stop line without ever stopping first
        if episode_passed and not episode_stopped:
            results['n_passed'] += 1

        # M_yield numerator: entered the intersection without FCFS priority
        if episode_yield_viol:
            results['n_yield_viol'] += 1

        # M_goal: properly stopped AND priority-compliant AND reached goal AND no collision
        if episode_goal and not episode_yield_viol and episode_stopped and not episode_collision:
            results['n_goal'] += 1

        # ── Log-fail-only: flush buffer on failure, discard on success ─────────
        # A failed episode is any episode that did NOT reach the goal event 'g'.
        # This covers: collision ('x'), ran stop sign ('p'), yield violation,
        # and timeout (truncated without 'g').
        if log_fail_only and _mem_handler is not None:
            episode_failed = not episode_goal
            if episode_failed:
                # Write a structured header so the failure is easy to find in the log.
                logger.info(
                    "FAIL ep %d/%d | collision=%s | passed_no_stop=%s | "
                    "yield_viol=%s | goal=%s | steps=%d | reward=%.2f",
                    episode + 1, n_episodes,
                    episode_collision, episode_passed,
                    episode_yield_viol, episode_goal,
                    step_count, total_reward,
                )
                _mem_handler.flush()  # write all buffered records (+ header) to file
            else:
                _mem_handler.buffer.clear()  # successful episode — discard silently

        # ── Real-time progress bar postfix ────────────────────────────────────
        ep_done   = episode + 1           # episodes completed so far
        n_col     = results['n_collision']
        n_pass    = results['n_passed']
        n_yv      = results['n_yield_viol']
        n_gl      = results['n_goal']
        n_valid   = results['n_stopped']   # episodes with a valid recorded stop distance
        avg_ds    = results['sum_d_stop'] / n_valid if n_valid > 0 else float('nan') # Strict avg_ds
        # avg_ds = results['sum_d_stop'] / (ep_done-n_pass) if n_valid > 0 else float('nan') # Relaxed avg_ds
        pbar.set_postfix({
            'M_goal':  f"{100 * n_gl  / ep_done:.1f}%",
            'M_safe':  f"{100 * (1 - n_col / ep_done):.1f}%",
            # 'M_stop':  f"{100 * (1 - n_pass / ep_done):.1f}%",
            'M_stop':  f"{100 * (n_valid / ep_done):.1f}%",
            'M_yield': f"{100 * (1 - n_yv  / ep_done):.1f}%",
            'avg_ds':  f"{avg_ds:.2f}m",
        })

        # Debug: report frame count and duration
        if save_gif:
            actual_duration = min(sim_time - episode_start_time, gif_duration)
            logger.info(f"Episode {episode + 1}: captured {len(frames)} frames ({actual_duration:.1f}s)")

        # Save GIF for this episode
        if save_gif and frames:
            try:
                import imageio.v2 as imageio
                gif_path = os.path.join(gif_dir, f'episode_{episode + 1}.gif')
                fps = 10  # 0.1 s SUMO step → 10 fps = real-time playback
                imageio.mimsave(gif_path, frames, fps=fps)
                logger.info(f"Saved GIF: {gif_path} ({len(frames)} frames @ {fps}fps)")
            except ImportError:
                logger.warning("imageio not installed. Install with: pip install imageio")
                save_gif = False
            except Exception as e:
                logger.warning(f"Failed to save GIF: {e}")

    # ── Compute final metrics ─────────────────────────────────────────────────
    N          = n_episodes
    n_col      = results['n_collision']
    n_pass     = results['n_passed']
    n_yv       = results['n_yield_viol']
    n_gl       = results['n_goal']
    n_valid    = results['n_stopped']  # episodes with a valid recorded stop distance
    avg_d_stop = results['sum_d_stop'] / n_valid if n_valid > 0 else float('nan') # Strict avg_ds
    # avg_d_stop = results['sum_d_stop'] / (N - n_pass) if n_valid > 0 else float('nan') # Relaxed avg_ds

    results['M_safe']     = 100.0 * (1 - n_col  / N)
    # results['M_stop']     = 100.0 * (1 - n_pass / N)
    results['M_stop']     = 100.0 * (n_valid / N)
    results['M_yield']    = 100.0 * (1 - n_yv   / N)
    results['M_goal']     = 100.0 *      n_gl   / N
    results['avg_d_stop'] = avg_d_stop

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total episodes          : {N}")
    print(f"M_goal  (goal reached)  : {results['M_goal']:.2f}%")
    print(f"M_safe  (no collision)  : {results['M_safe']:.2f}%")
    print(f"M_stop  (stopped first) : {results['M_stop']:.2f}%")
    print(f"M_yield (FCFS compliant): {results['M_yield']:.2f}%")
    print(f"Avg. d_stop             : {avg_d_stop:.2f} m  (over {n_valid}/{N} eps w/ valid stop)")
    print("-" * 60)
    print(f"  Collisions            : {n_col}/{N}")
    print(f"  Passed w/o stop       : {n_pass}/{N}")
    print(f"  Yield violations      : {n_yv}/{N}")
    print(f"  Goal completions      : {n_gl}/{N}")
    print("=" * 60)

    env.close()

    # ── Restore original logging handler ──────────────────────────────────────
    if no_log or log_fail_only:
        if _mem_handler is not None:
            _mem_handler.buffer.clear()   # discard any trailing unflushed records
            _mem_handler.close()          # flushOnClose=False, so only closes target
            _root.removeHandler(_mem_handler)
        if _file_handler_removed is not None:
            _root.addHandler(_file_handler_removed)  # restore for subsequent code

    return results


def debug_fixed_actions(
    n_episodes: int = 5,
    render: bool = True,
    use_scenic: bool = False,
    use_sumo: bool = False,
    action_sequence: Optional[list] = None,
    use_policy: bool = True,
    rho: float = 0.2,
):
    """
    Debug mode: Run fixed action sequence without trained model for labeling function debugging.

    Args:
        n_episodes: Number of episodes to run
        render: Whether to render episodes
        use_scenic: Whether to use Scenic with Newtonian simulator
        use_sumo: Whether to use SUMO/TraCI (recommended for FCFS)
        action_sequence: List of actions to repeat. If None, uses default policy.
        use_policy: If True, uses state-based policy instead of fixed sequence.
    """
    print("\n" + "=" * 80)
    print("DEBUG MODE: Fixed Action Sequence (Labeling Function Debugging)")
    print("=" * 80)

    # Action space: 0=SLOWER (decelerate), 1=IDLE (maintain), 2=FASTER (accelerate)
    if use_policy:
        print("Using state-based policy:")
        print("  - Distance > 25m: IDLE (maintain speed)")
        print("  - 25m >= Distance > 5m: SLOWER (decelerate to stop)")
        print("  - Distance <= 5m, speed > 0.5: SLOWER (ensure full stop)")
        print("  - Distance <= 5m, speed <= 0.5: FASTER (proceed after stop)")
        action_sequence = None  # Will be computed dynamically
    elif action_sequence is None:
        # Simple sequence: accelerate for a while, then maintain speed
        action_sequence = [2] * 20 + [1] * 10  # Accelerate (action=2) then idle (action=1)
        print(f"Using default action sequence: accelerate x20, idle x10")
        print(f"Action mapping: 0=SLOWER, 1=IDLE, 2=FASTER")
    else:
        print(f"Using custom action sequence: {action_sequence}")

    render_mode = 'human' if render else None
    env = create_environment(use_scenic=use_scenic, use_sumo=use_sumo, render_mode=render_mode, enable_logging=True, rho=rho)

    results = {
        'completions': 0,
        'collisions': 0,
        'violations': 0,
        'timeouts': 0,
        'total_rewards': [],
        'episode_lengths': []
    }

    for episode in range(n_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*80}")

        _, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        action_idx = 0

        # Log initial state
        logger.info(f"\n{'='*80}")
        logger.info(f"EPISODE {episode + 1} START")
        logger.info(f"{'='*80}")

        while not done:
            # Extract detailed state information BEFORE action selection (for policy)
            # Get physical state from base environment
            if hasattr(env, 'env'):
                base_env = env.env
            else:
                base_env = env

            phys_state = {}
            if hasattr(base_env, 'ego_vehicle_id'):
                try:
                    # SUMO-specific state extraction
                    ego_pos = base_env.traci.vehicle.getPosition(base_env.ego_vehicle_id)
                    ego_speed = base_env.traci.vehicle.getSpeed(base_env.ego_vehicle_id)

                    # Calculate distance to stop line (assuming intersection center at 100, 100)
                    dist_to_center = np.sqrt((ego_pos[0] - 100.0)**2 + (ego_pos[1] - 100.0)**2)
                    dist_to_stop_line = max(0, dist_to_center - 5.0)  # Stop line ~5m from center

                    # Get front vehicle info
                    leader_info = base_env.traci.vehicle.getLeader(base_env.ego_vehicle_id, 100.0)
                    if leader_info:
                        front_id, gap = leader_info
                        front_speed = base_env.traci.vehicle.getSpeed(front_id)
                        d_ego_to_front = gap
                        v_front = front_speed
                    else:
                        d_ego_to_front = 999.0
                        v_front = 0.0

                    phys_state = {
                        'x_e': ego_pos[0],
                        'y_e': ego_pos[1],
                        'v_e': ego_speed,
                        'dist_to_stop_line': dist_to_stop_line,
                        'd_ego_to_front': d_ego_to_front,
                        'v_front': v_front
                    }
                except Exception as e:
                    logger.debug(f"Error extracting SUMO state: {e}")
                    phys_state = {
                        'x_e': 0.0,
                        'y_e': 0.0,
                        'v_e': 0.0,
                        'dist_to_stop_line': 0.0,
                        'd_ego_to_front': 999.0,
                        'v_front': 0.0
                    }
            else:
                # Fallback for non-SUMO environments
                phys_state = {
                    'x_e': 0.0,
                    'y_e': 0.0,
                    'v_e': 0.0,
                    'dist_to_stop_line': 0.0,
                    'd_ego_to_front': 999.0,
                    'v_front': 0.0
                }

            # Select action based on policy or sequence
            if use_policy:
                # State-based policy:
                # - Distance > 25m, speed > 0: IDLE (maintain speed)
                # - 25m >= Distance > 5m, speed > 1.0: SLOWER (decelerate toward stop line)
                # - Distance > 5m, speed < 0.1: FASTER (stuck before stop line, resume approach)
                # - Distance <= 5m, speed > 0.5: SLOWER (ensure full stop at stop line)
                # - Distance <= 5m, speed <= 0.5: FASTER (proceed after full stop)
                dist = phys_state['dist_to_stop_line']
                speed = phys_state['v_e']

                if speed < 0.1 and dist > 5.0:
                    # Vehicle stopped too early, need to resume moving toward stop line
                    action = 2  # FASTER - approach stop line
                elif dist > 25.0:
                    action = 1  # IDLE - maintain speed while far away
                elif dist > 5.0 and speed > 1.0:
                    action = 0  # SLOWER - decelerate to approach stop line
                elif dist > 5.0 and speed <= 1.0:
                    action = 1  # IDLE - coast toward stop line at low speed
                elif speed > 0.5:
                    action = 0  # SLOWER - ensure complete stop at stop line
                else:
                    action = 2  # FASTER - proceed through intersection after full stop
            else:
                # Use fixed action sequence
                action = action_sequence[action_idx % len(action_sequence)]
                action_idx += 1

            # Take action
            _, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Get event labels (true_props) from info if available
            true_props = info.get('events', set())
            if isinstance(true_props, set):
                true_props = ','.join(sorted(true_props)) if true_props else 'none'

            # Get RM state
            rm_state = env.current_u_id if hasattr(env, 'current_u_id') else 'N/A'

            # Log detailed state
            log_message = (
                f"T={info.get('time', step_count * 0.1):.1f} | "
                f"Action: {action} | "
                f"X: {phys_state['x_e']:.1f} | Y: {phys_state['y_e']:.1f} | "
                f"d_e2s: {phys_state['dist_to_stop_line']:.1f} | "
                f"d_e2f: {phys_state['d_ego_to_front']:.1f} | "
                f"v_f: {phys_state['v_front']:.2f} | v_e: {phys_state['v_e']:.2f} | "
                f"Evt: {true_props} | RM State: {rm_state} | "
                f"r: {reward:.2f} | done={done}"
            )
            logger.info(log_message)

            if render:
                env.render()

            done = done or truncated

            # Safety: prevent infinite loops
            if step_count > 1000:
                logger.warning("Episode exceeded 1000 steps, terminating")
                break

        # Episode summary
        rm_state = env.current_u_id if hasattr(env, 'current_u_id') else 'N/A'
        logger.info(f"\n{'='*80}")
        logger.info(f"EPISODE {episode + 1} END")
        logger.info(f"Total Reward: {total_reward:.2f} | Steps: {step_count} | Final RM State: {rm_state}")
        logger.info(f"{'='*80}\n")

        # Categorize outcome
        if hasattr(env, 'current_u_id'):
            if env.current_u_id == 7:  # COMPLETED
                results['completions'] += 1
            elif env.current_u_id == 8:  # VIOLATION
                if info.get('crashed', False):
                    results['collisions'] += 1
                else:
                    results['violations'] += 1
            else:
                results['timeouts'] += 1
        else:
            results['timeouts'] += 1

        results['total_rewards'].append(total_reward)
        results['episode_lengths'].append(step_count)

        print(f"Episode {episode + 1} Summary: reward={total_reward:.2f}, steps={step_count}, rm_state={rm_state}")

    # Print summary
    print("\n" + "=" * 80)
    print("DEBUG MODE RESULTS")
    print("=" * 80)
    print(f"Episodes: {n_episodes}")
    print(f"Completions: {results['completions']} ({100*results['completions']/n_episodes:.1f}%)")
    print(f"Collisions: {results['collisions']} ({100*results['collisions']/n_episodes:.1f}%)")
    print(f"Violations: {results['violations']} ({100*results['violations']/n_episodes:.1f}%)")
    print(f"Timeouts: {results['timeouts']} ({100*results['timeouts']/n_episodes:.1f}%)")
    print(f"Avg Reward: {np.mean(results['total_rewards']):.2f} ± {np.std(results['total_rewards']):.2f}")
    print(f"Avg Length: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
    print("=" * 80)
    print(f"\nDetailed logs saved to: log_scenic_intersection_rm.txt")

    env.close()

    return results


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train/Evaluate RL agent on All-Way Stop Intersection with Reward Machine"
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'eval', 'demo', 'debug', 'sim'],
        help="Mode: train, eval, demo, debug, or sim (simulate to collect action distributions)"
    )
    parser.add_argument(
        '--algorithm', type=str, default='DQN',
        choices=['DQN', 'PPO'],
        help="RL algorithm to use"
    )
    parser.add_argument(
        '--timesteps', type=int, default=50000,
        help="Total training timesteps"
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help="Model path for evaluation"
    )
    parser.add_argument(
        '--episodes', type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        '--scenic', action='store_true',
        help="Use Scenic with Newtonian simulator"
    )
    parser.add_argument(
        '--sumo', action='store_true',
        help="Use SUMO/TraCI directly (recommended for intersection with FCFS)"
    )
    parser.add_argument(
        '--render', action='store_true',
        help="Render during evaluation"
    )
    parser.add_argument(
        '--save-gif', action='store_true',
        help="Save evaluation episodes as GIF files (requires imageio)"
    )
    parser.add_argument(
        '--gif-duration', type=float, default=12.0,
        help="Duration in seconds to capture for each GIF (default: 12.0)"
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help="Random seed for reproducibility"
    )
    _log_group = parser.add_mutually_exclusive_group()
    _log_group.add_argument(
        '--log-fail-only', action='store_true',
        help=(
            "Evaluation only: suppress log output for successful episodes. "
            "Only failed episodes (ending without goal event 'g', e.g. collision, "
            "timeout, ran stop sign) are written to log_scenic_intersection_rm.txt. "
            "Mutually exclusive with --no-log."
        )
    )
    _log_group.add_argument(
        '--no-log', action='store_true',
        help=(
            "Evaluation only: suppress ALL logging during evaluation — no new "
            "records are written to log_scenic_intersection_rm.txt for any episode. "
            "Mutually exclusive with --log-fail-only."
        )
    )
    parser.add_argument(
        '--trajectories', type=int, default=10000,
        help="Sim mode: number of trajectories to sample (default: 10000)"
    )
    parser.add_argument(
        '--max-len', type=int, default=500,
        help="Sim mode: maximum trajectory length in steps (default: 500, matching "
             "ScenicIntersectionEnv.max_episode_steps; at dt=0.1s/step this is 50s "
             "of simulation time — the same hard cap used by eval)"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Sim mode: output pickle path (default: ./objects/sim_allway_stop_<n>traj.pkl)"
    )
    parser.add_argument(
        '--rho', type=float, default=0.2,
        help="Policy update period [s] (= AllWayStopEnvRM.delta_t). Default: 0.2"
    )

    args = parser.parse_args()

    # Determine which backend will be used
    if args.sumo:
        if HAS_TRACI:
            backend = "SUMO/TraCI (with FCFS intersection)"
        else:
            backend = "HighwayEnv (TraCI not available)"
    elif args.scenic:
        if HAS_SCENIC:
            backend = "Scenic/Newtonian"
        elif HAS_TRACI:
            backend = "SUMO/TraCI (Scenic not installed)"
        else:
            backend = "HighwayEnv (fallback)"
    else:
        backend = "HighwayEnv"

    # Determine use_scenic and use_sumo flags
    use_scenic_flag = args.scenic
    use_sumo_flag = args.sumo

    if args.mode == 'train':
        print(f"\n{'='*60}")
        print(f"Training {args.algorithm} on All-Way Stop Intersection")
        print(f"Timesteps: {args.timesteps}")
        print(f"Seed:      {args.seed}")
        print(f"Backend: {backend}")
        print(f"  HAS_SCENIC: {HAS_SCENIC}")
        print(f"  HAS_TRACI: {HAS_TRACI}")
        print(f"{'='*60}\n")

        model = train(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            use_scenic=use_scenic_flag,
            use_sumo=use_sumo_flag,
            seed=args.seed,
            rho=args.rho,
        )

    elif args.mode == 'eval':
        if args.model is None:
            args.model = f"{args.algorithm.lower()}_allway_stop_rm_{args.timesteps}"

        print(f"\n{'='*60}")
        print(f"Evaluating model: {args.model}")
        print(f"Episodes: {args.episodes}")
        print(f"Backend: {backend}")
        print(f"Save GIF: {args.save_gif}")
        if args.save_gif:
            print(f"GIF Duration: {args.gif_duration}s")
        print(f"Log fail-only: {args.log_fail_only}")
        print(f"No log:        {args.no_log}")
        print(f"{'='*60}\n")

        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            render=args.render,
            use_scenic=use_scenic_flag,
            use_sumo=use_sumo_flag,
            save_gif=args.save_gif,
            gif_duration=args.gif_duration,
            log_fail_only=args.log_fail_only,
            no_log=args.no_log,
            rho=args.rho,
        )

    elif args.mode == 'demo':
        # Quick demo with rendering
        print("\n" + "=" * 60)
        print("Demo: All-Way Stop Intersection Environment")
        print(f"Backend: {backend}")
        print("=" * 60 + "\n")

        env = create_environment(use_scenic=use_scenic_flag, use_sumo=use_sumo_flag, render_mode='human', enable_logging=True, rho=args.rho)

        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Random action for demo
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            done = done or truncated

        print(f"\nDemo episode finished. Total reward: {total_reward:.2f}")
        print(f"Final RM state: {env.current_u_id}")

        env.close()

    elif args.mode == 'debug':
        # Debug mode: fixed actions for labeling function debugging
        print(f"\n{'='*60}")
        print("Debug Mode: Fixed Action Sequence")
        print(f"Episodes: {args.episodes}")
        print(f"Backend: {backend}")
        print(f"{'='*60}\n")

        debug_fixed_actions(
            n_episodes=args.episodes,
            render=args.render,
            use_scenic=use_scenic_flag,
            use_sumo=use_sumo_flag,
            action_sequence=None,  # Uses default policy
            use_policy=True,  # Use state-based policy by default
            rho=args.rho,
        )

    elif args.mode == 'sim':
        if args.model is None:
            args.model = f"{args.algorithm.lower()}_allway_stop_rm_{args.timesteps}"

        output_path = args.output or f"./objects/sim_allway_stop_{args.trajectories}traj.pkl"

        print(f"\n{'='*60}")
        print(f"Sim Mode: AllWayStop Action Distribution Collection")
        print(f"Model:        {args.model}")
        print(f"Trajectories: {args.trajectories}")
        print(f"Max length:   {args.max_len} steps")
        print(f"Output:       {output_path}")
        print(f"Backend:      {backend}")
        print(f"{'='*60}\n")

        # Load policy — detect algorithm from model name, consistent with evaluate().
        if 'ppo' in args.model.lower():
            policy = PPO.load(args.model, device='cpu')
        else:
            policy = DQN.load(args.model, device='cpu')

        # Create environment (no render, no logging overhead)
        env = create_environment(use_scenic=use_scenic_flag, use_sumo=use_sumo_flag, enable_logging=False, rho=args.rho)

        sim = AllWayStopSimulator(env, policy)

        import time
        import pickle
        from tqdm import tqdm

        t0 = time.time()
        attempts = 0
        with tqdm(total=args.trajectories, desc="Sampling trajectories") as pbar:
            while pbar.n < args.trajectories:
                success = sim.sample_trajectory(len_traj=args.max_len, render=False)
                attempts += 1
                if success:
                    pbar.update(1)

        sim.compute_action_distributions()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        sim.policy = None  # Drop policy before pickling (not serialisable)
        sim.env = None     # Drop env before pickling (traci module is not serialisable)
        with open(output_path, 'wb') as f:
            pickle.dump(sim, f)

        elapsed = time.time() - t0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        success_rate = 100.0 * args.trajectories / attempts if attempts > 0 else 0.0
        print(f"\nDone. {args.trajectories} successful trajectories ({attempts} attempts, "
              f"{success_rate:.1f}% success rate) in {hours}h {minutes}m. Saved to {output_path}.")
        env.close()
