import numpy as np

from gym_art.quadrotor_multi.scenarios.utils import get_z_value
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_patrol_dual_goal(QuadrotorScenario):
    """
    Each agent patrols between two waypoints that are distributed evenly across the playable area.
    The waypoints remain static for the duration of an episode.
    """

    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.goal_pairs = None  # Shape: (num_agents, 2, 3)
        self.active_goal_index = None
        self.steps_since_switch = None
        self.switch_radius = self.approch_goal_metric
        self.min_start_distance = 1.5
        self.min_goal_distance = 1.0
        self.max_sampler_attempts = 1000
        self.start_points = None
        self.spawn_points = None
        # Small dwell window prevents rapid toggling when hovering on the boundary
        self.min_switch_interval = int(0.2 * self.envs[0].control_freq)
        self.margin = 1.
        self.spawn_box = [
            [self.margin - room_dims[0] / 2., room_dims[0] / 2. - self.margin],
            [self.margin - room_dims[1] / 2., room_dims[1] / 2. - self.margin],
            [self.margin + 1., room_dims[2] - self.margin]]

    def _is_far_enough(self, candidate: np.ndarray, points: list, min_dist: float) -> bool:
        for point in points:
            if np.linalg.norm(candidate - point) < min_dist:
                return False
        return True

    def _sample_point(self, existing_points: list, min_distance: float) -> np.ndarray:
        scales = [1.0, 0.7, 0.7]
        for scale in scales:
            threshold = min_distance * scale
            for _ in range(self.max_sampler_attempts):
                candidate = self._get_point()
                if self._is_far_enough(candidate, existing_points, threshold):
                    return candidate
        # fallback: keep trying without a minimum-distance constraint
        for _ in range(self.max_sampler_attempts):
            candidate = self._get_point()
            # as a final fallback, do not block episode setup on cross-point constraints
            return candidate
        return self._get_point()

    def _sample_goal_point(self, existing_goal_points: list, start_point: np.ndarray) -> np.ndarray:
        scales = [1.0, 0.7, 0.7]
        for scale in scales:
            threshold = self.min_goal_distance * scale
            for _ in range(self.max_sampler_attempts):
                candidate = self._get_point()
                if np.linalg.norm(candidate - start_point) <= self.switch_radius * 5:
                    continue
                if not self._is_far_enough(candidate, existing_goal_points, threshold):
                    continue
                return candidate

        # fallback: preserve the hard pairwise separation with this agent's start point only
        for _ in range(self.max_sampler_attempts):
            candidate = self._get_point()
            if np.linalg.norm(candidate - start_point) > self.switch_radius * 5:
                return candidate

        return self._get_point()

    def _get_point(self):
        x = np.random.uniform(low=self.spawn_box[0][0], high=self.spawn_box[0][1])
        y = np.random.uniform(low=self.spawn_box[1][0], high=self.spawn_box[1][1])
        z = np.random.uniform(low=self.spawn_box[2][0], high=self.spawn_box[2][1])
        return np.array([x, y, z])

    def _generate_patrol_pairs(self):
        """Random endpoints for patrol pairs."""
        goal_pairs = np.zeros((self.num_agents, 2, 3), dtype=np.float64)
        if self.num_agents == 1:  # Special case: go back and forth around (0,0)
            goal_pairs[0, 0] = np.array([2, 2, 1])
            goal_pairs[0, 1] = np.array([-2, -2, 1])
            self.start_points = np.array([goal_pairs[0, 0]], dtype=np.float64)
            self.goal_pairs = goal_pairs
        else:
            start_points = []
            existing_goal_points = []
            for i in range(self.num_agents):
                start_point = self._sample_point(start_points, self.min_start_distance)
                goal_pairs[i, 0] = start_point
                start_points.append(start_point)
                existing_goal_points.append(start_point)

                end_point = self._sample_goal_point(existing_goal_points, start_point)
                goal_pairs[i, 1] = end_point
                existing_goal_points.append(end_point)
            self.start_points = np.array(start_points, dtype=np.float64)
            self.goal_pairs = goal_pairs
            self.spawn_points = np.array(self.start_points, copy=True)

    def _activate_goals(self):
        self.active_goal_index = np.zeros(self.num_agents, dtype=np.int64)
        self.steps_since_switch = np.zeros(self.num_agents, dtype=np.int64)
        self.goals = np.array([self.goal_pairs[i, 0] for i in range(self.num_agents)], dtype=np.float64)
        self.spawn_points = np.array(self.goals, copy=True)
        if self.start_points is not None:
            self.spawn_points = np.array(self.start_points, copy=True)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal.copy()

    def step(self):
        for idx, env in enumerate(self.envs):
            self.steps_since_switch[idx] += 1
            # Which goal is quad idx going towards?
            active = self.active_goal_index[idx]
            target = self.goal_pairs[idx, active]

            # If within switch_radius and min_switch_interval time has passed,
            # switch to the other goal.
            dist = np.linalg.norm(env.dynamics.pos - target)
            if dist <= self.switch_radius and self.steps_since_switch[idx] >= self.min_switch_interval:
                self.active_goal_index[idx] = 1 - active
                self.steps_since_switch[idx] = 0
                new_target = self.goal_pairs[idx, self.active_goal_index[idx]]
                env.goal = new_target.copy()

        self.goals = np.array([env.goal for env in self.envs], dtype=np.float64)

    def reset(self, *_args, **_kwargs):
        self.update_formation_and_relate_param()
        self.formation_center = np.array([0.0, 0.0, 2.0], dtype=np.float64)

        self._generate_patrol_pairs()
        self._activate_goals()
        for env in self.envs:
            env.box = 0.1
