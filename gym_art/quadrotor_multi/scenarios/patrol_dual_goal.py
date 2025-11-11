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
        # Small dwell window prevents rapid toggling when hovering on the boundary
        self.min_switch_interval = int(0.2 * self.envs[0].control_freq)

    def _get_point(self):
        box_size = self.envs[0].box
        # Slightly unsafe to just multiply box_size by 1.5 but in practice it's 2
        # and the actual environment size is -4 to 4 for x and y
        x, y = np.random.uniform(low=-box_size * 1.5, high=box_size * 1.5, size=(2,))
        # Get z value, and make sure all goals will above the ground
        z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                        box_size=box_size, formation=self.formation, formation_size=self.formation_size)
        return np.array([x, y, z])

    def _generate_patrol_pairs(self):
        """Random endpoints for patrol pairs."""
        goal_pairs = np.zeros((self.num_agents, 2, 3), dtype=np.float64)
        if self.num_agents == 1: # Special case: go back and forth around (0,0)
            goal_pairs[0, 0] = np.array([2, 2, 1])
            goal_pairs[0, 1] = np.array([-2, -2, 1])
            self.goal_pairs = goal_pairs
        else: 
            for i in range(self.num_agents):
                goal_pairs[i, 0] = self._get_point() 
                end_point = self._get_point()
                # Ensure end point isn't too close to starting point
                while np.linalg.norm(goal_pairs[i, 0] - end_point) <= self.switch_radius * 5:
                    end_point = self._get_point()
                goal_pairs[i, 1] = end_point
            self.goal_pairs = goal_pairs

    def _activate_goals(self):
        self.active_goal_index = np.zeros(self.num_agents, dtype=np.int64)
        self.steps_since_switch = np.zeros(self.num_agents, dtype=np.int64)
        self.goals = np.array([self.goal_pairs[i, 0] for i in range(self.num_agents)], dtype=np.float64)
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
