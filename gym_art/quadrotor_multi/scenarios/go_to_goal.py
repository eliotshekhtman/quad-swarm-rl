import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from gym_art.quadrotor_multi.scenarios.utils import get_z_value


class Scenario_go_to_goal(QuadrotorScenario):
    """Send each quadrotor to one fixed goal for the entire episode."""

    def _get_point(self):
        box_size = self.envs[0].box
        x, y = np.random.uniform(
            low=-box_size * 1.5,
            high=box_size * 1.5,
            size=(2,),
        )
        z = get_z_value(
            num_agents=self.num_agents,
            num_agents_per_layer=self.num_agents_per_layer,
            box_size=box_size,
            formation=self.formation,
            formation_size=self.formation_size,
        )
        return np.array([x, y, z])

    def step(self):
        return

    def reset(self, *_args, **_kwargs):
        self.update_formation_and_relate_param()
        self.formation_center = np.array([0.0, 0.0, 2.0], dtype=np.float64)

        if self.num_agents == 1:
            self.goals = np.array([[2.0, 2.0, 1.0]], dtype=np.float64)
        else:
            self.goals = np.stack(
                [self._get_point() for _ in range(self.num_agents)],
                axis=0,
            )
