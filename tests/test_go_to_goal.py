import argparse
import unittest
from types import SimpleNamespace

import numpy as np

from gym_art.quadrotor_multi.scenarios.go_to_goal import Scenario_go_to_goal
from gym_art.quadrotor_multi.scenarios.mix import create_scenario
from gym_art.quadrotor_multi.scenarios.patrol_dual_goal import (
    Scenario_patrol_dual_goal,
)
from gym_art.quadrotor_multi.scenarios.utils import (
    QUADS_MODE_LIST,
    QUADS_MODE_LIST_SINGLE,
)
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args
from project_utils.conformal_utils import _agent_goals


class _FakeQuad:
    def __init__(self):
        self.box = 2.0
        self.control_freq = 100
        self.goal = np.zeros(3, dtype=np.float64)
        self.dynamics = SimpleNamespace(pos=np.zeros(3, dtype=np.float64))


class GoToGoalScenarioTests(unittest.TestCase):
    @staticmethod
    def _scenario(num_agents=6):
        envs = [_FakeQuad() for _ in range(num_agents)]
        return create_scenario(
            "go_to_goal",
            envs=envs,
            num_agents=num_agents,
            room_dims=np.array([10.0, 10.0, 10.0]),
        )

    def test_factory_constructs_registered_scenario(self):
        scenario = self._scenario()
        self.assertIsInstance(scenario, Scenario_go_to_goal)

    def test_environment_parser_accepts_mode(self):
        parser = argparse.ArgumentParser()
        add_quadrotors_env_args("quadrotor_multi", parser)
        args = parser.parse_args(["--quads_mode", "go_to_goal"])
        self.assertEqual(args.quads_mode, "go_to_goal")

    def test_mode_is_not_added_to_random_training_mix(self):
        self.assertNotIn("go_to_goal", QUADS_MODE_LIST)
        self.assertNotIn("go_to_goal", QUADS_MODE_LIST_SINGLE)

    def test_reset_is_reproducible_and_goals_are_valid(self):
        np.random.seed(17)
        first = self._scenario()
        first.reset()

        np.random.seed(17)
        second = self._scenario()
        second.reset()

        self.assertEqual(first.goals.shape, (6, 3))
        self.assertTrue(np.isfinite(first.goals).all())
        self.assertTrue(np.all(np.abs(first.goals[:, :2]) <= 3.0))
        self.assertTrue(np.all(first.goals[:, 2] >= 0.25))
        np.testing.assert_array_equal(first.goals, second.goals)

    def test_first_goal_matches_patrol_distribution_for_same_seed(self):
        np.random.seed(31)
        go_to_goal = self._scenario()
        go_to_goal.reset()

        np.random.seed(31)
        patrol_envs = [_FakeQuad() for _ in range(6)]
        patrol = Scenario_patrol_dual_goal(
            "patrol_dual_goal",
            envs=patrol_envs,
            num_agents=6,
            room_dims=np.array([10.0, 10.0, 10.0]),
        )
        patrol.reset()

        np.testing.assert_array_equal(go_to_goal.goals[0], patrol.goal_pairs[0, 0])

    def test_goals_never_switch(self):
        np.random.seed(23)
        scenario = self._scenario()
        scenario.reset()
        initial_goals = scenario.goals.copy()

        for _ in range(1_000):
            scenario.step()

        np.testing.assert_array_equal(scenario.goals, initial_goals)
        self.assertFalse(hasattr(scenario, "goal_pairs"))
        self.assertFalse(hasattr(scenario, "active_goal_index"))

    def test_generic_goal_reader_reports_no_static_goal_changes(self):
        np.random.seed(23)
        scenario = self._scenario()
        scenario.reset()
        for env, goal in zip(scenario.envs, scenario.goals):
            env.goal = goal.copy()
        wrapped = SimpleNamespace(
            unwrapped=SimpleNamespace(envs=scenario.envs)
        )
        initial_goals = _agent_goals(wrapped)

        scenario.step()

        current_goals = _agent_goals(wrapped)
        for initial, current in zip(initial_goals, current_goals):
            np.testing.assert_array_equal(initial, current)

    def test_generic_goal_reader_still_detects_patrol_switches(self):
        np.random.seed(29)
        envs = [_FakeQuad() for _ in range(2)]
        scenario = Scenario_patrol_dual_goal(
            "patrol_dual_goal",
            envs=envs,
            num_agents=2,
            room_dims=np.array([10.0, 10.0, 10.0]),
        )
        scenario.reset()
        wrapped = SimpleNamespace(
            unwrapped=SimpleNamespace(envs=scenario.envs)
        )
        initial_goals = _agent_goals(wrapped)
        scenario.steps_since_switch[:] = scenario.min_switch_interval
        for agent_id, env in enumerate(envs):
            env.dynamics.pos = initial_goals[agent_id].copy()

        scenario.step()

        switched_goals = _agent_goals(wrapped)
        for initial, switched in zip(initial_goals, switched_goals):
            self.assertFalse(np.array_equal(initial, switched))

    def test_single_agent_keeps_patrol_first_goal(self):
        scenario = self._scenario(num_agents=1)
        scenario.reset()
        np.testing.assert_array_equal(scenario.goals, [[2.0, 2.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
