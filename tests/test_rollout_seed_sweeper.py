import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from rollout_seed_sweeper import (
    _paired_metrics,
    _proximity,
    _qualification_flags,
    _rank_summaries,
    parse_args,
)


class SeedSweeperMetricTests(unittest.TestCase):
    def test_proximity_is_time_aligned(self):
        ego = np.zeros((3, 3))
        others = np.array(
            [
                [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[1.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
            ]
        )

        result = _proximity(ego, others, threshold=0.75)

        self.assertEqual(result["min_distance"], 0.25)
        self.assertEqual(result["closest_agent"], 1)
        self.assertEqual(result["closest_step"], 2)
        self.assertAlmostEqual(result["close_fraction"], 2.0 / 3.0)

    def test_paired_metrics_separate_ego_and_other_agent_response(self):
        baseline_positions = np.array(
            [
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        filtered_positions = baseline_positions.copy()
        filtered_positions[:, 2, 1] = [0.1, 0.2, 0.3]
        filtered_positions[1, 0, 0] += 0.4
        filtered_positions[2, 1, 0] -= 0.2
        baseline = {"trajectory_length": 3, "positions": baseline_positions}
        filtered = {"trajectory_length": 3, "positions": filtered_positions}
        predictions = np.zeros((2, 3, 6))
        predictions[:, :, :3] = np.transpose(
            baseline_positions[:, :2], (1, 0, 2)
        )

        result = _paired_metrics(
            baseline,
            filtered,
            predictions,
            [0.0, 0.02, 0.04],
            num_multi_agents=2,
            proximity_threshold=0.75,
            action_delta_threshold=0.01,
        )

        self.assertAlmostEqual(result["baseline_min_distance"], 0.4)
        self.assertAlmostEqual(result["action_delta_mean"], 0.02)
        self.assertAlmostEqual(result["action_delta_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(result["ego_response_max"], 0.3)
        self.assertAlmostEqual(result["other_response_max"], 0.4)
        self.assertEqual(result["other_response_max_step"], 1)
        self.assertEqual(result["other_response_max_agent"], 0)
        self.assertAlmostEqual(result["other_response_mean"], 0.1)
        self.assertAlmostEqual(result["other_response_final_max"], 0.2)

    def test_paired_metrics_require_one_delta_per_filtered_step(self):
        trajectory = {"trajectory_length": 2, "positions": np.zeros((2, 2, 3))}
        with self.assertRaisesRegex(ValueError, "one action delta"):
            _paired_metrics(
                trajectory,
                trajectory,
                np.zeros((1, 2, 6)),
                [0.0],
                num_multi_agents=1,
                proximity_threshold=1.0,
                action_delta_threshold=0.01,
            )


class SeedSweeperRankingTests(unittest.TestCase):
    def test_qualification_requires_every_condition(self):
        row = {
            "initial_min_distance": 1.0,
            "ego_goal_distance": 2.5,
            "encounter_min_distance": 0.5,
            "action_delta_fraction": 0.2,
            "other_response_max": 0.3,
            "other_response_mean": 0.05,
        }
        args = SimpleNamespace(
            min_initial_distance=0.75,
            min_ego_goal_distance=2.0,
            max_encounter_distance=0.75,
            min_action_fraction=0.1,
            min_other_response_max=0.15,
            min_other_response_mean=0.02,
        )

        self.assertEqual(_qualification_flags(row, args)["interesting"], 1)
        row["other_response_mean"] = 0.01
        self.assertEqual(_qualification_flags(row, args)["interesting"], 0)

    def test_qualified_seed_ranks_above_larger_unqualified_response(self):
        def candidate(seed, interesting, response):
            return {
                "scenario_seed": seed,
                "interesting": interesting,
                "other_response_max": response,
                "other_response_mean": response / 10.0,
                "encounter_min_distance": 0.5,
            }

        ranked = _rank_summaries(
            [candidate(1, 0, 1.0), candidate(2, 1, 0.2), candidate(3, 1, 0.4)]
        )
        self.assertEqual([row["scenario_seed"] for row in ranked], [3, 2, 1])


class SeedSweeperCliTests(unittest.TestCase):
    def test_defaults_sweep_scenarios_with_one_paired_rollout_seed(self):
        args = parse_args([])
        self.assertEqual((args.seed_start, args.seed_stop), (0, 100))
        self.assertEqual(args.rollout_seed, 0)
        self.assertEqual(args.episode_length, 300)
        self.assertEqual(args.num_multi_agents, 5)
        self.assertEqual(args.radii, [0.35, 0.6, 1.0, 1.4])

    def test_rejects_nonpositive_radius(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--radii", "0"])


if __name__ == "__main__":
    unittest.main()
