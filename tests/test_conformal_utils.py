import unittest

import numpy as np

from project_utils.conformal_utils import (
    calibrate_once_final_radius,
    conformal_radii,
    explicit_radius_update,
    fall_down,
    get_alpha_bar,
    next_radius,
    summarize_rollouts,
)


class ConformalQuantileTests(unittest.TestCase):
    def test_get_alpha_bar(self):
        actual = get_alpha_bar(alpha=0.8, delta=0.9, num_trajectories=4)
        expected = 0.8 - np.sqrt(np.log(1.0 / 0.9) / 8.0)
        self.assertAlmostEqual(actual, expected)

    def test_conformal_radii_uses_trajectory_max_and_requested_quantile(self):
        scores_by_agent = ([1.0, 4.0, 2.0, 3.0], [0.5, 2.0, 1.0, 1.5])
        logs = {}
        for agent_id, scores in enumerate(scores_by_agent):
            logs[agent_id] = []
            for score in scores:
                logs[agent_id].append(
                    {
                        "position": np.array([[score, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                        "velocity": np.zeros((2, 3)),
                    }
                )

        predictions = [np.zeros((2, 6)), np.zeros((2, 6))]
        radii = conformal_radii(
            logs,
            num_multi_agents=2,
            pred_trajectories=predictions,
            alpha=0.25,
            episode_length=2,
        )

        np.testing.assert_allclose(radii, [3.0, 1.5])


class RadiusUpdateTests(unittest.TestCase):
    def test_robust_update_when_quantile_decreases(self):
        radius = explicit_radius_update(4.0, 2.0, 0.6, 0.1, 8.0)
        self.assertAlmostEqual(radius, 2.75)

    def test_robust_update_is_clipped_to_maximum(self):
        radius = explicit_radius_update(4.0, 6.0, 0.6, 0.1, 8.0)
        self.assertAlmostEqual(radius, 8.0)

    def test_named_robust_update_matches_explicit_update(self):
        actual = next_radius("robust", 4.0, 2.0, 0.6, 0.1, 8.0)
        self.assertAlmostEqual(actual, 2.75)

    def test_naive_update_clips_quantile_to_physical_minimum(self):
        actual = next_radius("naive", 8.0, 0.01, 0.6, 0.1, 8.0)
        self.assertAlmostEqual(actual, 0.1)

    def test_calibrate_once_recurrence_matches_legacy_schedule(self):
        actual = calibrate_once_final_radius(
            initial_radius=8.0,
            calibrated_radius=0.0,
            kappa=0.6,
            num_updates=2,
        )
        self.assertAlmostEqual(actual, 2.88)

    def test_unknown_update_procedure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            next_radius("unknown", 4.0, 2.0, 0.6, 0.1, 8.0)


class RolloutSummaryTests(unittest.TestCase):
    @staticmethod
    def _run_log(positions, goal_dist):
        return {
            "position": np.asarray(positions, dtype=np.float64),
            "velocity": np.zeros((3, 3), dtype=np.float64),
            "goal_dist": np.asarray(goal_dist, dtype=np.float64),
            "goal_swap": np.array([False, False, False]),
        }

    def test_summary_tracks_progress_tubes_and_avoidable_crashes(self):
        logs = {
            0: [
                self._run_log(
                    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    [0.0, 0.0, 0.0],
                ),
                self._run_log(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [0.0, 0.0, 0.0],
                ),
            ],
            1: [
                self._run_log(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [0.0, 0.0, 0.0],
                ),
                self._run_log(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [0.0, 0.0, 0.0],
                ),
            ],
            2: [
                self._run_log(
                    [[5.0, 0.0, 0.0], [2.4, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    [3.0, 2.0, 1.0],
                ),
                self._run_log(
                    [[5.0, 0.0, 0.0], [0.4, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [3.0, 4.0, 3.0],
                ),
            ],
        }
        predictions = [np.zeros((3, 6)), np.zeros((3, 6))]

        summary = summarize_rollouts(
            logs,
            predictions,
            radius=1.0,
            min_radius=0.5,
            num_multi_agents=2,
            episode_length=3,
            num_runs=2,
        )

        self.assertAlmostEqual(summary["cumulative_reward"], 1.5)
        self.assertAlmostEqual(summary["tube_coverage"], 0.75)
        self.assertAlmostEqual(summary["crashes"], 1.5)
        self.assertAlmostEqual(summary["bad_crashes"], 0.5)
        self.assertAlmostEqual(summary["safety"], 0.5)
        np.testing.assert_allclose(
            summary["cumulative_reward_per_run"],
            [2.0, 1.0],
        )
        self.assertEqual(len(summary["agent_locs"]), 3)

    def test_fall_down_accepts_rollout_keyword_interface(self):
        action = fall_down(
            base_action=np.ones(4),
            env_state=object(),
            swarm_state=object(),
        )
        np.testing.assert_array_equal(action, -np.ones(4))


if __name__ == "__main__":
    unittest.main()
