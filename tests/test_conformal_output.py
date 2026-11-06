import tempfile
import unittest
from argparse import Namespace

import numpy as np

from conformal import (
    _empty_metrics,
    _record_anytime_episode,
    _record_episode,
    _save_metrics,
    _validate_predictions,
)
from project_utils.anytime_utils import AnytimeEpisodeResult


class ConformalOutputTests(unittest.TestCase):
    def test_metrics_file_preserves_legacy_schema_and_records_procedure(self):
        metrics = _empty_metrics()
        summary = {
            "tube_coverage": 0.5,
            "crashes": 1.0,
            "bad_crashes": 0.0,
            "safety": 1.0,
            "cumulative_reward": 2.0,
            "cumulative_reward_per_run": np.array([2.0], dtype=np.float32),
            "agent_locs": np.zeros((2, 2, 3)),
        }
        predictions = np.zeros((1, 2, 6))
        _record_episode(metrics, 0.25, 0.5, summary, predictions)
        args = Namespace(
            alpha=0.1,
            delta=0.2,
            kappa=0.5,
            update_procedure="naive",
        )

        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = _save_metrics(
                directory,
                args,
                episode_ids=[0],
                metrics=metrics,
                alpha_bar=0.05,
            )
            with np.load(path) as data:
                expected_keys = {
                    "episodes",
                    "qj_per_episode",
                    "radius_per_episode",
                    "tube_coverage_per_episode",
                    "crashes_per_episode",
                    "bad_crashes_per_episode",
                    "safety_per_episode",
                    "cumulative_reward_per_episode",
                    "cumulative_reward_per_run",
                    "agent_locs_per_episode",
                    "predicted_traj_per_episode",
                    "alpha",
                    "delta",
                    "bar_alpha",
                    "kappa",
                    "update_procedure",
                }
                self.assertEqual(set(data.files), expected_keys)
                self.assertEqual(data["update_procedure"].item(), "naive")
                self.assertEqual(data["kappa"].item(), 0.5)
                self.assertEqual(data["cumulative_reward_per_run"].shape, (1, 1))
                self.assertEqual(
                    data["predicted_traj_per_episode"].shape,
                    (1, 1, 2, 6),
                )

    def test_prediction_validation_rejects_short_trajectories(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            _validate_predictions(
                np.zeros((2, 3, 6)),
                num_multi_agents=2,
                episode_length=4,
            )

    def test_anytime_metrics_include_stopping_diagnostics(self):
        metrics = _empty_metrics()
        summary = {
            "tube_coverage": 1.0,
            "crashes": 0.0,
            "bad_crashes": 0.0,
            "safety": 1.0,
            "cumulative_reward": 2.0,
            "cumulative_reward_per_run": np.array([2.0], dtype=np.float32),
            "agent_locs": np.zeros((2, 2, 3)),
        }
        _record_episode(
            metrics,
            np.nan,
            1.2,
            summary,
            np.zeros((1, 2, 6)),
        )
        result = AnytimeEpisodeResult(
            branch="conservative",
            termination_reason="conservative",
            family="jeffreys",
            radius=1.3,
            lower_radius=0.2,
            next_radius=1.2,
            next_lower_radius=0.3,
            current_width=1.1,
            next_width=0.9,
            num_samples=2,
            exceedances=0,
            quantile_lower=0.7,
            quantile_upper=1.0,
            radius_certificate=1.0,
            scores=np.array([0.7, 0.8]),
            indicators=np.array([0, 0], dtype=np.int8),
            quantile_lower_history=np.array([0.0, 0.7]),
            quantile_upper_history=np.array([np.inf, 1.0]),
            few_critical_history=np.array([-1, 0]),
            many_critical_history=np.array([2, 2]),
            candidate_radius_history=np.array([1.3, 1.2]),
            candidate_lower_radius_history=np.array([0.2, 0.3]),
            candidate_width_history=np.array([1.1, 0.9]),
        )
        _record_anytime_episode(metrics, result, episode_delta=0.05, min_radius=0.2)
        metrics["anytime_config"] = {
            "family": "jeffreys",
            "kappa": 0.6,
            "gamma": 0.75,
            "zeta": 0.875,
            "radius_tolerance": 0.15,
            "k_epsilon": 15,
            "min_radius": 0.2,
            "initial_width": 1.8,
            "initial_safety_mode": "clopper-pearson",
            "delta_init": 0.05,
            "initial_trajectory_count": 2,
            "initial_exceedance_count": 0,
            "initial_probability_lower": 0.0,
            "initial_probability_upper": 0.8,
            "initial_scores": np.array([0.7, 0.8]),
            "initial_indicators": np.array([0, 0], dtype=np.int8),
        }
        metrics["anytime_run_termination_reason"] = "driver-cap"
        args = Namespace(
            alpha=0.1,
            delta=0.1,
            kappa=0.6,
            update_procedure="anytime",
        )

        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = _save_metrics(
                directory,
                args,
                episode_ids=[0],
                metrics=metrics,
                alpha_bar=np.nan,
            )
            with np.load(path) as data:
                self.assertEqual(
                    data["anytime_branch_per_episode"].item(),
                    "conservative",
                )
                np.testing.assert_array_equal(
                    data["anytime_trajectory_count_per_episode"],
                    [2],
                )
                np.testing.assert_allclose(data["anytime_scores"], [[0.7, 0.8]])
                np.testing.assert_array_equal(data["anytime_indicators"], [[0, 0]])
                np.testing.assert_allclose(
                    data["anytime_sample_radii"], [[1.3, 1.3]]
                )
                np.testing.assert_allclose(
                    data["anytime_quantile_upper_history"], [[np.inf, 1.0]]
                )
                np.testing.assert_array_equal(
                    data["anytime_few_critical_history"], [[-1, 0]]
                )
                self.assertAlmostEqual(
                    data["anytime_next_radius_per_episode"].item(), 1.2
                )
                self.assertAlmostEqual(data["tube_radius_per_episode"].item(), 1.0)
                self.assertEqual(data["anytime_schema_version"].item(), 2)
                self.assertEqual(data["anytime_family"].item(), "jeffreys")
                self.assertEqual(
                    data["anytime_run_termination_reason"].item(),
                    "driver-cap",
                )
                self.assertNotIn("anytime_lower_probability_per_episode", data)
                self.assertNotIn("anytime_density_lower", data)


if __name__ == "__main__":
    unittest.main()
