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
                    "update_procedure",
                }
                self.assertEqual(set(data.files), expected_keys)
                self.assertEqual(data["update_procedure"].item(), "naive")
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
            radius=1.3,
            next_radius=1.2,
            safe=True,
            next_safe=True,
            num_samples=2,
            exceedances=0,
            lower=0.0,
            upper=0.05,
            width=0.05,
            eta=0.5,
            radius_certificate=0.25,
            stopped_forecast=1.0 / 6.0,
            log_hedge_conservative=2.0,
            log_hedge_under=-1.0,
            scores=np.array([0.7, 0.8]),
            indicators=np.array([0, 0], dtype=np.int8),
            lower_history=np.array([0.0, 0.0]),
            upper_history=np.array([0.2, 0.05]),
        )
        _record_anytime_episode(metrics, result, episode_delta=0.05, min_radius=0.2)
        metrics["anytime_config"] = {
            "kappa": 0.6,
            "density_lower": 1.0,
            "density_upper": 1.0,
            "lower_slope": 0.4,
            "upper_slope": 1.6,
            "min_radius": 0.2,
        }
        args = Namespace(
            alpha=0.1,
            delta=0.1,
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
                    data["anytime_upper_probability_history"], [[0.2, 0.05]]
                )
                self.assertAlmostEqual(data["anytime_lower_slope"].item(), 0.4)
                self.assertAlmostEqual(
                    data["anytime_next_radius_per_episode"].item(), 1.2
                )
                self.assertAlmostEqual(data["tube_radius_per_episode"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
