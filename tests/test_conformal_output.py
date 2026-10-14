import tempfile
import unittest
from argparse import Namespace

import numpy as np

from conformal import (
    _empty_metrics,
    _record_episode,
    _save_metrics,
    _validate_predictions,
)


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


if __name__ == "__main__":
    unittest.main()
