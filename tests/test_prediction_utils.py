import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from collect_homogeneous_predictions import parse_args as parse_collection_args
from conformal import parse_args as parse_conformal_args
from project_utils.prediction_utils import (
    load_prediction_artifact,
    save_prediction_artifact,
)
from project_utils.restart_utils import QuadState


class PredictionArtifactTests(unittest.TestCase):
    @staticmethod
    def _states(num_agents=3):
        return [
            QuadState(
                position=np.array([agent_id, 0.0, 1.0], dtype=np.float64),
                velocity=np.zeros(3, dtype=np.float64),
                rotation=np.eye(3, dtype=np.float64),
                omega=np.zeros(3, dtype=np.float64),
                goal=np.array([0.0, agent_id, 2.0], dtype=np.float64),
            )
            for agent_id in range(num_agents)
        ]

    def _write_artifact(self, directory):
        path = os.path.join(directory, "pred_trajectories.npz")
        all_trajectories = np.arange(3 * 4 * 6, dtype=np.float32).reshape(
            3, 4, 6
        )
        save_prediction_artifact(
            path,
            all_trajectories[:-1],
            all_trajectories,
            self._states(),
            policy_experiment="multi",
            policy_checkpoint="checkpoint_123.pth",
            seed=7,
        )
        return path, all_trajectories

    def test_round_trip_validates_and_preserves_homogeneous_subset(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path, all_trajectories = self._write_artifact(directory)
            predictions = load_prediction_artifact(
                path,
                expected_num_multi_agents=2,
                expected_episode_length=3,
                expected_seed=7,
                expected_policy_experiment="multi",
                expected_policy_checkpoint="/tmp/checkpoint_123.pth",
                expected_initial_states=self._states(),
            )

        np.testing.assert_array_equal(predictions, all_trajectories[:-1])

    def test_rejects_legacy_artifact_without_provenance(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = os.path.join(directory, "pred_trajectories.npz")
            np.savez(path, pred_trajectories=np.zeros((2, 4, 6)))
            with self.assertRaisesRegex(ValueError, "provenance metadata"):
                load_prediction_artifact(
                    path,
                    expected_num_multi_agents=2,
                    expected_episode_length=4,
                    expected_seed=7,
                    expected_policy_experiment="multi",
                    expected_policy_checkpoint="checkpoint_123.pth",
                    expected_initial_states=self._states(),
                )

    def test_rejects_seed_and_initial_state_mismatches(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path, _ = self._write_artifact(directory)
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                load_prediction_artifact(
                    path,
                    expected_num_multi_agents=2,
                    expected_episode_length=4,
                    expected_seed=8,
                    expected_policy_experiment="multi",
                    expected_policy_checkpoint="checkpoint_123.pth",
                    expected_initial_states=self._states(),
                )

            changed_states = self._states()
            changed_states[0].goal[0] = 1.0
            with self.assertRaisesRegex(ValueError, "initial goals"):
                load_prediction_artifact(
                    path,
                    expected_num_multi_agents=2,
                    expected_episode_length=4,
                    expected_seed=7,
                    expected_policy_experiment="multi",
                    expected_policy_checkpoint="checkpoint_123.pth",
                    expected_initial_states=changed_states,
                )


class FixedScenarioCliTests(unittest.TestCase):
    def test_collection_defaults_match_requested_artifact(self):
        args = parse_collection_args([])
        self.assertEqual(args.multi_experiment, "multi_ddg_6")
        self.assertEqual(args.experiment_name, "homog_gtg_pred_traj_data")
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.episode_length, 300)
        self.assertEqual(args.num_multi_agents, 5)
        self.assertFalse(hasattr(args, "quads_mode"))

    def test_scenario_is_not_exposed_by_collection_or_conformal_cli(self):
        conformal_base = [
            "--multi_experiment",
            "multi",
            "--solo_experiment",
            "solo",
            "--experiment_name",
            "experiment",
        ]
        for parser, argv in (
            (parse_collection_args, ["--quads_mode", "patrol_dual_goal"]),
            (
                parse_conformal_args,
                conformal_base + ["--quads_mode", "patrol_dual_goal"],
            ),
        ):
            with self.subTest(parser=parser.__module__):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser(argv)


if __name__ == "__main__":
    unittest.main()
