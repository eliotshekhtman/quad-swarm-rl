import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import conformal
from project_utils.restart_utils import set_global_seed


def draw():
    return [random.random(), np.random.random(), torch.rand(()).item()]


def run_driver(extra_args=(), *, driver=conformal, generated_predictions=False):
    """Exercise the real driver with inexpensive, stochastic simulated rollouts."""
    events, samples, initial_observations = [], [], []
    env = mock.Mock(quad_arm=0.08)
    env.observation_space.shape = (3,)
    cfg = SimpleNamespace(quads_neighbor_visible_num=1, quads_neighbor_obs_type="test")

    def reset(_env, seed, stored_states):
        assert stored_states is None
        set_global_seed(seed)
        events.append("scenario")
        obs = np.tile(draw(), (2, 1))
        initial_observations.append(obs.copy())
        return obs, ["fixed scenario"]

    def predictions(*args, **kwargs):
        if not generated_predictions:
            assert kwargs["expected_seed"] == 439
            assert kwargs["expected_initial_states"] == ["fixed scenario"]
        events.append("predictions")
        draw()
        return np.zeros((2 if generated_predictions else 1, 2, 6))

    def seed_run(seed):
        events.append(("run-seed", seed))
        set_global_seed(seed)

    def rollouts(_env, obs, *args, **kwargs):
        events.append("rollouts")
        np.testing.assert_array_equal(obs, initial_observations[0])
        logs = {0: [], 1: []}
        for _ in range(kwargs["num_runs"]):
            sample = draw()
            samples.append(sample)
            for agent in logs:
                position = [0.2 + sample[1] * 0.1, 0, 0] if agent == 0 else [4, 0, 0]
                logs[agent].append(dict(
                    position=np.tile(position, (2, 1)), velocity=np.zeros((2, 3)),
                    goal_dist=[2.0, 1.0], goal_swap=[False, False],
                ))
        return logs

    replacements = dict(
        register_swarm_components=lambda: None,
        parse_swarm_cfg=lambda *args, **kwargs: cfg,
        load_cfg=lambda *args: cfg,
        make_quadrotor_env=lambda *args, **kwargs: env,
        latest_checkpoint=lambda *args, **kwargs: "checkpoint",
        load_actor=lambda *args: None,
        get_rnn_size=lambda *args: 1,
        deterministic_reset=reset,
        safe_capture_env_snapshot=lambda _env: "fixed snapshot",
        clone_env_from_snapshot=lambda snapshot: env,
        load_prediction_artifact=predictions,
        collect_homogeneous_trajectories=predictions,
        make_cbf_filter=lambda *args, **kwargs: None,
        run_multi_agents=rollouts,
    )
    # The pre-change driver has no imported runtime seed helper.
    if hasattr(driver, "set_global_seed"):
        replacements["set_global_seed"] = seed_run
    with tempfile.TemporaryDirectory() as directory:
        argv = [
            "conformal.py", "--multi_experiment", "multi", "--solo_experiment", "solo",
            "--experiment_name", "test", "--train_dir", directory, "--seed", "439",
            "--episode_length", "2", "--num_multi_agents", "1", "--num_trajectories", "40",
            "--num_eval_trajs", "3", "--num_episodes", "2", "--alpha", ".5", "--kappa", ".2",
        ]
        if not generated_predictions:
            argv += ["--init_predictions", "fixed_predictions"]
        argv += list(extra_args)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("sys.argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for name, replacement in replacements.items():
                stack.enter_context(mock.patch.object(driver, name, replacement))
            driver.main()
        result_dir = Path(directory) / "test"
        metadata = json.loads((result_dir / "conformal_args.json").read_text())
        with np.load(result_dir / "conformal_metrics.npz") as data:
            metrics = {name: data[name].copy() for name in data.files}
    return dict(
        events=events, samples=np.asarray(samples), initial_obs=initial_observations[0],
        metadata=metadata, metrics=metrics,
    )


class ConformalSeedingTests(unittest.TestCase):
    def test_omitted_seed_preserves_continuing_startup_stream(self):
        for generated in (False, True):
            with self.subTest(generated=generated):
                result = run_driver(generated_predictions=generated)
                self.assertEqual(result["events"][:3], ["scenario", "predictions", "rollouts"])
                self.assertFalse(any(isinstance(event, tuple) for event in result["events"]))
                self.assertIsNone(result["metadata"]["run_seed"])
                set_global_seed(439)
                draw()  # Scenario setup.
                draw()  # Prediction setup.
                expected = [draw() for _ in result["samples"]]
                np.testing.assert_array_equal(result["samples"], expected)

    def test_run_seed_is_applied_once_after_predictions_in_every_procedure(self):
        procedures = [(procedure, []) for procedure in conformal.UPDATE_PROCEDURES]
        procedures += [("anytime", ["--anytime-family", family]) for family in
                       ("informed", "jeffreys-informed", "directional")]
        for procedure, extra in procedures:
            for generated in (False, True):
                with self.subTest(procedure=procedure, extra=extra, generated=generated):
                    argv = ["--update-procedure", procedure, "--run-seed", "0", *extra]
                    if procedure in conformal.ANYTIME_PROCEDURES:
                        argv += ["--anytime-init-trajectories", "10"]
                    result = run_driver(argv, generated_predictions=generated)
                    self.assertEqual(result["events"][:4],
                                     ["scenario", "predictions", ("run-seed", 0), "rollouts"])
                    self.assertEqual(result["events"].count(("run-seed", 0)), 1)
                    self.assertEqual(result["metadata"]["seed"], 439)
                    self.assertEqual(result["metadata"]["run_seed"], 0)
                    set_global_seed(0)
                    expected = [draw() for _ in result["samples"]]
                    np.testing.assert_array_equal(result["samples"], expected)

    def test_repetitions_change_samples_not_initial_observations_or_predictions(self):
        legacy = run_driver()
        first = run_driver(["--run-seed", "100"])
        repeat = run_driver(["--run-seed", "100"])
        different = run_driver(["--run-seed", "101"])
        equal_to_scenario = run_driver(["--run-seed", "439"])
        np.testing.assert_array_equal(first["samples"], repeat["samples"])
        for name in first["metrics"]:
            np.testing.assert_array_equal(first["metrics"][name], repeat["metrics"][name])
        for result in (first, repeat, different, equal_to_scenario):
            np.testing.assert_array_equal(result["initial_obs"], legacy["initial_obs"])
            np.testing.assert_array_equal(result["metrics"]["predicted_traj_per_episode"],
                                          legacy["metrics"]["predicted_traj_per_episode"])
        self.assertFalse(np.array_equal(first["samples"], different["samples"]))
        self.assertFalse(np.array_equal(legacy["samples"], equal_to_scenario["samples"]))


if __name__ == "__main__":
    unittest.main()
