import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_all.sh"


class RunAllTests(unittest.TestCase):
    def run_script(self, extra_args=(), *, predictions_exist=True, device="cpu"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "l4dc" / "quad-swarm-rl" / "run_all.sh"
            script.parent.mkdir(parents=True)
            shutil.copyfile(SCRIPT, script)
            train_dir = root / "train_dir"
            train_dir.mkdir()
            prediction = train_dir / "homog_gtg_pred_traj_data_seed439" / "pred_trajectories.npz"
            if predictions_exist:
                prediction.parent.mkdir()
                prediction.write_bytes(b"unchanged predictions")
            old_log = train_dir / "gtg_s439_robust_kappa0p2.log"
            old_log.write_text("old results")
            env = dict(os.environ)
            # Replace python with an exported bash function; no experiments are launched.
            env["BASH_FUNC_python%%"] = '() { printf "%s\\n" "$@"; }'
            result = subprocess.run(
                ["bash", str(script), "--seed", "439", "--device", device, "--kappa", "0.2",
                 *extra_args], env=env, capture_output=True, text=True,
            )
            logs = {path.name: path.read_text() for path in train_dir.glob("*.log")}
            if predictions_exist:
                self.assertEqual(prediction.read_bytes(), b"unchanged predictions")
            return result, logs

    def test_omitted_run_seed_keeps_old_names_and_arguments(self):
        result, logs = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(logs), 6)
        for name, content in logs.items():
            argv = content.splitlines()
            self.assertNotIn("--run-seed", argv)
            self.assertNotIn("_run", name)
            self.assertEqual(argv[argv.index("--seed") + 1], "439")
            self.assertNotIn("_run", argv[argv.index("--experiment_name") + 1])

    def test_explicit_run_seed_is_forwarded_and_protects_old_outputs(self):
        for seed in ("0", "100", "439", "4294967295"):
            with self.subTest(seed=seed):
                result, logs = self.run_script(["--run-seed", seed])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(logs.pop("gtg_s439_robust_kappa0p2.log"), "old results")
                self.assertEqual(len(logs), 6)
                for name, content in logs.items():
                    argv = content.splitlines()
                    self.assertEqual(argv[argv.index("--run-seed") + 1], seed)
                    self.assertEqual(argv[argv.index("--seed") + 1], "439")
                    self.assertEqual(argv[argv.index("--init_predictions") + 1],
                                     "homog_gtg_pred_traj_data_seed439")
                    self.assertTrue(name.endswith(f"_run{seed}.log"))
                    self.assertTrue(argv[argv.index("--experiment_name") + 1].endswith(f"_run{seed}"))

    def test_prediction_collection_uses_only_scenario_seed(self):
        result, _ = self.run_script(["--run-seed", "100"], predictions_exist=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("collect_homogeneous_predictions.py", result.stdout)
        self.assertIn("--seed\n439\n", result.stdout)
        self.assertNotIn("--run-seed", result.stdout)
        self.assertNotIn("_run100", result.stdout)

    def test_gpu_dispatch_also_forwards_run_seed_to_every_algorithm(self):
        result, logs = self.run_script(["--run-seed", "100"], device="gpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        logs.pop("gtg_s439_robust_kappa0p2.log")
        self.assertEqual(len(logs), 6)
        for content in logs.values():
            argv = content.splitlines()
            self.assertEqual(argv[argv.index("--device") + 1], "gpu")
            self.assertEqual(argv[argv.index("--run-seed") + 1], "100")

    def test_invalid_run_seeds_fail_before_launching(self):
        for extra in (["--run-seed"], ["--run-seed", ""], ["--run-seed", "-1"],
                      ["--run-seed", "4294967296"], ["--run-seed", "99999999999999999999"],
                      ["--run-seed", "1.5"], ["--run-seed", "nan"]):
            with self.subTest(extra=extra):
                result, logs = self.run_script(extra)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--run-seed", result.stderr)
                self.assertEqual(logs, {"gtg_s439_robust_kappa0p2.log": "old results"})


if __name__ == "__main__":
    unittest.main()
