import contextlib
import io
import unittest

from conformal import parse_args


BASE_ARGS = [
    "--multi_experiment",
    "multi",
    "--solo_experiment",
    "solo",
    "--experiment_name",
    "experiment",
]


class ConformalCliTests(unittest.TestCase):
    def test_robust_is_the_backward_compatible_default(self):
        args = parse_args(BASE_ARGS)
        self.assertEqual(args.update_procedure, "robust")
        self.assertEqual(args.device, "cpu")

    def test_gpu_device_is_accepted(self):
        args = parse_args(BASE_ARGS + ["--device", "gpu"])
        self.assertEqual(args.device, "gpu")

    def test_all_update_procedures_are_accepted(self):
        for procedure in (
            "robust",
            "naive",
            "non-interactive",
            "calibrate-once",
            "anytime",
        ):
            with self.subTest(procedure=procedure):
                args = parse_args(
                    BASE_ARGS + ["--update-procedure", procedure]
                )
                self.assertEqual(args.update_procedure, procedure)

    def test_non_iterative_modes_reject_prediction_updates(self):
        for procedure in ("non-interactive", "calibrate-once"):
            with self.subTest(procedure=procedure):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(
                            BASE_ARGS
                            + [
                                "--update-procedure",
                                procedure,
                                "--update_predictions",
                            ]
                        )

    def test_iterative_modes_require_at_least_one_episode(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(BASE_ARGS + ["--num_episodes", "0"])

    def test_non_iterative_modes_allow_zero_episode_updates(self):
        for procedure in ("non-interactive", "calibrate-once"):
            with self.subTest(procedure=procedure):
                args = parse_args(
                    BASE_ARGS
                    + [
                        "--update-procedure",
                        procedure,
                        "--num_episodes",
                        "0",
                    ]
                )
                self.assertEqual(args.num_episodes, 0)

    def test_anytime_rejects_inverted_density_bounds(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    BASE_ARGS
                    + [
                        "--update-procedure",
                        "anytime",
                        "--anytime-score-density-lower",
                        "2.0",
                        "--anytime-score-density-upper",
                        "1.0",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
