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
        self.assertEqual(args.r_init, 2.0)
        self.assertEqual(args.kappa, 0.6)

    def test_kappa_is_configurable(self):
        args = parse_args(BASE_ARGS + ["--kappa", "0.5"])
        self.assertEqual(args.kappa, 0.5)

    def test_kappa_rejects_values_outside_small_gain_domain(self):
        for value in ("-0.1", "1", "nan", "inf"):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(BASE_ARGS + ["--kappa", value])

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
            "anytime-nonparametric",
        ):
            with self.subTest(procedure=procedure):
                procedure_args = ["--update-procedure", procedure]
                if procedure.startswith("anytime"):
                    procedure_args.append("--anytime-assume-initial-safe")
                args = parse_args(
                    BASE_ARGS + procedure_args
                )
                self.assertEqual(args.update_procedure, procedure)
                if procedure.startswith("anytime"):
                    self.assertEqual(args.anytime_delta_init, 0.0)

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

    def test_anytime_requires_explicit_initial_safety_mode(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    BASE_ARGS
                    + [
                        "--update-procedure",
                        "anytime",
                    ]
                )

    def test_anytime_accepts_initial_calibration(self):
        args = parse_args(
            BASE_ARGS
            + [
                "--update-procedure",
                "anytime",
                "--anytime-init-trajectories",
                "100",
            ]
        )
        self.assertEqual(args.anytime_init_trajectories, 100)
        self.assertEqual(args.anytime_delta_init, 0.05)
        self.assertFalse(args.anytime_assume_initial_safe)

    def test_anytime_rejects_two_initial_safety_modes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    BASE_ARGS
                    + [
                        "--update-procedure",
                        "anytime",
                        "--anytime-init-trajectories",
                        "100",
                        "--anytime-assume-initial-safe",
                    ]
                )

    def test_anytime_rejects_prediction_updates(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    BASE_ARGS
                    + [
                        "--update-procedure",
                        "anytime",
                        "--anytime-assume-initial-safe",
                        "--update_predictions",
                    ]
                )

    def test_anytime_rejects_combined_error_budget_at_least_one(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    BASE_ARGS
                    + [
                        "--update-procedure",
                        "anytime",
                        "--anytime-init-trajectories",
                        "100",
                        "--delta",
                        "0.6",
                        "--anytime-delta-init",
                        "0.4",
                    ]
                )

    def test_removed_probability_slope_argument_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    BASE_ARGS
                    + ["--anytime-score-density-lower", "1.0"]
                )


if __name__ == "__main__":
    unittest.main()
