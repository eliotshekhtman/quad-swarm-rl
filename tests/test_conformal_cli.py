import contextlib
import io
import unittest
from unittest import mock

from conformal import parse_args
from project_utils import full_cbf_utils


BASE_ARGS = [
    "--multi_experiment",
    "multi",
    "--solo_experiment",
    "solo",
    "--experiment_name",
    "experiment",
]


class ConformalCliTests(unittest.TestCase):
    def test_anytime_families_and_alias(self):
        base = BASE_ARGS + ["--update-procedure", "anytime", "--anytime-init-trajectories", "1000"]
        self.assertEqual(parse_args(base).anytime_family, "jeffreys")
        for family in ("jeffreys", "informed", "jeffreys-informed", "directional", "normal"):
            args = parse_args(base + ["--anytime-family", family])
            self.assertEqual(args.anytime_family, family)
            self.assertEqual(args.anytime_jeffreys_weight, .5)
            self.assertEqual(args.anytime_directional_weight, .5)
        self.assertEqual(parse_args(BASE_ARGS + ["--update-procedure", "anytime-nonparametric",
            "--anytime-assume-initial-safe"]).anytime_family, "normal")
        args = parse_args(base + ["--anytime-family", "directional", "--anytime-jeffreys-weight", "1",
                                 "--anytime-directional-weight", ".7"])
        self.assertEqual(args.anytime_jeffreys_weight, 1)
        self.assertEqual(args.anytime_directional_weight, .7)

    def test_invalid_mixture_combinations_rejected(self):
        base = ["--update-procedure", "anytime", "--anytime-init-trajectories", "1000"]
        invalid = [
            ["--anytime-family", "informed"],
            ["--update-procedure", "anytime-nonparametric", "--anytime-family", "directional",
             "--anytime-init-trajectories", "1000"],
            base + ["--anytime-jeffreys-weight", ".5"],
            base + ["--anytime-family", "informed", "--anytime-directional-weight", ".5"],
        ]
        for family in ("informed", "jeffreys-informed", "directional"):
            invalid.append(["--update-procedure", "anytime", "--anytime-family", family,
                            "--anytime-assume-initial-safe"])
        for flag in ("--anytime-jeffreys-weight", "--anytime-directional-weight"):
            for value in ("0", "-1", "1.1", "nan", "inf"):
                invalid.append(base + ["--anytime-family", "directional", flag, value])
        invalid.append(base + ["--anytime-family", "directional", "--anytime-directional-weight", "1"])
        for extra in invalid:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(BASE_ARGS + extra)

    def test_robust_is_the_backward_compatible_default(self):
        args = parse_args(BASE_ARGS)
        self.assertIsNone(args.run_seed)
        self.assertEqual(args.update_procedure, "robust")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.r_init, 2.0)
        self.assertEqual(args.kappa, 0.6)
        self.assertEqual(args.cbf_k1, 2.5)
        self.assertEqual(args.cbf_solver, "cascade")

    def test_run_seed_is_optional_and_independent_of_scenario_seed(self):
        for seed in (0, 100, 439, 2**32 - 1):
            with self.subTest(seed=seed):
                args = parse_args(BASE_ARGS + ["--seed", "439", "--run-seed", str(seed)])
                self.assertEqual(args.seed, 439)
                self.assertEqual(args.run_seed, seed)

    def test_invalid_run_seeds_are_rejected(self):
        for seed in ("-1", str(2**32), "1.5", "nan", ""):
            with self.subTest(seed=seed), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(BASE_ARGS + ["--run-seed", seed])

    def test_legacy_controller_is_explicit_and_independent_of_kappa(self):
        args = parse_args(BASE_ARGS + ["--cbf-k1", "4", "--cbf-solver", "legacy"])
        self.assertEqual(args.cbf_k1, 4.0)
        self.assertEqual(args.cbf_solver, "legacy")
        self.assertEqual(args.kappa, 0.6)

    def test_default_gain_respects_grid_worker_override(self):
        with mock.patch.object(full_cbf_utils, "CBF_K1", 3.5):
            self.assertEqual(parse_args(BASE_ARGS).cbf_k1, 3.5)

    def test_invalid_controller_settings_are_rejected(self):
        for extra in (["--cbf-k1", "0"], ["--cbf-k1", "-1"],
                      ["--cbf-k1", "nan"], ["--cbf-k1", "inf"],
                      ["--cbf-solver", "unknown"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(BASE_ARGS + extra)

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
            "calibrate-once-paper",
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
        for procedure in ("non-interactive", "calibrate-once", "calibrate-once-paper"):
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
        for procedure in ("non-interactive", "calibrate-once", "calibrate-once-paper"):
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
