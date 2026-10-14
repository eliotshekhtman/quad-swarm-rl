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

    def test_all_update_procedures_are_accepted(self):
        for procedure in ("robust", "naive", "calibrate-once"):
            with self.subTest(procedure=procedure):
                args = parse_args(
                    BASE_ARGS + ["--update-procedure", procedure]
                )
                self.assertEqual(args.update_procedure, procedure)

    def test_calibrate_once_rejects_prediction_updates(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    BASE_ARGS
                    + [
                        "--update-procedure",
                        "calibrate-once",
                        "--update_predictions",
                    ]
                )

    def test_iterative_modes_require_at_least_one_episode(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(BASE_ARGS + ["--num_episodes", "0"])


if __name__ == "__main__":
    unittest.main()
