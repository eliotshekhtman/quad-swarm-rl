import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_calibration import (
    calibration_budget,
    load_initial_radius,
    load_noninteractive_radius,
    main,
    plot_cumulative_trajectories,
    plot_quantile_bounds,
    quantile_bound_panels,
)


class PlotCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "conformal_metrics.npz"
        (self.root / "conformal_args.json").write_text(json.dumps({
            "r_init": 2.0, "num_trajectories": 1000,
        }))

    def write_anytime(self, procedure="anytime", references=None):
        np.savez(
            self.path, update_procedure=procedure, episodes=[0, 1, 2],
            anytime_trajectory_count_per_episode=[2, 3, 4],
            anytime_initial_trajectory_count=1000,
            anytime_current_radius_per_episode=[2.0, 1.8, 1.6],
            anytime_quantile_lower_history=[
                [0, 0.8, np.nan, np.nan], [0, 0.7, 0.9, np.nan], [0, 0.8, 1.0, 1.1],
            ],
            anytime_quantile_upper_history=[
                [np.inf, 1.8, np.nan, np.nan], [np.inf, 1.8, 1.6, np.nan], [np.inf, 1.8, 1.6, 1.5],
            ],
            anytime_termination_reason_per_episode=["conservative", "conservative", "certificate"],
            **({} if references is None else {
                "anytime_reference_quantile_per_episode": references,
            }),
        )

    def test_anytime_budgets_include_initialization_separately(self):
        for procedure in ("anytime", "anytime-nonparametric"):
            self.write_anytime(procedure)
            episodes, counts, initial = calibration_budget(self.path)
            np.testing.assert_array_equal(episodes, [0, 1, 2])
            np.testing.assert_array_equal(counts, [2, 5, 9])
            self.assertEqual(initial, 1000)

    def test_baseline_counts_exclude_diagnostic_batches(self):
        for procedure, expected in (
            ("robust", [1000, 2000, 3000]),
            ("naive", [1000, 2000, 3000]),
            ("non-interactive", [1000, 1000, 1000]),
            ("calibrate-once", [1000, 1000, 1000]),
            ("calibrate-once-paper", [1000, 1000, 1000]),
        ):
            np.savez(self.path, update_procedure=procedure, episodes=[0, 1, 21])
            episodes, counts, initial = calibration_budget(self.path)
            np.testing.assert_array_equal(episodes, [0, 1, 21])
            np.testing.assert_array_equal(counts, expected)
            self.assertEqual(initial, 0)

    def test_missing_baseline_budget_is_not_guessed(self):
        (self.root / "conformal_args.json").unlink()
        np.savez(self.path, update_procedure="robust", episodes=[0])
        with self.assertRaisesRegex(ValueError, "num_trajectories"):
            calibration_budget(self.path)

    def test_radius_sources_and_historical_fallback(self):
        self.assertEqual(load_initial_radius(self.path, {}), 2.0)
        self.assertEqual(load_initial_radius(self.path, {
            "anytime_current_radius_per_episode": [1.65],
        }), 1.65)
        (self.root / "conformal_args.json").unlink()
        self.assertEqual(load_initial_radius(self.path, {}), 8.0)
        np.savez(self.path, update_procedure="non-interactive", radius_per_episode=[2, 1.48])
        self.assertEqual(load_noninteractive_radius(self.path), 1.48)

    def test_bounds_slice_padding_and_keep_infinity(self):
        self.write_anytime()
        panels = quantile_bound_panels(self.path)
        self.assertEqual([p["episode"] for p in panels], [0, 1, 2])
        self.assertEqual([len(p["lower"]) for p in panels], [2, 3, 4])
        self.assertTrue(np.isinf(panels[0]["upper"][0]))
        self.assertIsNone(panels[0]["reference_quantile"])
        self.assertEqual(quantile_bound_panels(self.path, [2])[0]["count"], 4)
        with self.assertRaisesRegex(ValueError, "Episode 5"):
            quantile_bound_panels(self.path, [5])

    def test_plotted_counts_and_bound_coordinates(self):
        self.write_anytime()
        with mock.patch("plot_calibration._save_figure", side_effect=lambda fig, *args: fig):
            fig = plot_cumulative_trajectories({"Anytime": self.path}, self.root)
            np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [2, 5, 9])
            np.testing.assert_array_equal(fig.axes[0].lines[1].get_ydata(), [1002, 1005, 1009])
            plt.close(fig)
            fig = plot_quantile_bounds(self.path, self.root, [0])
            ax = fig.axes[0]
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2])
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 0.8])
            self.assertTrue(np.isnan(ax.lines[1].get_ydata()[0]))
            np.testing.assert_array_equal(ax.lines[3].get_xdata(), [2, 2])
            self.assertNotIn(r"Independent estimate of $q_j$", ax.get_legend_handles_labels()[1])
            plt.close(fig)

    def test_independent_references_are_not_shifted_or_replaced_by_radius(self):
        for procedure in ("anytime", "anytime-nonparametric"):
            self.write_anytime(procedure, references=[1.0, 1.2, 1.4])
            panels = quantile_bound_panels(self.path, [2, 0])
            self.assertEqual([p["reference_quantile"] for p in panels], [1.4, 1.0])
            with mock.patch("plot_calibration._save_figure", side_effect=lambda fig, *args: fig):
                fig = plot_quantile_bounds(self.path, self.root, [2, 0])
            for ax, reference, radius in zip(fig.axes, [1.4, 1.0], [1.6, 2.0]):
                lines = {line.get_label(): line for line in ax.lines}
                q_line = lines[r"Independent estimate of $q_j$"]
                r_line = lines[r"Deployed radius $r_j$"]
                np.testing.assert_array_equal(q_line.get_ydata(), [reference, reference])
                np.testing.assert_array_equal(r_line.get_ydata(), [radius, radius])
                self.assertEqual(q_line.get_linestyle(), ":")
                self.assertEqual(r_line.get_linestyle(), "-.")
            plt.close(fig)

    def test_misaligned_reference_array_is_rejected(self):
        self.write_anytime(references=[1.0, 1.2])
        with self.assertRaisesRegex(ValueError, "must match episodes"):
            quantile_bound_panels(self.path)

    def test_missing_reference_in_one_panel_is_not_invented(self):
        self.write_anytime(references=[np.nan, 1.2, 1.4])
        with mock.patch("plot_calibration._save_figure", side_effect=lambda fig, *args: fig):
            fig = plot_quantile_bounds(self.path, self.root)
        label = r"Independent estimate of $q_j$"
        self.assertNotIn(label, fig.axes[0].get_legend_handles_labels()[1])
        self.assertIn(label, [text.get_text() for text in fig.legends[0].get_texts()])
        plt.close(fig)

    def test_cli_renders_both_formats(self):
        self.write_anytime()
        output = self.root / "plots"
        with mock.patch.object(sys, "argv", [
            "plot_calibration.py", "--plot_data", str(self.path), "--output_dir", str(output),
        ]), plt.rc_context({"text.usetex": False}):
            main()
        self.assertEqual(len(list(output.iterdir())), 4)
        for path in output.iterdir():
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
