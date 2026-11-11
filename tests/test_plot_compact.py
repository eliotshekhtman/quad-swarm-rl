import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_compact as compact


class PlotCompactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=".")
        self.root = Path(self.temporary.name)
        self.paths = {}
        for key, (_, _, _, procedure) in compact.METHODS.items():
            if key in compact.OPTIONAL_METHODS:
                continue
            folder = self.root / key
            folder.mkdir()
            (folder / "conformal_args.json").write_text(json.dumps({
                "r_init": 2.0, "num_trajectories": 1000,
            }))
            self.paths[key] = folder / "conformal_metrics.npz"
            self.write_run(key, procedure)

    def tearDown(self):
        plt.close("all")
        self.temporary.cleanup()

    def write_run(self, key, procedure):
        n = 3 if key in ("robust", "naive") else 2
        radii = np.linspace(1.6, 1.3, n)
        positions = np.random.default_rng(1).normal(size=(n, 3, 7, 3))
        fields = {
            "episodes": np.arange(n),
            "radius_per_episode": radii,
            "qj_per_episode": radii + 0.5,
            "tube_coverage_per_episode": np.linspace(0.75, 0.95, n),
            "safety_per_episode": np.linspace(0.97, 1.0, n),
            "cumulative_reward_per_episode": np.linspace(0.4, 0.6, n),
            "cumulative_reward_per_run": np.tile([[0.2, 0.4, 0.5, 0.8]], (n, 1)),
            "alpha": 0.1,
            "update_procedure": procedure,
            "agent_locs_per_episode": positions,
            "predicted_traj_per_episode": np.concatenate([positions[:, :-1], positions[:, :-1]], axis=-1),
        }
        assessed_radii = np.r_[2.0, radii] if key in ("robust", "naive", *compact.GROUPS[1]) else radii
        fields["assessment_radii"] = assessed_radii
        fields["assessment_quantiles"] = np.linspace(1.1, .8, len(assessed_radii))
        if key in compact.GROUPS[1]:
            fields.update({
                "anytime_family": compact.ANYTIME_FAMILIES[key],
                "anytime_current_radius_per_episode": np.r_[2.0, radii[:-1]],
                "anytime_next_radius_per_episode": radii,
                "tube_radius_per_episode": radii - 0.2,
                "anytime_trajectory_count_per_episode": np.array([3, 5]),
                "anytime_initial_trajectory_count": 10,
                "anytime_quantile_lower_history": np.array([[0.3, 0.4, 0.5, 999, 999], [0.3, 0.4, 0.5, 0.6, 0.7]]),
                "anytime_quantile_upper_history": np.array([[np.inf, 1.5, 1.4, 999, 999], [np.inf, 1.4, 1.3, 1.2, 1.1]]),
                "anytime_termination_reason_per_episode": np.array(["conservative", "certified"]),
                "anytime_reference_quantile_per_episode": np.array([0.9, 0.95]),
            })
        np.savez(self.paths[key], **fields)

    def replace_fields(self, key, remove=(), **changes):
        with np.load(self.paths[key]) as data:
            fields = dict(data)
        for name in remove:
            fields.pop(name)
        fields.update(changes)
        np.savez(self.paths[key], **fields)

    def test_exact_series_equivalence_to_combined(self):
        # Characterize actual rendered series, not just copies of its helpers.
        with plt.rc_context():
            combined = importlib.import_module("plot_combined")
        figures = []
        argv = ["plot_combined.py", "--output_dir", str(self.root / "legacy")]
        for key, path in self.paths.items():
            argv += [f"--{key}", str(path)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(plt.Figure, "savefig"), \
                mock.patch.object(plt, "close", side_effect=lambda fig: figures.append(fig)), \
                plt.rc_context({"text.usetex": False}), contextlib.redirect_stdout(io.StringIO()):
            combined.main()
        _, series = compact.prepare_suite(self.paths)
        order = ("robust", "naive", "calibrate_once", "non_interactive", "anytime", "anytime_nonparametric")
        for index, key in enumerate(order):
            s = series[key]
            line = figures[0].axes[0].lines[index]
            np.testing.assert_array_equal(line.get_xdata(), s["radius_x"])
            np.testing.assert_array_equal(line.get_ydata(), s["radius"])
            line = figures[1].axes[0].containers[index].lines[0]
            np.testing.assert_array_equal(line.get_xdata(), s["metric_x"])
            np.testing.assert_array_equal(line.get_ydata(), s["cumulative_reward"])
            for figure, field in [(figures[2], "tube"), (figures[3], "safety")]:
                line = figure.axes[0].lines[index + 1]
                np.testing.assert_array_equal(line.get_xdata(), s["metric_x"])
                np.testing.assert_array_equal(line.get_ydata(), s[field])
            errors = compact.reward_errors(s["cumulative_reward"], s["cumulative_reward_runs"], 0.1)
            segments = figures[1].axes[0].containers[index].lines[2][0].get_segments()
            np.testing.assert_allclose(np.array(segments)[:, :, 1],
                                       np.c_[s["cumulative_reward"] - errors[0], s["cumulative_reward"] + errors[1]])

    def test_budget_counts_and_real_initial_radii(self):
        _, series = compact.prepare_suite(self.paths)
        np.testing.assert_allclose(series["robust"]["radius"], [2, 1.6, 1.45], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(series["robust"]["budget"], [1000, 2000, 3000])
        for key in ("non_interactive", "calibrate_once"):
            np.testing.assert_array_equal(series[key]["budget"], [1000, 1000])
            np.testing.assert_array_equal(series[key]["radius"], [1.6, 1.3, 1.3])
        for key in compact.GROUPS[1]:
            if key not in series:
                continue
            np.testing.assert_array_equal(series[key]["radius"], [2, 1.6, 1.3])
            np.testing.assert_array_equal(series[key]["budget"], [3, 8])
            self.assertEqual(series[key]["initial"], 10)

    def test_paper_and_legacy_one_time_labels(self):
        _, series = compact.prepare_suite(self.paths)
        self.assertEqual(series["calibrate_once"]["label"], "One-time")
        self.replace_fields("calibrate_once", update_procedure="calibrate-once")
        _, series = compact.prepare_suite(self.paths)
        self.assertEqual(series["calibrate_once"]["label"], "Calibrate-once (legacy)")

    def test_progression_percentages_targets_and_log_budgets(self):
        runs, series = compact.prepare_suite(self.paths)
        with plt.rc_context({**compact.PAPER_STYLE, "text.usetex": False}):
            fig = compact.plot_progression(runs, series, 0.1)
        self.assertEqual(len(fig.axes), 8)
        for row, keys in enumerate(compact.GROUPS):
            coverage = fig.axes[4 * row + 2]
            for key, line in zip([key for key in keys if key in series], coverage.lines):
                np.testing.assert_array_equal(line.get_ydata(), 100 * series[key]["tube"])
            np.testing.assert_array_equal(coverage.lines[-1].get_ydata(), [90, 90])
            self.assertEqual(fig.axes[4 * row + 3].get_yscale(), "log")
        budget_lines = fig.axes[7].lines
        np.testing.assert_array_equal(budget_lines[0].get_ydata(), [3, 8])
        np.testing.assert_array_equal(budget_lines[1].get_ydata(), [13, 18])
        self.assertEqual(len(fig.axes[0].lines), 8)
        for index, key in enumerate(compact.GROUPS[0]):
            line = fig.axes[0].lines[2 * index + 1]
            self.assertEqual(line.get_linestyle(), ":")
            np.testing.assert_array_equal(line.get_ydata(), series[key]["qj"])
        self.assertEqual(fig.axes[0].get_ylabel(), r"$r_j,\ q_j$ (m)")

    def test_quantile_column_uses_independent_quantile_and_final_policy_padding(self):
        runs, series = compact.prepare_suite(self.paths)
        fig = compact.plot_progression(runs, series, .1, second_panel="qj")
        self.assertEqual(len(fig.axes[0].lines), 4)
        self.assertEqual(fig.axes[1].get_ylabel(), r"$q_j$ (m)")
        self.assertEqual(fig.axes[0].get_ylabel(), r"$r_j$ (m)")
        for row, keys in enumerate(compact.GROUPS):
            for line, key in zip(fig.axes[4 * row + 1].lines, [k for k in keys if k in series]):
                np.testing.assert_array_equal(line.get_xdata(), series[key]["radius_x"])
                np.testing.assert_array_equal(line.get_ydata(), series[key]["qj"])
        np.testing.assert_allclose(series["anytime"]["qj"], [1.1, .95, .8])
        np.testing.assert_allclose(series["non_interactive"]["qj"], [1.1, .8, .8])
        np.testing.assert_allclose(series["robust"]["qj"], [1.1, 1., .9])

    def test_old_files_never_substitute_calibration_qj(self):
        self.replace_fields("robust", remove=("assessment_radii", "assessment_quantiles"))
        runs, series = compact.prepare_suite(self.paths)
        with self.assertRaisesRegex(ValueError, "Independent assessment q_j missing for: robust"):
            compact.plot_progression(runs, series, .1, second_panel="qj")
        with self.assertWarnsRegex(UserWarning, "dotted q_j omitted"):
            fig = compact.plot_progression(runs, series, .1, second_panel="reward")
        self.assertEqual(len(fig.axes[0].lines), 7)

    def test_misaligned_quantiles_rejected(self):
        self.replace_fields("anytime", assessment_radii=np.array([2., 1.6, 1.1]))
        with self.assertRaisesRegex(ValueError, "align with the policy radii"):
            compact.prepare_suite(self.paths)

    def test_second_panel_cli(self):
        argv = [arg for key, path in self.paths.items() for arg in (f"--{key}", str(path))]
        self.assertEqual(compact.parse_args(argv).second_panel, "reward")
        self.assertEqual(compact.parse_args(argv + ["--second-panel", "qj"]).second_panel, "qj")

    def test_bounds_reference_stop_infinity_and_selected_order(self):
        fig = compact.plot_bounds(self.paths["anytime"], [1, 0])
        self.assertEqual(len(fig.axes), 2)
        ax = fig.axes[1]
        lower, upper, reference, lower_stop, upper_stop = ax.lines
        np.testing.assert_array_equal(lower.get_ydata(), [0.3, 0.4, 0.5])
        self.assertTrue(np.isnan(upper.get_ydata()[0]))
        self.assertEqual(upper.get_drawstyle(), "steps-post")
        np.testing.assert_array_equal(reference.get_ydata(), [0.9, 0.9])
        np.testing.assert_array_equal(lower_stop.get_xdata(), [3])
        np.testing.assert_array_equal(upper_stop.get_ydata(), [1.4])
        self.assertNotIn("r_j", " ".join(t.get_text() for t in fig.legends[0].texts))

    def test_old_bounds_never_invent_reference(self):
        self.replace_fields("anytime", remove=("anytime_reference_quantile_per_episode",))
        with self.assertWarnsRegex(UserWarning, "dotted reference omitted"):
            fig = compact.plot_bounds(self.paths["anytime"])
        for ax in fig.axes:
            self.assertFalse(any(line.get_linestyle() == ":" for line in ax.lines))
        with self.assertRaisesRegex(ValueError, "not found"):
            compact.plot_bounds(self.paths["anytime"], [99])

    def test_spheres_and_saved_post_update_tubes(self):
        center = np.array([1, -2, 3])
        mesh = np.stack(compact.sphere_mesh(center, 0.7), axis=-1)
        np.testing.assert_allclose(np.linalg.norm(mesh - center, axis=-1), 0.7)
        runs, _ = compact.prepare_suite(self.paths)
        with mock.patch.object(compact, "sphere_mesh", wraps=compact.sphere_mesh) as spheres:
            fig = compact.plot_tube_progression(runs["anytime"], 1, 30, -60)
        self.assertEqual({round(call.args[1], 6) for call in spheres.call_args_list}, {1.4, 1.1})
        self.assertIn("r_{j+1}=1.600", fig.axes[0].get_title())
        for index, ax in enumerate(fig.axes):
            ego = np.array(ax.lines[-3].get_data_3d()).T
            np.testing.assert_array_equal(ego, runs["anytime"]["locations"][index, -1])
            self.assertEqual((ax.elev, ax.azim), (30, -60))
            np.testing.assert_array_equal(ax.get_xlim(), fig.axes[0].get_xlim())
        with self.assertRaisesRegex(ValueError, "non-ego"):
            compact.plot_tube_progression(runs["anytime"], 2, 20, 45)

    def test_final_trajectories_ego_last_and_common_limits(self):
        runs, series = compact.prepare_suite(self.paths)
        fig = compact.plot_final_trajectories(runs, series, 15, 80)
        for ax, key in zip(fig.axes, runs):
            np.testing.assert_array_equal(np.array(ax.lines[-3].get_data_3d()).T, runs[key]["locations"][-1, -1])
            for getter in ("get_xlim", "get_ylim", "get_zlim"):
                self.assertEqual(getattr(ax, getter)(), getattr(fig.axes[0], getter)())

    def test_spatial_axis_labels_are_inside_pdf_page(self):
        runs, series = compact.prepare_suite(self.paths)
        with plt.rc_context({**compact.PAPER_STYLE, "text.usetex": False}):
            figures = [compact.plot_final_trajectories(runs, series, 20, 45),
                       compact.plot_tube_progression(runs["anytime"], 0, 20, 45)]
            for fig in figures:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for ax in fig.axes:
                    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                        bounds = axis.label.get_window_extent(renderer)
                        self.assertGreaterEqual(bounds.x0, 0)
                        self.assertGreaterEqual(bounds.y0, 0)
                        self.assertLessEqual(bounds.x1, fig.bbox.x1)
                        self.assertLessEqual(bounds.y1, fig.bbox.y1)

    def test_invalid_method_alpha_alignment_and_shapes(self):
        self.replace_fields("naive", update_procedure="robust")
        with self.assertRaisesRegex(ValueError, "expected naive"):
            compact.prepare_suite(self.paths)
        self.replace_fields("naive", update_procedure="naive", alpha=0.2)
        with self.assertRaisesRegex(ValueError, "same coverage target"):
            compact.prepare_suite(self.paths)
        self.replace_fields("naive", alpha=0.1, episodes=np.arange(1, 4))
        with self.assertRaisesRegex(ValueError, "episode grid"):
            compact.prepare_suite(self.paths)
        self.replace_fields("naive", episodes=np.arange(3), radius_per_episode=np.ones(2))
        with self.assertRaisesRegex(ValueError, "radius must match"):
            compact.prepare_suite(self.paths)

    def test_disagreeing_anytime_radii_rejected(self):
        self.replace_fields("anytime", anytime_next_radius_per_episode=np.array([1.4, 1.3]))
        with self.assertRaisesRegex(ValueError, "disagree"):
            compact.prepare_suite(self.paths)

    def test_missing_supplementary_fields_skip_only_supplements(self):
        self.replace_fields("anytime", remove=("agent_locs_per_episode", "anytime_quantile_lower_history"))
        runs, series = compact.prepare_suite(self.paths)
        with self.assertWarnsRegex(UserWarning, "histories missing"):
            self.assertIsNone(compact.plot_bounds(self.paths["anytime"]))
        with self.assertWarnsRegex(UserWarning, "positions or predictions missing"):
            self.assertIsNone(compact.plot_final_trajectories(runs, series, 20, 45))

    def test_five_pdf_smoke_and_style_is_scoped(self):
        argv = ["--output_dir", str(self.root / "output")]
        for key, path in self.paths.items():
            argv += [f"--{key}", str(path)]
        before = {key: plt.rcParams[key] for key in compact.PAPER_STYLE}
        with mock.patch.dict(compact.PAPER_STYLE, {"text.usetex": False}), contextlib.redirect_stdout(io.StringIO()):
            compact.main(argv)
        expected = {"episode_progression", "ego_safety", "anytime_quantile_bounds", "final_trajectories", "anytime_tube_progression"}
        self.assertEqual({path.stem for path in (self.root / "output").glob("*.pdf")}, expected)
        for path in (self.root / "output").glob("*.pdf"):
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))
            self.assertGreater(path.stat().st_size, 1000)
        self.assertEqual(before, {key: plt.rcParams[key] for key in compact.PAPER_STYLE})

    def test_optional_mixtures_nine_methods_and_family_checks(self):
        for key in compact.OPTIONAL_METHODS:
            folder = self.root / key
            folder.mkdir()
            (folder / "conformal_args.json").write_text(json.dumps({"r_init": 2.0}))
            self.paths[key] = folder / "conformal_metrics.npz"
            self.write_run(key, "anytime")
        runs, series = compact.prepare_suite(self.paths)
        self.assertEqual(len(series), 9)
        with plt.rc_context({**compact.PAPER_STYLE, "text.usetex": False}):
            progression = compact.plot_progression(runs, series, .1)
            quantiles = compact.plot_progression(runs, series, .1, "qj")
            trajectories = compact.plot_final_trajectories(runs, series, 20, 45)
            for fig in (progression, quantiles, trajectories):
                fig.canvas.draw()
            for fig in (progression, quantiles):
                for text in fig.texts:
                    bounds = text.get_window_extent(fig.canvas.get_renderer())
                    self.assertGreaterEqual(bounds.x0, 0)
                    self.assertGreaterEqual(bounds.y0, 0)
                    self.assertLessEqual(bounds.x1, fig.bbox.x1)
                    self.assertLessEqual(bounds.y1, fig.bbox.y1)
            self.assertEqual(len(trajectories.axes), 9)
            for legend in progression.legends:
                bounds = legend.get_window_extent(progression.canvas.get_renderer())
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, progression.bbox.x1)
        labels = [text.get_text() for text in progression.legends[1].texts]
        self.assertEqual(labels, ["Mixtures:", "Jeffreys", "Informed", "Jeffreys-informed", "Directional", "Normal"])
        for key in compact.OPTIONAL_METHODS:
            np.testing.assert_array_equal(series[key]["budget"], [3, 8])
            np.testing.assert_array_equal(series[key]["radius"], [2, 1.6, 1.3])
        self.test_five_pdf_smoke_and_style_is_scoped()
        self.replace_fields("anytime_informed", anytime_family="directional")
        with self.assertRaisesRegex(ValueError, "expected family informed"):
            compact.prepare_suite(self.paths)

    def test_normal_alias_and_old_metadata(self):
        self.replace_fields("anytime_nonparametric", update_procedure="anytime")
        _, series = compact.prepare_suite(self.paths)
        self.assertEqual(series["anytime_nonparametric"]["label"], "Normal")
        self.replace_fields("anytime", remove=("anytime_family",))
        self.assertEqual(compact.prepare_suite(self.paths)[1]["anytime"]["label"], "Jeffreys")


if __name__ == "__main__":
    unittest.main()
