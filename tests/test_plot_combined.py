import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_combined import (
    main,
    pad_repeat,
    prepare_anytime_run,
    prepare_conformal_radius,
)


class PlotCombinedTests(unittest.TestCase):
    def test_legacy_alignment_helpers_are_unchanged(self):
        np.testing.assert_allclose(
            prepare_conformal_radius(np.array([4.0, 3.0, 2.0])),
            [8.0, 4.0, 3.0],
        )
        np.testing.assert_allclose(
            pad_repeat(np.array([1.0, 2.0]), 4),
            [1.0, 2.0, 2.0, 2.0],
        )

    def test_anytime_alignment_keeps_final_update_and_repeats_terminal_data(self):
        run = {
            "episodes": np.array([0, 1]),
            "radius": np.array([1.5, 1.25]),
            "cumulative_reward": np.array([0.4, 0.6]),
            "cumulative_reward_runs": np.array([[0.3, 0.5], [0.5, 0.7]]),
            "tube": np.array([0.8, 0.9]),
            "safety": np.array([0.95, 1.0]),
            "update_procedure": "anytime",
            "anytime_current_radius": np.array([2.0, 1.5]),
            "anytime_next_radius": np.array([1.5, 1.25]),
        }

        aligned = prepare_anytime_run(run, 5, "anytime")

        np.testing.assert_allclose(
            aligned["radius"],
            [2.0, 1.5, 1.25, 1.25, 1.25],
        )
        np.testing.assert_allclose(
            aligned["cumulative_reward"],
            [0.4, 0.6, 0.6, 0.6, 0.6],
        )
        np.testing.assert_allclose(
            aligned["tube"],
            [0.8, 0.9, 0.9, 0.9, 0.9],
        )
        np.testing.assert_allclose(
            aligned["cumulative_reward_runs"][-1],
            [0.5, 0.7],
        )

    def test_anytime_alignment_rejects_disagreeing_deployed_radius(self):
        run = {
            "episodes": np.array([0]),
            "radius": np.array([1.4]),
            "cumulative_reward": np.array([0.4]),
            "cumulative_reward_runs": None,
            "tube": np.array([0.9]),
            "safety": np.array([1.0]),
            "update_procedure": "anytime-nonparametric",
            "anytime_current_radius": np.array([2.0]),
            "anytime_next_radius": np.array([1.5]),
        }

        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_anytime_run(run, 3, "anytime-nonparametric")

    def test_six_method_plot_smoke(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            paths = {
                name: root / f"{name}.npz"
                for name in (
                    "robust",
                    "naive",
                    "calibrate_once",
                    "non_interactive",
                    "anytime",
                    "anytime_nonparametric",
                )
            }
            self._write_metrics(paths["robust"], "robust", 3)
            self._write_metrics(paths["naive"], "naive", 3)
            self._write_metrics(paths["calibrate_once"], "calibrate-once", 2)
            self._write_metrics(
                paths["non_interactive"], "non-interactive", 2
            )
            self._write_metrics(paths["anytime"], "anytime", 2, anytime=True)
            self._write_metrics(
                paths["anytime_nonparametric"],
                "anytime-nonparametric",
                2,
                anytime=True,
            )
            output_dir = root / "plots"
            argv = ["plot_combined.py"]
            for name, path in paths.items():
                argv.extend([f"--{name}", str(path)])
            argv.extend(["--output_dir", str(output_dir)])

            with mock.patch.object(sys, "argv", argv), plt.rc_context(
                {"text.usetex": False}
            ):
                main()

            expected = {
                "radius_across_episodes.pdf",
                "performance_cumulative_reward.pdf",
                "tube_coverage.pdf",
                "empirical_safety_coverage.pdf",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected,
            )
            for filename in expected:
                self.assertGreater((output_dir / filename).stat().st_size, 0)

    @staticmethod
    def _write_metrics(path, procedure, episode_count, anytime=False):
        episodes = np.arange(episode_count)
        radius = np.linspace(1.5, 1.25, episode_count)
        fields = {
            "episodes": episodes,
            "radius_per_episode": radius,
            "qj_per_episode": radius + 0.1,
            "tube_coverage_per_episode": np.linspace(
                0.9, 1.0, episode_count
            ),
            "safety_per_episode": np.ones(episode_count),
            "cumulative_reward_per_episode": np.linspace(
                0.4, 0.6, episode_count
            ),
            "cumulative_reward_per_run": np.tile(
                np.array([[0.3, 0.7]]), (episode_count, 1)
            ),
            "alpha": 0.1,
            "update_procedure": procedure,
        }
        if anytime:
            fields["anytime_current_radius_per_episode"] = np.concatenate(
                [[2.0], radius[:-1]]
            )
            fields["anytime_next_radius_per_episode"] = radius
        np.savez(path, **fields)


if __name__ == "__main__":
    unittest.main()
