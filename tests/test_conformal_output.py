import tempfile
import unittest
from argparse import Namespace

import numpy as np

from conformal import (
    _empty_metrics,
    _record_anytime_episode,
    _record_episode,
    _save_metrics,
    _validate_predictions,
)
from project_utils.anytime_utils import AnytimeEpisodeResult, initialize_mixture_prior, run_anytime_episode


class ConformalOutputTests(unittest.TestCase):
    def test_independent_assessments_include_initial_and_final_policies(self):
        summary = dict(tube_coverage=.9, crashes=0, bad_crashes=0, safety=1,
                       cumulative_reward=2, cumulative_reward_per_run=np.ones(10),
                       agent_locs=np.zeros((2, 2, 3)))
        for procedure in ("robust", "naive", "anytime", "anytime-nonparametric",
                          "non-interactive", "calibrate-once", "calibrate-once-paper"):
            with self.subTest(procedure=procedure):
                metrics = _empty_metrics()
                iterative = procedure in ("robust", "naive", "anytime", "anytime-nonparametric")
                radii = [1.7, 1.6] if iterative else [2, 1.6]
                scores = np.arange(10, dtype=float) / 10
                for index, radius in enumerate(radii):
                    _record_episode(metrics, 9.0, radius, summary, np.zeros((1, 2, 6)),
                                    scores + index)
                expected_scores = np.array([scores, scores + 1])
                if iterative:
                    metrics["initial_reference_scores"] = scores + 2
                    expected_scores = np.vstack([scores + 2, expected_scores])
                    radii = [2, *radii]
                args = Namespace(alpha=.1, delta=.05, kappa=.2, cbf_k1=4,
                                 cbf_solver="legacy", update_procedure=procedure, r_init=2)
                state = np.random.get_state()
                with tempfile.TemporaryDirectory(dir=".") as directory:
                    path = _save_metrics(directory, args, [0, 1], metrics, .05)
                    with np.load(path) as data:
                        np.testing.assert_array_equal(data["assessment_radii"], radii)
                        np.testing.assert_array_equal(data["assessment_scores"], expected_scores)
                        np.testing.assert_array_equal(data["assessment_quantiles"], expected_scores[:, 8])
                        np.testing.assert_array_equal(data["qj_per_episode"], [9, 9])
                        self.assertNotEqual(data["assessment_quantiles"][-1], expected_scores[-1, 9])
                after = np.random.get_state()
                for before_value, after_value in zip(state, after):
                    np.testing.assert_array_equal(before_value, after_value)

    def test_assessment_scores_are_copied(self):
        metrics = _empty_metrics()
        scores = np.array([.3, .4])
        summary = dict(tube_coverage=1, crashes=0, bad_crashes=0, safety=1,
                       cumulative_reward=2, cumulative_reward_per_run=np.ones(2),
                       agent_locs=np.zeros((2, 2, 3)))
        _record_episode(metrics, .5, 1, summary, np.zeros((1, 2, 6)), scores)
        scores[:] = 99
        np.testing.assert_array_equal(metrics["evaluation_scores_per_episode"], [[.3, .4]])

    def test_informed_prior_metadata_round_trip(self):
        metrics = _empty_metrics()
        scores = np.array([.4, .5, .6, .7])
        prior = initialize_mixture_prior(scores, radius=2, kappa=.6)
        result = run_anytime_episode(lambda: .5, radius=2, lower_radius=.2,
            alpha=.1, delta=.05, kappa=.6, radius_tolerance=.15, max_samples=100,
            family="directional", prior=prior)
        _record_anytime_episode(metrics, result, .05, .2)
        metrics["anytime_config"] = dict(
            family="directional", kappa=.6, gamma=.75, zeta=.875,
            radius_tolerance=.15, k_epsilon=15, min_radius=.2, initial_width=1.8,
            initial_safety_mode="clopper-pearson", delta_init=.05,
            initial_trajectory_count=4, initial_exceedance_count=0,
            initial_probability_lower=0, initial_probability_upper=.05,
            initial_scores=scores, initial_indicators=np.zeros(4, dtype=np.int8),
        )
        metrics["anytime_run_termination_reason"] = "driver-cap"
        args = Namespace(alpha=.1, delta=.05, kappa=.6, cbf_k1=4,
                         cbf_solver="legacy", update_procedure="anytime")
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = _save_metrics(directory, args, [0], metrics, np.nan)
            with np.load(path) as data:
                self.assertEqual(data["anytime_family"].item(), "directional")
                for field in ("a", "b", "mean", "concentration", "source_count", "source_radius",
                              "shift", "near_fraction", "jeffreys_weight", "directional_weight"):
                    np.testing.assert_array_equal(data[f"anytime_prior_{field}_per_episode"],
                                                  [getattr(prior, field)])
                np.testing.assert_array_equal(data["anytime_initial_scores"], scores)
                np.testing.assert_array_equal(data["anytime_scores"][0], result.scores)

    def test_metrics_file_preserves_legacy_schema_and_records_procedure(self):
        metrics = _empty_metrics()
        summary = {
            "tube_coverage": 0.5,
            "crashes": 1.0,
            "bad_crashes": 0.0,
            "safety": 1.0,
            "cumulative_reward": 2.0,
            "cumulative_reward_per_run": np.array([2.0], dtype=np.float32),
            "agent_locs": np.zeros((2, 2, 3)),
        }
        predictions = np.zeros((1, 2, 6))
        _record_episode(metrics, 0.25, 0.5, summary, predictions)
        args = Namespace(
            alpha=0.1,
            delta=0.2,
            kappa=0.5,
            cbf_k1=4.0,
            cbf_solver="legacy",
            update_procedure="naive",
        )

        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = _save_metrics(
                directory,
                args,
                episode_ids=[0],
                metrics=metrics,
                alpha_bar=0.05,
            )
            with np.load(path) as data:
                expected_keys = {
                    "episodes",
                    "qj_per_episode",
                    "radius_per_episode",
                    "tube_coverage_per_episode",
                    "crashes_per_episode",
                    "bad_crashes_per_episode",
                    "safety_per_episode",
                    "cumulative_reward_per_episode",
                    "cumulative_reward_per_run",
                    "agent_locs_per_episode",
                    "predicted_traj_per_episode",
                    "alpha",
                    "delta",
                    "bar_alpha",
                    "kappa",
                    "cbf_k1",
                    "cbf_solver",
                    "update_procedure",
                }
                self.assertEqual(set(data.files), expected_keys)
                self.assertEqual(data["update_procedure"].item(), "naive")
                self.assertEqual(data["kappa"].item(), 0.5)
                self.assertEqual(data["cbf_k1"].item(), 4.0)
                self.assertEqual(data["cbf_solver"].item(), "legacy")
                self.assertEqual(data["cumulative_reward_per_run"].shape, (1, 1))
                self.assertEqual(
                    data["predicted_traj_per_episode"].shape,
                    (1, 1, 2, 6),
                )

    def test_prediction_validation_rejects_short_trajectories(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            _validate_predictions(
                np.zeros((2, 3, 6)),
                num_multi_agents=2,
                episode_length=4,
            )

    def test_anytime_metrics_include_stopping_diagnostics(self):
        metrics = _empty_metrics()
        summary = {
            "tube_coverage": 1.0,
            "crashes": 0.0,
            "bad_crashes": 0.0,
            "safety": 1.0,
            "cumulative_reward": 2.0,
            "cumulative_reward_per_run": np.array([2.0], dtype=np.float32),
            "agent_locs": np.zeros((2, 2, 3)),
        }
        _record_episode(
            metrics,
            np.nan,
            1.2,
            summary,
            np.zeros((1, 2, 6)),
        )
        result = AnytimeEpisodeResult(
            branch="conservative",
            termination_reason="conservative",
            family="jeffreys",
            radius=1.3,
            lower_radius=0.2,
            next_radius=1.2,
            next_lower_radius=0.3,
            current_width=1.1,
            next_width=0.9,
            num_samples=2,
            exceedances=0,
            quantile_lower=0.7,
            quantile_upper=1.0,
            radius_certificate=1.0,
            scores=np.array([0.7, 0.8]),
            indicators=np.array([0, 0], dtype=np.int8),
            quantile_lower_history=np.array([0.0, 0.7]),
            quantile_upper_history=np.array([np.inf, 1.0]),
            few_critical_history=np.array([-1, 0]),
            many_critical_history=np.array([2, 2]),
            candidate_radius_history=np.array([1.3, 1.2]),
            candidate_lower_radius_history=np.array([0.2, 0.3]),
            candidate_width_history=np.array([1.1, 0.9]),
        )
        _record_anytime_episode(metrics, result, episode_delta=0.05, min_radius=0.2)
        metrics["anytime_config"] = {
            "family": "jeffreys",
            "kappa": 0.6,
            "gamma": 0.75,
            "zeta": 0.875,
            "radius_tolerance": 0.15,
            "k_epsilon": 15,
            "min_radius": 0.2,
            "initial_width": 1.8,
            "initial_safety_mode": "clopper-pearson",
            "delta_init": 0.05,
            "initial_trajectory_count": 2,
            "initial_exceedance_count": 0,
            "initial_probability_lower": 0.0,
            "initial_probability_upper": 0.8,
            "initial_scores": np.array([0.7, 0.8]),
            "initial_indicators": np.array([0, 0], dtype=np.int8),
        }
        metrics["anytime_run_termination_reason"] = "driver-cap"
        metrics["anytime_reference_scores"] = [[0.3, 0.7, 0.9, 1.1]]
        args = Namespace(
            alpha=0.1,
            delta=0.1,
            kappa=0.6,
            cbf_k1=2.5,
            cbf_solver="cascade",
            update_procedure="anytime",
        )

        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = _save_metrics(
                directory,
                args,
                episode_ids=[0],
                metrics=metrics,
                alpha_bar=np.nan,
            )
            with np.load(path) as data:
                self.assertEqual(
                    data["anytime_branch_per_episode"].item(),
                    "conservative",
                )
                np.testing.assert_array_equal(
                    data["anytime_trajectory_count_per_episode"],
                    [2],
                )
                np.testing.assert_allclose(data["anytime_scores"], [[0.7, 0.8]])
                np.testing.assert_array_equal(data["anytime_indicators"], [[0, 0]])
                np.testing.assert_allclose(
                    data["anytime_sample_radii"], [[1.3, 1.3]]
                )
                np.testing.assert_allclose(
                    data["anytime_quantile_upper_history"], [[np.inf, 1.0]]
                )
                np.testing.assert_array_equal(
                    data["anytime_few_critical_history"], [[-1, 0]]
                )
                self.assertAlmostEqual(
                    data["anytime_next_radius_per_episode"].item(), 1.2
                )
                self.assertAlmostEqual(data["tube_radius_per_episode"].item(), 1.0)
                self.assertEqual(data["anytime_schema_version"].item(), 2)
                self.assertEqual(data["anytime_family"].item(), "jeffreys")
                np.testing.assert_array_equal(
                    data["anytime_reference_scores"], [[0.3, 0.7, 0.9, 1.1]],
                )
                np.testing.assert_array_equal(
                    data["anytime_reference_quantile_per_episode"], [1.1],
                )
                self.assertEqual(
                    data["anytime_run_termination_reason"].item(),
                    "driver-cap",
                )
                self.assertNotIn("anytime_lower_probability_per_episode", data)
                self.assertNotIn("anytime_density_lower", data)


if __name__ == "__main__":
    unittest.main()
