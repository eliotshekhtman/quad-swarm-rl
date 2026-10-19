import unittest

import numpy as np
from scipy.special import betaln

from project_utils.anytime_utils import (
    BRANCH_CERTIFICATE,
    BRANCH_CONSERVATIVE,
    BRANCH_UNDER,
    BRANCH_UNRESOLVED,
    BernoulliAnytimeState,
    episode_error_budget,
    probability_slopes,
    run_anytime_episode,
    select_branch,
    update_radius,
)


class AnytimeConfidenceTests(unittest.TestCase):
    def test_episode_budgets_are_dyadic(self):
        budgets = [episode_error_budget(0.2, episode) for episode in range(4)]
        np.testing.assert_allclose(budgets, [0.1, 0.05, 0.025, 0.0125])

    def test_fixed_kappa_and_density_bounds_define_probability_slopes(self):
        lower, upper = probability_slopes(0.6, 0.8, 1.25)
        self.assertAlmostEqual(lower, 0.32)
        self.assertAlmostEqual(upper, 2.0)

    def test_grid_uses_closed_form_beta_mixture_and_never_reactivates(self):
        state = BernoulliAnytimeState(
            alpha=0.1,
            delta=0.05,
            nominal_forecast=0.05,
            grid_size=19,
        )
        state.update(0)
        active_after_one = state.active.copy()

        expected_log_values = (
            betaln(0.5, 1.5)
            - betaln(0.5, 0.5)
            - np.log1p(-state.grid)
        )
        np.testing.assert_array_equal(
            active_after_one,
            expected_log_values < np.log(1.0 / 0.05),
        )

        state.update(1)
        self.assertTrue(np.all(~state.active | active_after_one))
        self.assertLessEqual(0.0, state.lower)
        self.assertLessEqual(state.lower, state.upper)
        self.assertLessEqual(state.upper, 1.0)

    def test_half_count_forecast_uses_stopped_episode_denominator(self):
        state = BernoulliAnytimeState(
            alpha=0.1,
            delta=0.05,
            nominal_forecast=0.05,
            grid_size=19,
        )
        state.update(1)
        state.update(0)
        self.assertAlmostEqual(state.stopped_forecast, 1.5 / 3.0)


class AnytimeBranchTests(unittest.TestCase):
    def test_branch_order_prioritizes_undercoverage_then_certificate(self):
        common = {
            "alpha": 0.1,
            "safe": True,
            "lower_slope": 0.4,
            "radius_tolerance": 0.1,
            "relative_precision": 3.0,
        }
        self.assertEqual(
            select_branch(lower=0.11, upper=0.2, **common),
            BRANCH_UNDER,
        )
        self.assertEqual(
            select_branch(lower=0.07, upper=0.08, **common),
            BRANCH_CERTIFICATE,
        )
        self.assertEqual(
            select_branch(lower=0.01, upper=0.06, **common),
            BRANCH_CONSERVATIVE,
        )
        self.assertEqual(
            select_branch(lower=0.0, upper=0.5, **common),
            BRANCH_UNRESOLVED,
        )

    def test_undercoverage_recovery_uses_upper_endpoint(self):
        radius, safe_override = update_radius(
            branch=BRANCH_UNDER,
            radius=1.0,
            upper=0.4,
            alpha=0.1,
            lower_slope=0.4,
            upper_slope=1.6,
            eta=0.5,
            min_radius=0.5,
            max_radius=2.0,
        )
        self.assertAlmostEqual(radius, 1.75)
        self.assertTrue(safe_override)

    def test_conservative_update_uses_upper_endpoint_and_gain(self):
        radius, safe_override = update_radius(
            branch=BRANCH_CONSERVATIVE,
            radius=1.0,
            upper=0.04,
            alpha=0.1,
            lower_slope=0.4,
            upper_slope=1.6,
            eta=0.5,
            min_radius=0.5,
            max_radius=2.0,
        )
        self.assertAlmostEqual(radius, 0.98125)
        self.assertTrue(safe_override)

    @staticmethod
    def _run(sample_score, **overrides):
        arguments = {
            "radius": 1.0,
            "safe": True,
            "alpha": 0.1,
            "delta": 0.05,
            "lower_slope": 0.4,
            "upper_slope": 1.6,
            "relative_precision": 3.0,
            "radius_tolerance": 0.0,
            "eta_min": 0.2,
            "damping_gamma": 0.02,
            "min_radius": 0.5,
            "max_radius": 2.0,
            "max_samples": 1000,
            "nominal_forecast": 0.05,
            "grid_size": 601,
            "check_every": 1,
        }
        arguments.update(overrides)
        return run_anytime_episode(sample_score, **arguments)

    def test_zero_exceedances_stop_early_and_reduce_radius(self):
        result = self._run(lambda: 0.0)
        self.assertEqual(result.branch, BRANCH_CONSERVATIVE)
        self.assertLess(result.num_samples, 1000)
        self.assertEqual(result.exceedances, 0)
        self.assertLess(result.next_radius, result.radius)
        self.assertGreaterEqual(result.eta, 0.2)
        self.assertLessEqual(result.eta, 1.0)

    def test_repeated_exceedances_stop_early_and_increase_radius(self):
        result = self._run(lambda: 2.0, safe=False)
        self.assertEqual(result.branch, BRANCH_UNDER)
        self.assertLess(result.num_samples, 1000)
        self.assertEqual(result.exceedances, result.num_samples)
        self.assertGreater(result.next_radius, result.radius)
        self.assertTrue(result.next_safe)

    def test_cap_retains_radius_when_no_branch_is_resolved(self):
        result = self._run(lambda: 0.0, max_samples=1)
        self.assertEqual(result.branch, BRANCH_UNRESOLVED)
        self.assertEqual(result.next_radius, result.radius)

    def test_certificate_stops_and_retains_radius(self):
        result = self._run(
            lambda: 0.0,
            max_samples=10,
            radius_tolerance=0.25,
        )
        self.assertEqual(result.branch, BRANCH_CERTIFICATE)
        self.assertEqual(result.num_samples, 1)
        self.assertEqual(result.next_radius, result.radius)

    def test_score_equal_to_radius_is_coverage(self):
        result = self._run(lambda: 1.0, max_samples=1)
        np.testing.assert_array_equal(result.indicators, [0])
        self.assertEqual(len(result.lower_history), result.num_samples)
        self.assertEqual(len(result.upper_history), result.num_samples)


if __name__ == "__main__":
    unittest.main()
