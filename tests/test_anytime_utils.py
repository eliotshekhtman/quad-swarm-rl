import itertools
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_function

from project_utils.anytime_utils import (
    BRANCH_CERTIFICATE,
    BRANCH_CONSERVATIVE,
    BRANCH_UNRESOLVED,
    E_PROCESS_JEFFREYS,
    E_PROCESS_NORMAL,
    QuantileConfidenceSequence,
    candidate_bracket,
    clopper_pearson_interval,
    contraction_factors,
    critical_exceedance_counts,
    episode_error_budget,
    jeffreys_log_e_value,
    log_e_value,
    max_active_episodes,
    normal_log_e_value,
    run_anytime_episode,
    select_branch,
)


class EProcessTests(unittest.TestCase):
    def test_jeffreys_formula_matches_direct_mixture_integral(self):
        alpha = 0.2
        for n in range(1, 7):
            for k in range(n + 1):
                numerator, _ = quad(
                    lambda angle: (
                        2.0
                        * np.sin(angle) ** (2 * k)
                        * np.cos(angle) ** (2 * (n - k))
                        / beta_function(0.5, 0.5)
                    ),
                    0.0,
                    np.pi / 2.0,
                )
                denominator = alpha**k * (1.0 - alpha) ** (n - k)
                expected = np.log(numerator / denominator)
                self.assertAlmostEqual(
                    jeffreys_log_e_value(n, k, alpha), expected, places=10
                )

    def test_normal_formula_matches_direct_gaussian_integral(self):
        alpha = 0.2
        gaussian_scale = np.sqrt(8.0 * np.pi)
        for n in range(1, 6):
            for k in range(n + 1):
                centered = k - n * alpha
                integral, _ = quad(
                    lambda value: (
                        np.exp(
                            value * centered
                            - n * value**2 / 8.0
                            - value**2 / 8.0
                        )
                        / gaussian_scale
                    ),
                    -20.0,
                    20.0,
                )
                self.assertAlmostEqual(
                    normal_log_e_value(n, k, alpha),
                    np.log(integral),
                    places=10,
                )

    def test_critical_counts_match_exhaustive_search(self):
        delta = 0.03
        boundary = -np.log(delta)
        for family in (E_PROCESS_JEFFREYS, E_PROCESS_NORMAL):
            for alpha in (0.05, 0.17, 0.5, 0.83, 0.95):
                for n in range(31):
                    values = np.asarray(
                        log_e_value(family, n, np.arange(n + 1), alpha)
                    )
                    lower = [
                        k
                        for k in range(n + 1)
                        if k <= n * alpha and values[k] >= boundary
                    ]
                    upper = [
                        k
                        for k in range(n + 1)
                        if k >= n * alpha and values[k] >= boundary
                    ]
                    expected = (
                        max(lower, default=-1),
                        min(upper, default=n + 1),
                    )
                    self.assertEqual(
                        critical_exceedance_counts(
                            n,
                            alpha=alpha,
                            delta=delta,
                            family=family,
                        ),
                        expected,
                    )

    def test_zero_error_budget_never_rejects(self):
        for family in (E_PROCESS_JEFFREYS, E_PROCESS_NORMAL):
            self.assertEqual(
                critical_exceedance_counts(
                    20,
                    alpha=0.1,
                    delta=0.0,
                    family=family,
                ),
                (-1, 21),
            )


class QuantileConfidenceSequenceTests(unittest.TestCase):
    @staticmethod
    def _order_statistic(scores, index):
        if index <= 0:
            return 0.0
        if index > len(scores):
            return np.inf
        return sorted(scores)[index - 1]

    def test_bounds_match_prefix_by_prefix_order_statistic_definition(self):
        scores = [0.7, 0.2, 1.1, 0.2, 0.9, 0.4, 1.3, 0.5]
        state = QuantileConfidenceSequence(
            alpha=0.2,
            delta=0.2,
            family=E_PROCESS_JEFFREYS,
            max_samples=len(scores),
        )
        expected_lower = 0.0
        expected_upper = np.inf
        prefix = []
        for score in scores:
            prefix.append(score)
            n = len(prefix)
            few, many = critical_exceedance_counts(
                n,
                alpha=0.2,
                delta=0.2,
                family=E_PROCESS_JEFFREYS,
            )
            expected_lower = max(
                expected_lower,
                self._order_statistic(prefix, n - many + 1),
            )
            expected_upper = min(
                expected_upper,
                self._order_statistic(prefix, n - few),
            )
            lower, upper, actual_few, actual_many = state.update(score)
            self.assertEqual((actual_few, actual_many), (few, many))
            self.assertEqual(lower, expected_lower)
            self.assertEqual(upper, expected_upper)

    def test_running_intersection_is_monotone_for_both_families(self):
        scores = np.random.default_rng(4).uniform(0.0, 2.0, size=100)
        for family in (E_PROCESS_JEFFREYS, E_PROCESS_NORMAL):
            state = QuantileConfidenceSequence(
                alpha=0.1,
                delta=0.05,
                family=family,
                max_samples=len(scores),
            )
            lowers = []
            uppers = []
            for score in scores:
                lower, upper, _, _ = state.update(score)
                lowers.append(lower)
                uppers.append(upper)
            self.assertTrue(
                all(current >= previous for previous, current in zip(lowers, lowers[1:]))
            )
            self.assertTrue(
                all(current <= previous for previous, current in zip(uppers, uppers[1:]))
            )

    def test_exact_finite_horizon_miscoverage_is_below_delta(self):
        alpha = 0.2
        delta = 0.2
        horizon = 10
        true_quantile = 0.5
        for family in (E_PROCESS_JEFFREYS, E_PROCESS_NORMAL):
            failure_probability = 0.0
            for outcomes in itertools.product((0, 1), repeat=horizon):
                state = QuantileConfidenceSequence(
                    alpha=alpha,
                    delta=delta,
                    family=family,
                    max_samples=horizon,
                )
                failed = False
                for outcome in outcomes:
                    lower, upper, _, _ = state.update(
                        1.5 if outcome else true_quantile
                    )
                    failed |= not (lower <= true_quantile <= upper)
                if failed:
                    successes = sum(outcomes)
                    failure_probability += (
                        alpha**successes
                        * (1.0 - alpha) ** (horizon - successes)
                    )
            self.assertLessEqual(failure_probability, delta)


class BudgetAndCalibrationTests(unittest.TestCase):
    def test_contraction_and_positive_tolerance_budget(self):
        gamma, zeta = contraction_factors(0.6)
        self.assertAlmostEqual(gamma, 0.75)
        self.assertAlmostEqual(zeta, 0.875)
        k_epsilon = max_active_episodes(2.0, 0.15, 0.6)
        self.assertEqual(k_epsilon, 20)
        budgets = [
            episode_error_budget(0.05, episode, k_epsilon)
            for episode in range(k_epsilon + 2)
        ]
        self.assertAlmostEqual(sum(budgets), 0.05)
        self.assertEqual(budgets[-1], 0.0)

    def test_zero_tolerance_budget_is_geometric(self):
        budgets = [episode_error_budget(0.2, episode, None) for episode in range(4)]
        np.testing.assert_allclose(budgets, [0.1, 0.05, 0.025, 0.0125])

    def test_zero_exceedance_clopper_pearson_endpoint(self):
        lower, upper = clopper_pearson_interval(0, 100, 0.05)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 1.0 - (0.05 / 2.0) ** (1.0 / 100.0))


class BracketUpdateTests(unittest.TestCase):
    def test_candidate_matches_equation_33(self):
        candidate = candidate_bracket(
            radius=2.0,
            lower_radius=0.2,
            quantile_lower=0.7,
            quantile_upper=1.0,
            kappa=0.6,
        )
        self.assertAlmostEqual(candidate.lower_radius, 0.2)
        self.assertAlmostEqual(candidate.radius, 1.375)
        self.assertAlmostEqual(candidate.width, 1.175)

    def test_status_prioritizes_certificate_over_contraction(self):
        candidate = candidate_bracket(
            radius=1.0,
            lower_radius=0.5,
            quantile_lower=0.6,
            quantile_upper=0.6,
            kappa=0.6,
        )
        self.assertEqual(
            select_branch(
                candidate=candidate,
                current_width=0.5,
                quantile_upper=0.6,
                radius=1.0,
                radius_tolerance=0.3,
                zeta=0.875,
            ),
            BRANCH_CERTIFICATE,
        )

    def test_candidate_contains_known_fixed_point_and_exactly_contracts(self):
        kappa = 0.6
        gamma, _ = contraction_factors(kappa)
        fixed_point = 0.4
        for radius in (0.6, 1.0, 2.0, 8.0):
            quantile = (
                kappa * radius + (1.0 - kappa) * fixed_point
            )
            candidate = candidate_bracket(
                radius=radius,
                lower_radius=fixed_point,
                quantile_lower=quantile,
                quantile_upper=quantile,
                kappa=kappa,
            )
            self.assertAlmostEqual(candidate.lower_radius, fixed_point)
            self.assertGreaterEqual(candidate.radius + 1.0e-12, fixed_point)
            self.assertAlmostEqual(
                candidate.width,
                gamma * (radius - fixed_point),
            )

    @staticmethod
    def _run(sample_score, **overrides):
        arguments = {
            "radius": 2.0,
            "lower_radius": 0.2,
            "alpha": 0.1,
            "delta": 0.05,
            "kappa": 0.6,
            "radius_tolerance": 0.1,
            "max_samples": 1000,
            "family": E_PROCESS_JEFFREYS,
            "check_every": 1,
        }
        arguments.update(overrides)
        return run_anytime_episode(sample_score, **arguments)

    def test_conservative_stream_stops_and_contracts_bracket(self):
        result = self._run(lambda: 0.2)
        self.assertEqual(result.branch, BRANCH_CONSERVATIVE)
        self.assertEqual(result.termination_reason, BRANCH_CONSERVATIVE)
        self.assertLess(result.num_samples, 1000)
        self.assertLess(result.next_radius, result.radius)
        self.assertLessEqual(result.next_width, 0.875 * result.current_width)

    def test_unsafe_stream_terminates_unresolved_without_increasing_radius(self):
        result = self._run(lambda: 3.0)
        self.assertEqual(result.branch, BRANCH_UNRESOLVED)
        self.assertEqual(result.termination_reason, "inconsistent-bounds")
        self.assertEqual(result.next_radius, result.radius)
        self.assertEqual(result.next_lower_radius, result.lower_radius)

    def test_sample_cap_retains_current_bracket(self):
        result = self._run(lambda: 0.2, max_samples=1)
        self.assertEqual(result.branch, BRANCH_UNRESOLVED)
        self.assertEqual(result.termination_reason, "sample-cap")
        self.assertEqual(result.next_radius, result.radius)
        self.assertEqual(result.next_lower_radius, result.lower_radius)

    def test_initially_narrow_bracket_certifies_without_sampling(self):
        calls = 0

        def sample():
            nonlocal calls
            calls += 1
            return 0.5

        result = self._run(
            sample,
            radius=1.0,
            lower_radius=0.9,
            radius_tolerance=0.1,
        )
        self.assertEqual(result.branch, BRANCH_CERTIFICATE)
        self.assertEqual(result.num_samples, 0)
        self.assertEqual(calls, 0)

    def test_score_equal_to_radius_is_not_an_exceedance(self):
        result = self._run(lambda: 2.0, max_samples=1)
        np.testing.assert_array_equal(result.indicators, [0])

    def test_nonparametric_family_uses_same_update_engine(self):
        result = self._run(
            lambda: 0.2,
            family=E_PROCESS_NORMAL,
            max_samples=2000,
        )
        self.assertIn(result.branch, (BRANCH_CERTIFICATE, BRANCH_CONSERVATIVE))
        self.assertEqual(result.family, E_PROCESS_NORMAL)

    def test_both_families_contract_to_a_certificate_across_episodes(self):
        for family in (E_PROCESS_JEFFREYS, E_PROCESS_NORMAL):
            radius = 2.0
            lower_radius = 0.1
            tolerance = 0.15
            k_epsilon = max_active_episodes(
                radius - lower_radius,
                tolerance,
                0.6,
            )
            for episode in range(k_epsilon):
                previous_radius = radius
                previous_lower = lower_radius
                result = run_anytime_episode(
                    lambda: 0.4,
                    radius=radius,
                    lower_radius=lower_radius,
                    alpha=0.1,
                    delta=episode_error_budget(0.05, episode, k_epsilon),
                    kappa=0.6,
                    radius_tolerance=tolerance,
                    max_samples=1000,
                    family=family,
                    check_every=10,
                )
                radius = result.next_radius
                lower_radius = result.next_lower_radius
                self.assertLessEqual(radius, previous_radius)
                self.assertGreaterEqual(lower_radius, previous_lower)
                if result.branch == BRANCH_CERTIFICATE:
                    break
                self.assertEqual(result.branch, BRANCH_CONSERVATIVE)
                self.assertLessEqual(
                    result.next_width,
                    0.875 * result.current_width,
                )
            self.assertEqual(result.branch, BRANCH_CERTIFICATE)
            self.assertLessEqual(result.next_width, tolerance)


if __name__ == "__main__":
    unittest.main()
