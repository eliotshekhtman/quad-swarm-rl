import itertools
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, betainc
from scipy.stats import binom

from project_utils.anytime_utils import (
    INFORMED_FAMILIES, MixturePrior, QuantileConfidenceSequence,
    _log_beta_tail, critical_exceedance_counts, initialize_mixture_prior,
    jeffreys_log_e_value, log_e_value, precompute_critical_counts,
    run_anytime_episode,
)


class MixtureTests(unittest.TestCase):
    def test_equation_24_strict_exceedances_and_inclusive_shift(self):
        prior = initialize_mixture_prior(
            [0.0, 0.5, 1.0, 1.5], radius=1.0, previous_radius=2.0, kappa=0.5,
        )
        self.assertEqual(prior.source_count, 4)
        self.assertEqual(prior.source_radius, 2)
        self.assertEqual(prior.shift, 0.5)
        self.assertEqual(prior.near_fraction, 0.75)
        self.assertAlmostEqual(prior.mean, 2 / 6)
        self.assertAlmostEqual(prior.concentration, 1)
        concentrated = initialize_mixture_prior([0.0] * 100, radius=1, kappa=0.6)
        self.assertEqual(concentrated.shift, 0)
        self.assertEqual(concentrated.near_fraction, 0)
        self.assertAlmostEqual(concentrated.concentration, 100)
        tied = initialize_mixture_prior([1.0] * 100, radius=1, kappa=0.6)
        self.assertEqual(tied.near_fraction, 1)
        self.assertAlmostEqual(tied.concentration, 1)
        interior = initialize_mixture_prior(
            [0] * 6 + [1] * 2 + [2] * 2, radius=1, kappa=0.2,
        )
        self.assertAlmostEqual(interior.concentration, 10)  # Upper clip.
        interior = initialize_mixture_prior(
            [0] * 4 + [1] * 4 + [2] * 2, radius=1, kappa=0.2,
        )
        self.assertAlmostEqual(interior.concentration, 0.25 * 0.75 / 0.2**2 - 1)

    def test_prior_rejects_missing_padded_or_invalid_inputs(self):
        for scores in ([], [np.nan], [np.inf], [-1], [[1]]):
            with self.assertRaises(ValueError):
                initialize_mixture_prior(scores, radius=1, kappa=0.6)
        for kwargs in ({"a": 0}, {"b": np.inf}, {"jeffreys_weight": 0},
                       {"jeffreys_weight": np.nan}, {"directional_weight": 1}):
            with self.assertRaises(ValueError):
                MixturePrior(**{**dict(a=1, b=1), **kwargs})
        for family in INFORMED_FAMILIES:
            with self.assertRaises(ValueError):
                log_e_value(family, 1, 0, 0.1)

    def test_all_mixtures_match_independent_quadrature(self):
        alpha = 0.2
        prior = MixturePrior(2.3, 5.1, jeffreys_weight=0.35, directional_weight=0.7)
        # theta=sin(angle)^2 removes the Jeffreys endpoint singularities.
        def density(angle, family):
            t = np.sin(angle)**2
            informed = np.exp((prior.a - 1)*np.log(t) + (prior.b - 1)*np.log1p(-t)
                              - betaln(prior.a, prior.b)) * 2*np.sin(angle)*np.cos(angle)
            if family == "informed":
                return informed
            return prior.jeffreys_weight * 2/np.pi + (1-prior.jeffreys_weight)*informed

        split = np.arcsin(np.sqrt(alpha))
        mass, _ = quad(lambda x: density(x, "directional"), 0, split)
        for family in INFORMED_FAMILIES:
            for n in range(7):
                for k in range(n + 1):
                    def integrand(x):
                        t = np.sin(x)**2
                        return (t/alpha)**k * ((1-t)/(1-alpha))**(n-k) * density(x, family)
                    lower = quad(integrand, 0, split)[0]
                    upper = quad(integrand, split, np.pi/2)[0]
                    expected = (prior.directional_weight * lower/mass
                                + (1-prior.directional_weight)*upper/(1-mass)
                                if family == "directional" else lower + upper)
                    self.assertAlmostEqual(log_e_value(family, n, k, alpha, prior),
                                           np.log(expected), places=9)

    def test_mixture_identities_and_normalization(self):
        for family in INFORMED_FAMILIES:
            self.assertAlmostEqual(log_e_value(family, 0, 0, .1, MixturePrior(.7, 200)), 0)
        for n in (1, 31, 8000):
            counts = np.array([0, n//10, n//2, n])
            expected = jeffreys_log_e_value(n, counts, .1)
            np.testing.assert_array_equal(
                log_e_value("jeffreys-informed", n, counts, .1, MixturePrior(3, 7, 1)), expected,
            )
            np.testing.assert_array_equal(
                log_e_value("informed", n, counts, .1, MixturePrior(.5, .5)), expected,
            )
            rho = .3*betainc(.5, .5, .1) + .7*betainc(3, 7, .1)
            prior = MixturePrior(3, 7, .3, rho)
            np.testing.assert_allclose(log_e_value("directional", n, counts, .1, prior),
                                       log_e_value("jeffreys-informed", n, counts, .1, prior),
                                       atol=2e-10, rtol=1e-12)

    def test_underflowed_beta_tails_against_binomial_log_sums(self):
        # For integer shapes, I_x(a,b)=P[Bin(a+b-1,x)>=a]. Compute the
        # independent log-sum explicitly, including tails far below float tiny.
        from scipy.special import logsumexp
        for a, b, x in ((4000, 4001, .1), (7500, 501, .1), (20, 7981, .9)):
            n = a+b-1
            lower = logsumexp(binom.logpmf(np.arange(a, n+1), n, x))
            upper = logsumexp(binom.logpmf(np.arange(a), n, x))
            self.assertAlmostEqual(float(_log_beta_tail(a, b, x)), lower, places=8)
            self.assertAlmostEqual(float(_log_beta_tail(a, b, x, upper=True)), upper, places=8)

    def test_critical_counts_match_exhaustive_search_and_cache_uses_prior(self):
        for prior in (MixturePrior(.1, 100), MixturePrior(100, .1), MixturePrior(2, 7)):
            for family in INFORMED_FAMILIES:
                for alpha in (.1, .5, .9):
                    for n in range(30):
                        values = log_e_value(family, n, np.arange(n+1), alpha, prior)
                        few = [k for k in range(n+1) if k <= n*alpha and values[k] >= -np.log(.1)]
                        many = [k for k in range(n+1) if k >= n*alpha and values[k] >= -np.log(.1)]
                        self.assertEqual(critical_exceedance_counts(n, alpha=alpha, delta=.1,
                                         family=family, prior=prior),
                                         (max(few, default=-1), min(many, default=n+1)))
        first = precompute_critical_counts(50, .1, .05, "informed", MixturePrior(1, 100))
        second = precompute_critical_counts(50, .1, .05, "informed", MixturePrior(100, 1))
        self.assertFalse(np.array_equal(first[0], second[0]))
        self.assertFalse(first[0].flags.writeable)

    def test_exact_null_crossing_probability_and_unit_expectation(self):
        alpha, delta, horizon = .2, .2, 10
        prior = MixturePrior(1, 8, .4, .7)
        for family in INFORMED_FAMILIES:
            values = np.exp(log_e_value(family, horizon, np.arange(horizon+1), alpha, prior))
            self.assertAlmostEqual(np.dot(binom.pmf(np.arange(horizon+1), horizon, alpha), values), 1)
            few, many = precompute_critical_counts(horizon, alpha, delta, family, prior)
            probability = 0
            for outcomes in itertools.product((0, 1), repeat=horizon):
                counts = np.cumsum(outcomes)
                if np.any((counts <= few[1:]) | (counts >= many[1:])):
                    probability += alpha**sum(outcomes) * (1-alpha)**(horizon-sum(outcomes))
            self.assertLessEqual(probability, delta)

    def test_episode_prior_is_frozen_and_scores_start_empty(self):
        prior = initialize_mixture_prior([.5]*100, radius=2, kappa=.6)
        for family in INFORMED_FAMILIES:
            state = QuantileConfidenceSequence(alpha=.1, delta=.05, family=family, max_samples=100, prior=prior)
            self.assertEqual(state.n, 0)
            self.assertEqual(state.sorted_scores, [])
            calls = []
            result = run_anytime_episode(lambda: calls.append(1) or .5, radius=2,
                lower_radius=.2, alpha=.1, delta=.05, kappa=.6, radius_tolerance=.15,
                max_samples=100, family=family, prior=prior, check_every=10)
            self.assertIs(result.prior, prior)
            self.assertEqual(result.num_samples, len(calls))
            self.assertGreater(result.num_samples, 0)
            self.assertEqual(result.num_samples % 10, 0)
            self.assertEqual(result.branch, "conservative")
            for score in result.scores:
                state.update(score)
            self.assertEqual(result.quantile_lower, state.lower)
            self.assertEqual(result.quantile_upper, state.upper)


if __name__ == "__main__":
    unittest.main()
