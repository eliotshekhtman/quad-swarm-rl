from __future__ import annotations

import bisect
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.special import betaln, betaincinv


BRANCH_CERTIFICATE = "certificate"
BRANCH_CONSERVATIVE = "conservative"
BRANCH_UNRESOLVED = "unresolved"

E_PROCESS_JEFFREYS = "jeffreys"
E_PROCESS_NORMAL = "normal"
E_PROCESS_FAMILIES = (E_PROCESS_JEFFREYS, E_PROCESS_NORMAL)


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")


def _validate_count(n: int, k) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative.")
    counts = np.asarray(k)
    if np.any(counts != np.floor(counts)) or np.any((counts < 0) | (counts > n)):
        raise ValueError("k must contain integers in [0, n].")
    return counts.astype(np.float64)


def jeffreys_log_e_value(n: int, k, alpha: float):
    """Log Jeffreys-mixture e-value for n Bernoulli trials and k successes."""
    _validate_alpha(alpha)
    counts = _validate_count(n, k)
    values = (
        betaln(counts + 0.5, n - counts + 0.5)
        - betaln(0.5, 0.5)
        - counts * np.log(alpha)
        - (n - counts) * np.log1p(-alpha)
    )
    return float(values) if values.ndim == 0 else values


def normal_log_e_value(n: int, k, alpha: float):
    """Log nonparametric normal-mixture e-value from equation (28)."""
    _validate_alpha(alpha)
    counts = _validate_count(n, k)
    values = -0.5 * np.log(n + 1.0) + 2.0 * (counts - n * alpha) ** 2 / (
        n + 1.0
    )
    return float(values) if values.ndim == 0 else values


def log_e_value(family: str, n: int, k, alpha: float):
    if family == E_PROCESS_JEFFREYS:
        return jeffreys_log_e_value(n, k, alpha)
    if family == E_PROCESS_NORMAL:
        return normal_log_e_value(n, k, alpha)
    raise ValueError(f"Unknown e-process family: {family}")


def critical_exceedance_counts(
    n: int,
    *,
    alpha: float,
    delta: float,
    family: str,
) -> tuple[int, int]:
    """Return the few- and many-exceedance critical counts in equation (30)."""
    _validate_alpha(alpha)
    if not np.isfinite(delta) or not 0.0 <= delta < 1.0:
        raise ValueError("delta must lie in [0, 1).")
    if family not in E_PROCESS_FAMILIES:
        raise ValueError(f"Unknown e-process family: {family}")
    if n < 0:
        raise ValueError("n must be non-negative.")
    if delta == 0.0:
        return -1, n + 1

    boundary = -np.log(delta)
    lower_end = int(np.floor(n * alpha))
    upper_start = int(np.ceil(n * alpha))

    def rejected(k: int) -> bool:
        return log_e_value(family, n, k, alpha) >= boundary

    # Each e-value is convex in k, so its rejection set consists of two tails.
    few = -1
    if rejected(0):
        low, high = 0, lower_end
        while low < high:
            middle = (low + high + 1) // 2
            if rejected(middle):
                low = middle
            else:
                high = middle - 1
        few = low

    many = n + 1
    if rejected(n):
        low, high = upper_start, n
        while low < high:
            middle = (low + high) // 2
            if rejected(middle):
                high = middle
            else:
                low = middle + 1
        many = low

    return few, many


@lru_cache(maxsize=32)
def precompute_critical_counts(
    max_samples: int,
    alpha: float,
    delta: float,
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative.")
    few = np.empty(max_samples + 1, dtype=np.int64)
    many = np.empty(max_samples + 1, dtype=np.int64)
    for n in range(max_samples + 1):
        few[n], many[n] = critical_exceedance_counts(
            n,
            alpha=alpha,
            delta=delta,
            family=family,
        )
    few.setflags(write=False)
    many.setflags(write=False)
    return few, many


class QuantileConfidenceSequence:
    """Order-statistic quantile confidence sequence from equations (30)-(31)."""

    def __init__(
        self,
        *,
        alpha: float,
        delta: float,
        family: str,
        max_samples: int,
    ) -> None:
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.family = family
        self.max_samples = int(max_samples)
        self.few_critical, self.many_critical = precompute_critical_counts(
            self.max_samples,
            self.alpha,
            self.delta,
            self.family,
        )
        self.sorted_scores: list[float] = []
        self.n = 0
        self.lower = 0.0
        self.upper = np.inf

    def _order_statistic(self, index: int) -> float:
        if index <= 0:
            return 0.0
        if index > self.n:
            return np.inf
        return self.sorted_scores[index - 1]

    def update(self, score: float) -> tuple[float, float, int, int]:
        if not np.isfinite(score) or score < 0.0:
            raise ValueError("score must be finite and non-negative.")
        if self.n >= self.max_samples:
            raise ValueError("quantile confidence sequence exceeded max_samples.")

        bisect.insort(self.sorted_scores, float(score))
        self.n += 1
        few = int(self.few_critical[self.n])
        many = int(self.many_critical[self.n])
        lower_candidate = self._order_statistic(self.n - many + 1)
        upper_candidate = self._order_statistic(self.n - few)
        self.lower = max(self.lower, lower_candidate)
        self.upper = min(self.upper, upper_candidate)
        return self.lower, self.upper, few, many


def contraction_factors(kappa: float) -> tuple[float, float]:
    if not np.isfinite(kappa) or not 0.0 <= kappa < 1.0:
        raise ValueError("kappa must lie in [0, 1).")
    gamma = 2.0 * kappa / (1.0 + kappa)
    zeta = (1.0 + gamma) / 2.0
    return float(gamma), float(zeta)


def max_active_episodes(
    initial_width: float,
    radius_tolerance: float,
    kappa: float,
) -> int | None:
    """Return k_epsilon from equation (37), or None for zero tolerance."""
    if not np.isfinite(initial_width) or initial_width < 0.0:
        raise ValueError("initial_width must be finite and non-negative.")
    if not np.isfinite(radius_tolerance) or radius_tolerance < 0.0:
        raise ValueError("radius_tolerance must be finite and non-negative.")
    if radius_tolerance == 0.0:
        return None
    if initial_width <= radius_tolerance:
        return 0

    _, zeta = contraction_factors(kappa)
    episodes = 0
    width = float(initial_width)
    while width > radius_tolerance:
        width *= zeta
        episodes += 1
    return episodes


def episode_error_budget(
    delta: float,
    episode: int,
    k_epsilon: int | None,
) -> float:
    """Return the per-episode budget in equation (37)."""
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")
    if episode < 0:
        raise ValueError("episode must be non-negative.")
    if k_epsilon is None:
        return float(np.ldexp(delta, -(episode + 1)))
    if k_epsilon < 0:
        raise ValueError("k_epsilon must be non-negative.")
    if k_epsilon == 0 or episode >= k_epsilon:
        return 0.0
    return float(delta / k_epsilon)


def clopper_pearson_interval(
    exceedances: int,
    num_samples: int,
    delta: float,
) -> tuple[float, float]:
    """Two-sided exact binomial interval with total error probability delta."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if exceedances < 0 or exceedances > num_samples:
        raise ValueError("exceedances must lie in [0, num_samples].")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")

    tail = delta / 2.0
    lower = (
        0.0
        if exceedances == 0
        else float(betaincinv(exceedances, num_samples - exceedances + 1, tail))
    )
    upper = (
        1.0
        if exceedances == num_samples
        else float(
            betaincinv(
                exceedances + 1,
                num_samples - exceedances,
                1.0 - tail,
            )
        )
    )
    return lower, upper


@dataclass(frozen=True)
class BracketCandidate:
    lower_radius: float
    radius: float
    width: float


def candidate_bracket(
    *,
    radius: float,
    lower_radius: float,
    quantile_lower: float,
    quantile_upper: float,
    kappa: float,
) -> BracketCandidate:
    contraction_factors(kappa)
    if not np.isfinite(radius) or not np.isfinite(lower_radius):
        raise ValueError("radius bounds must be finite.")
    if lower_radius > radius:
        raise ValueError("lower_radius must not exceed radius.")
    if quantile_lower < 0.0 or quantile_upper < 0.0:
        raise ValueError("quantile bounds must be non-negative.")

    next_lower = max(
        lower_radius,
        (quantile_lower - kappa * radius) / (1.0 - kappa),
    )
    next_radius = max(
        next_lower,
        min(radius, (quantile_upper + kappa * radius) / (1.0 + kappa)),
    )
    return BracketCandidate(
        lower_radius=float(next_lower),
        radius=float(next_radius),
        width=float(next_radius - next_lower),
    )


def select_branch(
    *,
    candidate: BracketCandidate,
    current_width: float,
    quantile_upper: float,
    radius: float,
    radius_tolerance: float,
    zeta: float,
) -> str:
    if candidate.width <= radius_tolerance:
        return BRANCH_CERTIFICATE
    if quantile_upper < radius and candidate.width <= zeta * current_width:
        return BRANCH_CONSERVATIVE
    return BRANCH_UNRESOLVED


@dataclass(frozen=True)
class AnytimeEpisodeResult:
    branch: str
    termination_reason: str
    family: str
    radius: float
    lower_radius: float
    next_radius: float
    next_lower_radius: float
    current_width: float
    next_width: float
    num_samples: int
    exceedances: int
    quantile_lower: float
    quantile_upper: float
    radius_certificate: float
    scores: np.ndarray
    indicators: np.ndarray
    quantile_lower_history: np.ndarray
    quantile_upper_history: np.ndarray
    few_critical_history: np.ndarray
    many_critical_history: np.ndarray
    candidate_radius_history: np.ndarray
    candidate_lower_radius_history: np.ndarray
    candidate_width_history: np.ndarray


def run_anytime_episode(
    sample_score: Callable[[], float],
    *,
    radius: float,
    lower_radius: float,
    alpha: float,
    delta: float,
    kappa: float,
    radius_tolerance: float,
    max_samples: int,
    family: str = E_PROCESS_JEFFREYS,
    check_every: int = 1,
) -> AnytimeEpisodeResult:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if check_every <= 0:
        raise ValueError("check_every must be positive.")
    if not np.isfinite(radius_tolerance) or radius_tolerance < 0.0:
        raise ValueError("radius_tolerance must be finite and non-negative.")

    _, zeta = contraction_factors(kappa)
    state = QuantileConfidenceSequence(
        alpha=alpha,
        delta=delta,
        family=family,
        max_samples=max_samples,
    )
    current_width = radius - lower_radius
    if current_width < 0.0:
        raise ValueError("lower_radius must not exceed radius.")

    scores: list[float] = []
    indicators: list[int] = []
    lower_history: list[float] = []
    upper_history: list[float] = []
    few_history: list[int] = []
    many_history: list[int] = []
    radius_history: list[float] = []
    lower_radius_history: list[float] = []
    width_history: list[float] = []

    candidate = candidate_bracket(
        radius=radius,
        lower_radius=lower_radius,
        quantile_lower=state.lower,
        quantile_upper=state.upper,
        kappa=kappa,
    )
    branch = select_branch(
        candidate=candidate,
        current_width=current_width,
        quantile_upper=state.upper,
        radius=radius,
        radius_tolerance=radius_tolerance,
        zeta=zeta,
    )
    termination_reason = BRANCH_CERTIFICATE if branch == BRANCH_CERTIFICATE else ""

    if branch != BRANCH_CERTIFICATE:
        for sample_index in range(1, max_samples + 1):
            score = float(sample_score())
            lower, upper, few, many = state.update(score)
            scores.append(score)
            indicators.append(int(score > radius))
            lower_history.append(lower)
            upper_history.append(upper)
            few_history.append(few)
            many_history.append(many)

            candidate = candidate_bracket(
                radius=radius,
                lower_radius=lower_radius,
                quantile_lower=lower,
                quantile_upper=upper,
                kappa=kappa,
            )
            radius_history.append(candidate.radius)
            lower_radius_history.append(candidate.lower_radius)
            width_history.append(candidate.width)

            should_check = (
                sample_index % check_every == 0 or sample_index == max_samples
            )
            if not should_check:
                continue

            if lower > min(upper, radius):
                branch = BRANCH_UNRESOLVED
                termination_reason = "inconsistent-bounds"
                break

            branch = select_branch(
                candidate=candidate,
                current_width=current_width,
                quantile_upper=upper,
                radius=radius,
                radius_tolerance=radius_tolerance,
                zeta=zeta,
            )
            if branch != BRANCH_UNRESOLVED:
                termination_reason = branch
                break

        if not termination_reason:
            termination_reason = "sample-cap"

    resolved = branch in (BRANCH_CERTIFICATE, BRANCH_CONSERVATIVE)
    next_radius = candidate.radius if resolved else radius
    next_lower_radius = candidate.lower_radius if resolved else lower_radius
    next_width = candidate.width if resolved else current_width
    radius_certificate = (
        radius - candidate.lower_radius
        if termination_reason != "inconsistent-bounds"
        else np.nan
    )

    return AnytimeEpisodeResult(
        branch=branch,
        termination_reason=termination_reason,
        family=family,
        radius=float(radius),
        lower_radius=float(lower_radius),
        next_radius=float(next_radius),
        next_lower_radius=float(next_lower_radius),
        current_width=float(current_width),
        next_width=float(next_width),
        num_samples=state.n,
        exceedances=int(np.sum(indicators)),
        quantile_lower=float(state.lower),
        quantile_upper=float(state.upper),
        radius_certificate=float(radius_certificate),
        scores=np.asarray(scores, dtype=np.float64),
        indicators=np.asarray(indicators, dtype=np.int8),
        quantile_lower_history=np.asarray(lower_history, dtype=np.float64),
        quantile_upper_history=np.asarray(upper_history, dtype=np.float64),
        few_critical_history=np.asarray(few_history, dtype=np.int64),
        many_critical_history=np.asarray(many_history, dtype=np.int64),
        candidate_radius_history=np.asarray(radius_history, dtype=np.float64),
        candidate_lower_radius_history=np.asarray(
            lower_radius_history, dtype=np.float64
        ),
        candidate_width_history=np.asarray(width_history, dtype=np.float64),
    )


__all__ = [
    "AnytimeEpisodeResult",
    "BRANCH_CERTIFICATE",
    "BRANCH_CONSERVATIVE",
    "BRANCH_UNRESOLVED",
    "BracketCandidate",
    "E_PROCESS_FAMILIES",
    "E_PROCESS_JEFFREYS",
    "E_PROCESS_NORMAL",
    "QuantileConfidenceSequence",
    "candidate_bracket",
    "clopper_pearson_interval",
    "contraction_factors",
    "critical_exceedance_counts",
    "episode_error_budget",
    "jeffreys_log_e_value",
    "log_e_value",
    "max_active_episodes",
    "normal_log_e_value",
    "precompute_critical_counts",
    "run_anytime_episode",
    "select_branch",
]
