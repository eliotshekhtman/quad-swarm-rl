from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import betaln, logsumexp


BRANCH_UNDER = "under"
BRANCH_CERTIFICATE = "certificate"
BRANCH_CONSERVATIVE = "conservative"
BRANCH_UNRESOLVED = "unresolved"


def episode_error_budget(delta: float, episode: int) -> float:
    """Return the paper's summable per-episode budget delta / 2^(j + 1)."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")
    if episode < 0:
        raise ValueError("episode must be non-negative.")
    return float(np.ldexp(delta, -(episode + 1)))


def probability_slopes(
    kappa: float,
    density_lower: float,
    density_upper: float,
) -> tuple[float, float]:
    """Convert fixed score sensitivity and density bounds into (m_p, L_p)."""
    if not np.isfinite(kappa) or not 0.0 <= kappa < 1.0:
        raise ValueError("kappa must lie in [0, 1).")
    if not np.isfinite(density_lower) or density_lower <= 0.0:
        raise ValueError("density_lower must be positive.")
    if not np.isfinite(density_upper) or density_upper < density_lower:
        raise ValueError("density_upper must be at least density_lower.")

    lower_slope = (1.0 - kappa) * density_lower
    upper_slope = (1.0 + kappa) * density_upper
    return float(lower_slope), float(upper_slope)


def select_branch(
    *,
    lower: float,
    upper: float,
    alpha: float,
    safe: bool,
    lower_slope: float,
    radius_tolerance: float,
    relative_precision: float,
) -> str:
    """Apply Algorithm 1's ordered branch rule to confidence endpoints."""
    if lower > alpha:
        return BRANCH_UNDER
    if safe and alpha - lower_slope * radius_tolerance <= lower <= alpha:
        return BRANCH_CERTIFICATE
    width = upper - lower
    if upper < alpha and width <= relative_precision * (alpha - upper):
        return BRANCH_CONSERVATIVE
    return BRANCH_UNRESOLVED


def update_radius(
    *,
    branch: str,
    radius: float,
    upper: float,
    alpha: float,
    lower_slope: float,
    upper_slope: float,
    eta: float,
    min_radius: float,
    max_radius: float,
) -> tuple[float, bool | None]:
    """Apply (84); return the projected radius and any safety-flag override."""
    if branch == BRANCH_CONSERVATIVE:
        candidate = radius - eta * (alpha - upper) / upper_slope
        safe_override = True
    elif branch == BRANCH_UNDER:
        candidate = radius + (upper - alpha) / lower_slope
        safe_override = True
    elif branch in (BRANCH_CERTIFICATE, BRANCH_UNRESOLVED):
        candidate = radius
        safe_override = None
    else:
        raise ValueError(f"Unknown anytime branch: {branch}")

    return float(np.clip(candidate, min_radius, max_radius)), safe_override


def _log_likelihood_factor(
    observation: int,
    alternative,
    null: float,
):
    alternative = np.asarray(alternative, dtype=np.float64)
    if observation == 1:
        return np.log(alternative) - np.log(null)
    return np.log1p(-alternative) - np.log1p(-null)


class BernoulliAnytimeState:
    """Sequential confidence grid and directional e-process diagnostics."""

    def __init__(
        self,
        *,
        alpha: float,
        delta: float,
        nominal_forecast: float,
        grid_size: int = 601,
        mixture_size: int = 41,
        alternative_clip: float = 1.0e-4,
    ) -> None:
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")
        if not np.isfinite(delta) or not 0.0 < delta < 1.0:
            raise ValueError("delta must lie in (0, 1).")
        if not np.isfinite(nominal_forecast) or not 0.0 < nominal_forecast < 1.0:
            raise ValueError("nominal_forecast must lie in (0, 1).")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if mixture_size <= 0:
            raise ValueError("mixture_size must be positive.")
        if not 0.0 < alternative_clip < 0.5:
            raise ValueError("alternative_clip must lie in (0, 0.5).")

        self.alpha = float(alpha)
        self.delta = float(delta)
        self.grid_size = int(grid_size)
        self.grid_spacing = 1.0 / (self.grid_size + 1)
        self.grid = (
            np.arange(1, self.grid_size + 1, dtype=np.float64)
            * self.grid_spacing
        )
        self.active = np.ones(self.grid_size, dtype=bool)
        self.log_boundary = float(np.log(1.0 / self.delta))

        clipped_forecast = float(
            np.clip(nominal_forecast, alternative_clip, 1.0 - alternative_clip)
        )
        self.nominal_conservative = min(self.alpha, clipped_forecast)
        self.nominal_under = max(self.alpha, clipped_forecast)
        self.alternative_clip = float(alternative_clip)

        indices = np.arange(1, mixture_size + 1, dtype=np.float64)
        mixture_locations = 0.5 * (
            1.0 - np.cos(np.pi * indices / (mixture_size + 1))
        )
        self.mixture_conservative = self.alpha * mixture_locations
        self.mixture_under = self.alpha + (1.0 - self.alpha) * mixture_locations
        self.log_mixture_weight = -float(np.log(mixture_size))

        self.n = 0
        self.exceedances = 0
        self.log_nominal_conservative = 0.0
        self.log_nominal_under = 0.0
        self.log_plugin_conservative = 0.0
        self.log_plugin_under = 0.0
        self.log_mixture_conservative = np.zeros(mixture_size, dtype=np.float64)
        self.log_mixture_under = np.zeros(mixture_size, dtype=np.float64)

    @property
    def lower(self) -> float:
        if not np.any(self.active):
            return 0.0
        first = int(np.flatnonzero(self.active)[0])
        return max(0.0, float(self.grid[first] - self.grid_spacing))

    @property
    def upper(self) -> float:
        if not np.any(self.active):
            return 1.0
        last = int(np.flatnonzero(self.active)[-1])
        return min(1.0, float(self.grid[last] + self.grid_spacing))

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def empirical_exceedance(self) -> float:
        if self.n == 0:
            return np.nan
        return self.exceedances / self.n

    @property
    def stopped_forecast(self) -> float:
        return (self.exceedances + 0.5) / (self.n + 1.0)

    @property
    def log_hedge_conservative(self) -> float:
        log_mixture = logsumexp(self.log_mixture_conservative) + self.log_mixture_weight
        return float(
            logsumexp(
                [
                    self.log_nominal_conservative,
                    self.log_plugin_conservative,
                    log_mixture,
                ]
            )
            - np.log(3.0)
        )

    @property
    def log_hedge_under(self) -> float:
        log_mixture = logsumexp(self.log_mixture_under) + self.log_mixture_weight
        return float(
            logsumexp(
                [
                    self.log_nominal_under,
                    self.log_plugin_under,
                    log_mixture,
                ]
            )
            - np.log(3.0)
        )

    def update(self, observation: int | bool) -> tuple[float, float]:
        observation = int(observation)
        if observation not in (0, 1):
            raise ValueError("observation must be zero or one.")

        next_n = self.n + 1
        half_count = (self.exceedances + 0.5) / next_n
        clipped_half_count = float(
            np.clip(
                half_count,
                self.alternative_clip,
                1.0 - self.alternative_clip,
            )
        )
        plugin_conservative = min(self.alpha, clipped_half_count)
        plugin_under = max(self.alpha, clipped_half_count)

        self.log_nominal_conservative += float(
            _log_likelihood_factor(
                observation,
                self.nominal_conservative,
                self.alpha,
            )
        )
        self.log_nominal_under += float(
            _log_likelihood_factor(
                observation,
                self.nominal_under,
                self.alpha,
            )
        )
        self.log_plugin_conservative += float(
            _log_likelihood_factor(
                observation,
                plugin_conservative,
                self.alpha,
            )
        )
        self.log_plugin_under += float(
            _log_likelihood_factor(
                observation,
                plugin_under,
                self.alpha,
            )
        )
        self.log_mixture_conservative += _log_likelihood_factor(
            observation,
            self.mixture_conservative,
            self.alpha,
        )
        self.log_mixture_under += _log_likelihood_factor(
            observation,
            self.mixture_under,
            self.alpha,
        )

        self.n = next_n
        self.exceedances += observation
        log_beta_mixture = (
            betaln(
                self.exceedances + 0.5,
                self.n - self.exceedances + 0.5,
            )
            - betaln(0.5, 0.5)
            - self.exceedances * np.log(self.grid)
            - (self.n - self.exceedances) * np.log1p(-self.grid)
        )
        self.active &= log_beta_mixture < self.log_boundary
        return self.lower, self.upper


@dataclass(frozen=True)
class AnytimeEpisodeResult:
    branch: str
    radius: float
    next_radius: float
    safe: bool
    next_safe: bool
    num_samples: int
    exceedances: int
    lower: float
    upper: float
    width: float
    eta: float
    radius_certificate: float
    stopped_forecast: float
    log_hedge_conservative: float
    log_hedge_under: float
    scores: np.ndarray
    indicators: np.ndarray
    lower_history: np.ndarray
    upper_history: np.ndarray


def run_anytime_episode(
    sample_score: Callable[[], float],
    *,
    radius: float,
    safe: bool,
    alpha: float,
    delta: float,
    lower_slope: float,
    upper_slope: float,
    relative_precision: float,
    radius_tolerance: float,
    eta_min: float,
    damping_gamma: float,
    min_radius: float,
    max_radius: float,
    max_samples: int,
    nominal_forecast: float,
    grid_size: int = 601,
    check_every: int = 1,
) -> AnytimeEpisodeResult:
    if (
        not np.isfinite(lower_slope)
        or not np.isfinite(upper_slope)
        or lower_slope <= 0.0
        or upper_slope < lower_slope
    ):
        raise ValueError("slopes must satisfy 0 < lower_slope <= upper_slope.")
    if not np.isfinite(relative_precision) or relative_precision <= 0.0:
        raise ValueError("relative_precision must be positive.")
    if not np.isfinite(radius_tolerance) or radius_tolerance < 0.0:
        raise ValueError("radius_tolerance must be non-negative.")
    if not np.isfinite(eta_min) or not 0.0 < eta_min <= 1.0:
        raise ValueError("eta_min must lie in (0, 1].")
    if not np.isfinite(damping_gamma) or damping_gamma <= 0.0:
        raise ValueError("damping_gamma must be positive.")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if check_every <= 0:
        raise ValueError("check_every must be positive.")
    if (
        not np.isfinite(min_radius)
        or not np.isfinite(radius)
        or not np.isfinite(max_radius)
        or min_radius > radius
        or radius > max_radius
    ):
        raise ValueError("radius must lie in [min_radius, max_radius].")

    state = BernoulliAnytimeState(
        alpha=alpha,
        delta=delta,
        nominal_forecast=nominal_forecast,
        grid_size=grid_size,
    )
    scores = []
    indicators = []
    lower_history = []
    upper_history = []
    branch = BRANCH_UNRESOLVED

    for sample_index in range(1, max_samples + 1):
        score = float(sample_score())
        if not np.isfinite(score):
            raise ValueError("sample_score returned a non-finite value.")
        indicator = int(score > radius)
        scores.append(score)
        indicators.append(indicator)
        lower, upper = state.update(indicator)
        lower_history.append(lower)
        upper_history.append(upper)

        should_check = sample_index % check_every == 0 or sample_index == max_samples
        if should_check:
            branch = select_branch(
                lower=state.lower,
                upper=state.upper,
                alpha=alpha,
                safe=safe,
                lower_slope=lower_slope,
                radius_tolerance=radius_tolerance,
                relative_precision=relative_precision,
            )
            if branch != BRANCH_UNRESOLVED:
                break

    information_rate = max(state.log_hedge_conservative, 0.0) / state.n
    eta = eta_min + (1.0 - eta_min) * information_rate / (
        information_rate + damping_gamma
    )
    next_radius, safe_override = update_radius(
        branch=branch,
        radius=radius,
        upper=state.upper,
        alpha=alpha,
        lower_slope=lower_slope,
        upper_slope=upper_slope,
        eta=eta,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    next_safe = safe if safe_override is None else safe_override
    radius_certificate = max(alpha - state.lower, 0.0) / lower_slope

    return AnytimeEpisodeResult(
        branch=branch,
        radius=float(radius),
        next_radius=next_radius,
        safe=bool(safe),
        next_safe=bool(next_safe),
        num_samples=state.n,
        exceedances=state.exceedances,
        lower=state.lower,
        upper=state.upper,
        width=state.width,
        eta=float(eta),
        radius_certificate=float(radius_certificate),
        stopped_forecast=float(state.stopped_forecast),
        log_hedge_conservative=state.log_hedge_conservative,
        log_hedge_under=state.log_hedge_under,
        scores=np.asarray(scores, dtype=np.float64),
        indicators=np.asarray(indicators, dtype=np.int8),
        lower_history=np.asarray(lower_history, dtype=np.float64),
        upper_history=np.asarray(upper_history, dtype=np.float64),
    )


__all__ = [
    "AnytimeEpisodeResult",
    "BRANCH_CERTIFICATE",
    "BRANCH_CONSERVATIVE",
    "BRANCH_UNDER",
    "BRANCH_UNRESOLVED",
    "BernoulliAnytimeState",
    "episode_error_budget",
    "probability_slopes",
    "run_anytime_episode",
    "select_branch",
    "update_radius",
]
