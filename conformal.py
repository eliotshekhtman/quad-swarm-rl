#!/usr/bin/env python3
"""Run fixed-sample or anytime-valid conformal CBF experiments."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from sample_factory.model.model_utils import get_rnn_size

from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.anytime_utils import (
    BRANCH_CERTIFICATE,
    BRANCH_CONSERVATIVE,
    E_PROCESS_JEFFREYS,
    E_PROCESS_NORMAL,
    E_PROCESS_FAMILIES,
    INFORMED_FAMILIES,
    clopper_pearson_interval,
    contraction_factors,
    episode_error_budget,
    max_active_episodes,
    initialize_mixture_prior,
    run_anytime_episode,
)
from project_utils.conformal_utils import (
    calibrate_once_final_radius,
    get_alpha_bar,
    joint_conformal_radius,
    joint_rollout_error,
    next_radius,
    paper_calibrate_once_radius,
    run_multi_agents,
    summarize_rollouts,
)
from project_utils import full_cbf_utils
from project_utils.full_cbf_utils import make_cbf_filter
from project_utils.prediction_utils import (
    EVALUATION_SCENARIO,
    collect_homogeneous_trajectories,
    load_prediction_artifact,
)
from project_utils.restart_utils import deterministic_reset, set_global_seed
from project_utils.utils import latest_checkpoint, load_actor, load_cfg


DEFAULT_INITIAL_RADIUS = 2.0
DEFAULT_KAPPA = 0.6
COLLISION_HITBOX_RADIUS = 2.5
UPDATE_PROCEDURES = (
    "robust",
    "naive",
    "non-interactive",
    "calibrate-once",
    "calibrate-once-paper",
    "anytime",
    "anytime-nonparametric",
)
ANYTIME_PROCEDURES = ("anytime", "anytime-nonparametric")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a conformal prediction experiment with the full ECBF."
    )
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", required=True)
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", default="train_dir")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Optional calibration/assessment seed, applied once after scenario and prediction setup. Omit to preserve the historical random stream.",
    )
    parser.add_argument("--r_init", type=float, default=DEFAULT_INITIAL_RADIUS)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument(
        "--kappa",
        type=float,
        default=DEFAULT_KAPPA,
        help="Small-gain coefficient used by radius updates (default: 0.6).",
    )
    parser.add_argument(
        "--cbf-k1",
        type=float,
        default=full_cbf_utils.CBF_K1,
        help="ECBF gain, independent of the radius-update kappa.",
    )
    parser.add_argument(
        "--cbf-solver",
        choices=("cascade", "legacy"),
        default="cascade",
        help="Solver behavior: validated cascade, or historical ECOS-only replay.",
    )
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=200,
        help="Fixed calibration size, or the per-episode cap in anytime mode.",
    )
    parser.add_argument("--num_eval_trajs", type=int, default=100)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--predictions_dir", default="train_dir")
    parser.add_argument("--init_predictions", default=None)
    parser.add_argument(
        "--update_predictions",
        action="store_true",
        help="Replace predictions after episode 0 (robust and naive only).",
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Device for policy-network inference. Simulation and ECBF solves remain on CPU.",
    )
    parser.add_argument(
        "--update_procedure",
        "--update-procedure",
        choices=UPDATE_PROCEDURES,
        default="robust",
        help="Radius update rule. calibrate-once-paper solves the fixed-reference One-time CP equation; calibrate-once preserves the legacy recurrence.",
    )
    parser.add_argument("--anytime-radius-tolerance", type=float, default=0.15)
    parser.add_argument(
        "--anytime-family", choices=E_PROCESS_FAMILIES,
        help="E-process mixture (default: jeffreys; anytime-nonparametric selects normal).",
    )
    parser.add_argument(
        "--anytime-jeffreys-weight", type=float, default=None,
        help="Jeffreys prior weight eta for hybrid/directional mixtures (default: 0.5).",
    )
    parser.add_argument(
        "--anytime-directional-weight", type=float, default=None,
        help="Lower-side mixture weight w for directional (default: 0.5).",
    )
    parser.add_argument(
        "--anytime-check-every",
        type=int,
        default=1,
        help="Check stopping conditions every N complete joint rollouts.",
    )
    parser.add_argument(
        "--anytime-init-trajectories",
        type=int,
        default=0,
        help="Independent trajectories used to certify the initial radius.",
    )
    parser.add_argument(
        "--anytime-delta-init",
        type=float,
        default=None,
        help="Initial-radius calibration error budget (default: 0.05).",
    )
    parser.add_argument(
        "--anytime-assume-initial-safe",
        action="store_true",
        help="Treat r_init as known safe instead of calibrating it from data.",
    )
    args = parser.parse_args(argv)

    if args.run_seed is not None and not 0 <= args.run_seed < 2**32:
        parser.error("--run-seed must lie in [0, 4294967295]")
    if args.episode_length < 2:
        parser.error("--episode_length must be at least 2")
    if args.num_trajectories <= 0:
        parser.error("--num_trajectories must be positive")
    if args.num_eval_trajs <= 0:
        parser.error("--num_eval_trajs must be positive")
    if args.num_episodes < 0:
        parser.error("--num_episodes must be non-negative")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie in (0, 1)")
    if not 0.0 < args.delta < 1.0:
        parser.error("--delta must lie in (0, 1)")
    if not np.isfinite(args.kappa) or not 0.0 <= args.kappa < 1.0:
        parser.error("--kappa must lie in [0, 1)")
    if not np.isfinite(args.cbf_k1) or args.cbf_k1 <= 0:
        parser.error("--cbf-k1 must be finite and positive")
    non_iterative_procedures = {"non-interactive", "calibrate-once", "calibrate-once-paper"}
    if (
        args.update_procedure not in non_iterative_procedures
        and args.num_episodes == 0
    ):
        parser.error("--num_episodes must be positive for iterative updates")
    if (
        args.update_procedure in non_iterative_procedures
        and args.update_predictions
    ):
        parser.error(
            "--update_predictions is not supported by non-iterative updates"
        )
    if (
        not np.isfinite(args.anytime_radius_tolerance)
        or args.anytime_radius_tolerance < 0.0
    ):
        parser.error("--anytime-radius-tolerance must be non-negative")
    if args.anytime_check_every <= 0:
        parser.error("--anytime-check-every must be positive")
    if args.anytime_init_trajectories < 0:
        parser.error("--anytime-init-trajectories must be non-negative")
    if args.update_procedure not in ANYTIME_PROCEDURES and any(value is not None for value in (
        args.anytime_family, args.anytime_jeffreys_weight, args.anytime_directional_weight,
    )):
        parser.error("anytime mixture options require an anytime update procedure")
    if args.update_procedure in ANYTIME_PROCEDURES:
        if args.update_procedure == "anytime-nonparametric":
            if args.anytime_family not in (None, E_PROCESS_NORMAL):
                parser.error("anytime-nonparametric requires the normal family")
            args.anytime_family = E_PROCESS_NORMAL
        elif args.anytime_family is None:
            args.anytime_family = E_PROCESS_JEFFREYS
        if args.anytime_jeffreys_weight is not None and args.anytime_family not in (
            "jeffreys-informed", "directional",
        ):
            parser.error("--anytime-jeffreys-weight requires a hybrid or directional mixture")
        if args.anytime_directional_weight is not None and args.anytime_family != "directional":
            parser.error("--anytime-directional-weight requires the directional mixture")
        if args.anytime_jeffreys_weight is None:
            args.anytime_jeffreys_weight = 0.5
        if args.anytime_directional_weight is None:
            args.anytime_directional_weight = 0.5
        if not 0 < args.anytime_jeffreys_weight <= 1:
            parser.error("--anytime-jeffreys-weight must lie in (0, 1]")
        if not 0 < args.anytime_directional_weight < 1:
            parser.error("--anytime-directional-weight must lie in (0, 1)")
        if args.anytime_family in INFORMED_FAMILIES and args.anytime_init_trajectories == 0:
            parser.error("informed mixtures require --anytime-init-trajectories for the initial prior")
        has_calibration = args.anytime_init_trajectories > 0
        if has_calibration == args.anytime_assume_initial_safe:
            parser.error(
                "anytime updates require exactly one of "
                "--anytime-init-trajectories N or --anytime-assume-initial-safe"
            )
        if args.update_predictions:
            parser.error(
                "--update_predictions is not supported by anytime updates"
            )
        if has_calibration:
            if args.anytime_delta_init is None:
                args.anytime_delta_init = 0.05
            if (
                not np.isfinite(args.anytime_delta_init)
                or not 0.0 < args.anytime_delta_init < 1.0
            ):
                parser.error("--anytime-delta-init must lie in (0, 1)")
            if args.delta + args.anytime_delta_init >= 1.0:
                parser.error("--delta + --anytime-delta-init must be below 1")
        else:
            if args.anytime_delta_init is not None:
                parser.error(
                    "--anytime-delta-init requires initial calibration"
                )
            args.anytime_delta_init = 0.0

    return args


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def _empty_metrics() -> dict:
    return {
        "qj_per_episode": [],
        "radius_per_episode": [],
        "tube_coverage_per_episode": [],
        "crashes_per_episode": [],
        "bad_crashes_per_episode": [],
        "safety_per_episode": [],
        "cumulative_reward_per_episode": [],
        "cumulative_reward_runs_per_episode": [],
        "agent_locs_per_episode": [],
        "predicted_traj_per_episode": [],
        "evaluation_scores_per_episode": [],
        "initial_reference_scores": [],
        "anytime_current_radius_per_episode": [],
        "anytime_reference_scores": [],
        "anytime_current_lower_radius_per_episode": [],
        "anytime_next_radius_per_episode": [],
        "anytime_next_lower_radius_per_episode": [],
        "anytime_current_width_per_episode": [],
        "anytime_next_width_per_episode": [],
        "anytime_tube_radius_per_episode": [],
        "anytime_quantile_lower_per_episode": [],
        "anytime_quantile_upper_per_episode": [],
        "anytime_trajectory_count_per_episode": [],
        "anytime_exceedance_count_per_episode": [],
        "anytime_empirical_exceedance_per_episode": [],
        "anytime_branch_per_episode": [],
        "anytime_termination_reason_per_episode": [],
        "anytime_delta_per_episode": [],
        "anytime_radius_certificate_per_episode": [],
        "anytime_scores_per_episode": [],
        "anytime_indicators_per_episode": [],
        "anytime_sample_radii_per_episode": [],
        "anytime_quantile_lower_history_per_episode": [],
        "anytime_quantile_upper_history_per_episode": [],
        "anytime_few_critical_history_per_episode": [],
        "anytime_many_critical_history_per_episode": [],
        "anytime_candidate_radius_history_per_episode": [],
        "anytime_candidate_lower_radius_history_per_episode": [],
        "anytime_candidate_width_history_per_episode": [],
        "anytime_config": None,
        "anytime_priors": [],
        "anytime_run_termination_reason": None,
    }


def _record_episode(
    metrics: dict,
    qj: float,
    radius: float,
    summary: dict,
    pred_trajectories,
    evaluation_scores=None,
) -> None:
    metrics["qj_per_episode"].append(qj)
    metrics["radius_per_episode"].append(radius)
    metrics["tube_coverage_per_episode"].append(summary["tube_coverage"])
    metrics["crashes_per_episode"].append(summary["crashes"])
    metrics["bad_crashes_per_episode"].append(summary["bad_crashes"])
    metrics["safety_per_episode"].append(summary["safety"])
    metrics["cumulative_reward_per_episode"].append(
        summary["cumulative_reward"]
    )
    metrics["cumulative_reward_runs_per_episode"].append(
        summary["cumulative_reward_per_run"]
    )
    metrics["agent_locs_per_episode"].append(summary["agent_locs"])
    metrics["predicted_traj_per_episode"].append(
        np.asarray(pred_trajectories, dtype=np.float32).copy()
    )
    if evaluation_scores is not None:
        metrics["evaluation_scores_per_episode"].append(
            np.asarray(evaluation_scores, dtype=np.float64).copy()
        )


def _record_anytime_episode(
    metrics: dict,
    result,
    episode_delta: float,
    min_radius: float,
) -> None:
    metrics["anytime_priors"].append(result.prior)
    metrics["anytime_current_radius_per_episode"].append(result.radius)
    metrics["anytime_current_lower_radius_per_episode"].append(
        result.lower_radius
    )
    metrics["anytime_next_radius_per_episode"].append(result.next_radius)
    metrics["anytime_next_lower_radius_per_episode"].append(
        result.next_lower_radius
    )
    metrics["anytime_current_width_per_episode"].append(result.current_width)
    metrics["anytime_next_width_per_episode"].append(result.next_width)
    metrics["anytime_tube_radius_per_episode"].append(
        max(result.next_radius - min_radius, 0.0)
    )
    metrics["anytime_quantile_lower_per_episode"].append(
        result.quantile_lower
    )
    metrics["anytime_quantile_upper_per_episode"].append(
        result.quantile_upper
    )
    metrics["anytime_trajectory_count_per_episode"].append(result.num_samples)
    metrics["anytime_exceedance_count_per_episode"].append(result.exceedances)
    metrics["anytime_empirical_exceedance_per_episode"].append(
        result.exceedances / result.num_samples
        if result.num_samples
        else np.nan
    )
    metrics["anytime_branch_per_episode"].append(result.branch)
    metrics["anytime_termination_reason_per_episode"].append(
        result.termination_reason
    )
    metrics["anytime_delta_per_episode"].append(episode_delta)
    metrics["anytime_radius_certificate_per_episode"].append(
        result.radius_certificate
    )
    metrics["anytime_scores_per_episode"].append(result.scores.copy())
    metrics["anytime_indicators_per_episode"].append(result.indicators.copy())
    metrics["anytime_sample_radii_per_episode"].append(
        np.full(result.num_samples, result.radius, dtype=np.float64)
    )
    metrics["anytime_quantile_lower_history_per_episode"].append(
        result.quantile_lower_history.copy()
    )
    metrics["anytime_quantile_upper_history_per_episode"].append(
        result.quantile_upper_history.copy()
    )
    metrics["anytime_few_critical_history_per_episode"].append(
        result.few_critical_history.copy()
    )
    metrics["anytime_many_critical_history_per_episode"].append(
        result.many_critical_history.copy()
    )
    metrics["anytime_candidate_radius_history_per_episode"].append(
        result.candidate_radius_history.copy()
    )
    metrics["anytime_candidate_lower_radius_history_per_episode"].append(
        result.candidate_lower_radius_history.copy()
    )
    metrics["anytime_candidate_width_history_per_episode"].append(
        result.candidate_width_history.copy()
    )


def _pad_rows(rows, fill_value, dtype):
    width = max((len(row) for row in rows), default=0)
    padded = np.full((len(rows), width), fill_value, dtype=dtype)
    for row_index, row in enumerate(rows):
        padded[row_index, : len(row)] = row
    return padded


def _save_metrics(
    experiment_dir: str,
    args: argparse.Namespace,
    episode_ids,
    metrics: dict,
    alpha_bar: float,
) -> str:
    metrics_path = os.path.join(experiment_dir, "conformal_metrics.npz")
    output = {
        "episodes": np.asarray(episode_ids, dtype=np.int64),
        "qj_per_episode": np.asarray(
            metrics["qj_per_episode"],
            dtype=np.float32,
        ),
        "radius_per_episode": np.asarray(
            metrics["radius_per_episode"],
            dtype=np.float32,
        ),
        "tube_coverage_per_episode": np.asarray(
            metrics["tube_coverage_per_episode"],
            dtype=np.float32,
        ),
        "crashes_per_episode": np.asarray(
            metrics["crashes_per_episode"],
            dtype=np.float32,
        ),
        "bad_crashes_per_episode": np.asarray(
            metrics["bad_crashes_per_episode"],
            dtype=np.float32,
        ),
        "safety_per_episode": np.asarray(
            metrics["safety_per_episode"],
            dtype=np.float32,
        ),
        "cumulative_reward_per_episode": np.asarray(
            metrics["cumulative_reward_per_episode"],
            dtype=np.float32,
        ),
        "cumulative_reward_per_run": np.asarray(
            metrics["cumulative_reward_runs_per_episode"],
            dtype=np.float32,
        ),
        "agent_locs_per_episode": np.asarray(
            metrics["agent_locs_per_episode"],
            dtype=np.float32,
        ),
        "predicted_traj_per_episode": np.asarray(
            metrics["predicted_traj_per_episode"],
            dtype=np.float32,
        ),
        "alpha": args.alpha,
        "delta": args.delta,
        "bar_alpha": alpha_bar,
        "kappa": args.kappa,
        "cbf_k1": args.cbf_k1,
        "cbf_solver": np.asarray(args.cbf_solver),
        "update_procedure": np.asarray(args.update_procedure),
    }

    if metrics["evaluation_scores_per_episode"]:
        scores = np.asarray(metrics["evaluation_scores_per_episode"], dtype=np.float64)
        radii = np.asarray(metrics["radius_per_episode"], dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] != len(radii) or not scores.shape[1]:
            raise ValueError("Assessment scores must have one nonempty row per recorded policy")
        if len(metrics["initial_reference_scores"]):
            scores = np.vstack([metrics["initial_reference_scores"], scores])
            radii = np.r_[args.r_init, radii]
        if not np.all(np.isfinite(scores)):
            raise ValueError("Assessment scores must be finite")
        output.update(
            assessment_radii=radii,
            assessment_scores=scores,
            assessment_quantiles=np.quantile(
                scores, 1.0 - args.alpha, axis=1, method="inverted_cdf",
            ),
        )

    if metrics["anytime_trajectory_count_per_episode"]:
        config = metrics["anytime_config"]
        output.update(
            anytime_schema_version=np.asarray(2, dtype=np.int64),
            anytime_family=np.asarray(config["family"]),
            anytime_run_termination_reason=np.asarray(
                metrics["anytime_run_termination_reason"]
            ),
            anytime_current_radius_per_episode=np.asarray(
                metrics["anytime_current_radius_per_episode"], dtype=np.float64
            ),
            anytime_current_lower_radius_per_episode=np.asarray(
                metrics["anytime_current_lower_radius_per_episode"],
                dtype=np.float64,
            ),
            anytime_next_radius_per_episode=np.asarray(
                metrics["anytime_next_radius_per_episode"], dtype=np.float64
            ),
            anytime_next_lower_radius_per_episode=np.asarray(
                metrics["anytime_next_lower_radius_per_episode"],
                dtype=np.float64,
            ),
            anytime_current_width_per_episode=np.asarray(
                metrics["anytime_current_width_per_episode"], dtype=np.float64
            ),
            anytime_next_width_per_episode=np.asarray(
                metrics["anytime_next_width_per_episode"], dtype=np.float64
            ),
            tube_radius_per_episode=np.asarray(
                metrics["anytime_tube_radius_per_episode"], dtype=np.float64
            ),
            anytime_quantile_lower_per_episode=np.asarray(
                metrics["anytime_quantile_lower_per_episode"], dtype=np.float64
            ),
            anytime_quantile_upper_per_episode=np.asarray(
                metrics["anytime_quantile_upper_per_episode"], dtype=np.float64
            ),
            anytime_trajectory_count_per_episode=np.asarray(
                metrics["anytime_trajectory_count_per_episode"], dtype=np.int64
            ),
            anytime_exceedance_count_per_episode=np.asarray(
                metrics["anytime_exceedance_count_per_episode"], dtype=np.int64
            ),
            anytime_empirical_exceedance_per_episode=np.asarray(
                metrics["anytime_empirical_exceedance_per_episode"],
                dtype=np.float64,
            ),
            anytime_branch_per_episode=np.asarray(
                metrics["anytime_branch_per_episode"]
            ),
            anytime_termination_reason_per_episode=np.asarray(
                metrics["anytime_termination_reason_per_episode"]
            ),
            anytime_delta_per_episode=np.asarray(
                metrics["anytime_delta_per_episode"], dtype=np.float64
            ),
            anytime_radius_certificate_per_episode=np.asarray(
                metrics["anytime_radius_certificate_per_episode"],
                dtype=np.float64,
            ),
            anytime_scores=_pad_rows(
                metrics["anytime_scores_per_episode"], np.nan, np.float64
            ),
            anytime_indicators=_pad_rows(
                metrics["anytime_indicators_per_episode"], -1, np.int8
            ),
            anytime_sample_radii=_pad_rows(
                metrics["anytime_sample_radii_per_episode"], np.nan, np.float64
            ),
            anytime_quantile_lower_history=_pad_rows(
                metrics["anytime_quantile_lower_history_per_episode"],
                np.nan,
                np.float64,
            ),
            anytime_quantile_upper_history=_pad_rows(
                metrics["anytime_quantile_upper_history_per_episode"],
                np.nan,
                np.float64,
            ),
            anytime_few_critical_history=_pad_rows(
                metrics["anytime_few_critical_history_per_episode"],
                -2,
                np.int64,
            ),
            anytime_many_critical_history=_pad_rows(
                metrics["anytime_many_critical_history_per_episode"],
                -2,
                np.int64,
            ),
            anytime_candidate_radius_history=_pad_rows(
                metrics["anytime_candidate_radius_history_per_episode"],
                np.nan,
                np.float64,
            ),
            anytime_candidate_lower_radius_history=_pad_rows(
                metrics[
                    "anytime_candidate_lower_radius_history_per_episode"
                ],
                np.nan,
                np.float64,
            ),
            anytime_candidate_width_history=_pad_rows(
                metrics["anytime_candidate_width_history_per_episode"],
                np.nan,
                np.float64,
            ),
            anytime_kappa=config["kappa"],
            anytime_gamma=config["gamma"],
            anytime_zeta=config["zeta"],
            anytime_radius_tolerance=config["radius_tolerance"],
            anytime_k_epsilon=(
                -1 if config["k_epsilon"] is None else config["k_epsilon"]
            ),
            anytime_min_radius=config["min_radius"],
            anytime_initial_width=config["initial_width"],
            anytime_initial_safety_mode=np.asarray(
                config["initial_safety_mode"]
            ),
            anytime_delta_init=config["delta_init"],
            anytime_initial_trajectory_count=config[
                "initial_trajectory_count"
            ],
            anytime_initial_exceedance_count=config[
                "initial_exceedance_count"
            ],
            anytime_initial_probability_lower=config[
                "initial_probability_lower"
            ],
            anytime_initial_probability_upper=config[
                "initial_probability_upper"
            ],
            anytime_initial_scores=np.asarray(
                config["initial_scores"], dtype=np.float64
            ),
            anytime_initial_indicators=np.asarray(
                config["initial_indicators"], dtype=np.int8
            ),
        )
        if config["family"] in INFORMED_FAMILIES:
            priors = metrics["anytime_priors"]
            for field in (
                "a", "b", "mean", "concentration", "source_count", "source_radius",
                "shift", "near_fraction", "jeffreys_weight", "directional_weight",
            ):
                output[f"anytime_prior_{field}_per_episode"] = np.asarray(
                    [getattr(prior, field) for prior in priors],
                    dtype=np.int64 if field == "source_count" else np.float64,
                )
        if len(metrics["anytime_reference_scores"]):
            reference_scores = np.asarray(metrics["anytime_reference_scores"], dtype=np.float64)
            output["anytime_reference_scores"] = reference_scores
            output["anytime_reference_quantile_per_episode"] = np.quantile(
                reference_scores, 1.0 - args.alpha, axis=1, method="inverted_cdf",
            )

    np.savez(metrics_path, **output)
    return metrics_path


def _validate_predictions(
    pred_trajectories,
    num_multi_agents: int,
    episode_length: int,
):
    predictions = np.asarray(pred_trajectories)
    if predictions.ndim != 3 or predictions.shape[0] != num_multi_agents:
        raise ValueError(
            "Predictions must have shape (num_multi_agents, steps, features)."
        )
    if predictions.shape[1] < episode_length:
        raise ValueError(
            "Predicted trajectories are shorter than --episode_length."
        )
    if predictions.shape[2] < 6:
        raise ValueError(
            "Predicted trajectories must contain position and velocity."
        )
    return [predictions[i] for i in range(num_multi_agents)]


def main() -> None:
    args = parse_args()
    print(f"ECBF gain={args.cbf_k1} solver={args.cbf_solver}")

    MAX_RADIUS = args.r_init

    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device gpu was requested, but CUDA is not available."
            )
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    torch.set_grad_enabled(False)
    register_swarm_components()

    experiment_dir = ensure_experiment_dir(
        args.train_dir,
        args.experiment_name,
    )
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    if args.num_multi_agents < 0:
        args.num_multi_agents = int(cfg_multi.quads_num_agents)
    if args.num_multi_agents <= 0:
        raise ValueError("--num_multi_agents must be positive.")

    args_path = os.path.join(experiment_dir, "conformal_args.json")
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(
            {**vars(args), "evaluation_scenario": EVALUATION_SCENARIO},
            handle,
            indent=2,
            sort_keys=True,
        )

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        f"--device={args.device}",
        f"--quads_mode={EVALUATION_SCENARIO}",
        f"--quads_num_agents={args.num_multi_agents + 1}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_collision_reward=8.0",
        f"--quads_collision_hitbox_radius={COLLISION_HITBOX_RADIUS}",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=True",
        "--quads_view_mode=topdown",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    env = make_quadrotor_env(
        "quadrotor_multi",
        cfg=eval_cfg,
        render_mode="rgb_array",
    )

    try:
        multi_ckpt = latest_checkpoint(
            args.multi_train_dir,
            args.multi_experiment,
            policy_index=0,
        )
        multi_actor = load_actor(
            cfg_multi,
            env.observation_space,
            env.action_space,
            multi_ckpt,
            device,
        )
        multi_rnn_size = get_rnn_size(cfg_multi)

        cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
        solo_env = make_quadrotor_env(
            "quadrotor_multi",
            cfg=cfg_solo,
            render_mode=None,
        )
        try:
            solo_ckpt = latest_checkpoint(
                args.solo_train_dir,
                args.solo_experiment,
                policy_index=0,
            )
            solo_actor = load_actor(
                cfg_solo,
                solo_env.observation_space,
                solo_env.action_space,
                solo_ckpt,
                device,
            )
            solo_obs_dim = int(solo_env.observation_space.shape[0])
            solo_rnn_size = get_rnn_size(cfg_solo)
        finally:
            solo_env.close()

        initial_obs, stored_states = deterministic_reset(env, args.seed, None)
        initial_snapshot = safe_capture_env_snapshot(env)

        def collect_rollouts(
            pred_trajectories,
            solo_action_fn,
            num_runs: int,
            deterministic: bool,
            first_run_deterministic: bool = True,
            show_progress: bool = True,
        ):
            multi_rnn_states = torch.zeros(
                (args.num_multi_agents, multi_rnn_size),
                dtype=torch.float32,
                device=device,
            )
            solo_rnn_states = torch.zeros(
                (1, solo_rnn_size),
                dtype=torch.float32,
                device=device,
            )
            temp_env = clone_env_from_snapshot(initial_snapshot)
            try:
                return run_multi_agents(
                    temp_env,
                    initial_obs,
                    args.num_multi_agents,
                    multi_actor,
                    multi_rnn_states,
                    solo_actor,
                    solo_rnn_states,
                    solo_obs_dim,
                    pred_trajectories,
                    solo_action_fn,
                    max_steps=args.episode_length,
                    num_runs=num_runs,
                    deterministic=deterministic,
                    first_run_deterministic=first_run_deterministic,
                    show_progress=show_progress,
                )
            finally:
                temp_env.close()

        if args.init_predictions is None:
            print("Collecting homogeneous predicted trajectories")
            prediction_rnn_states = torch.zeros(
                (args.num_multi_agents + 1, multi_rnn_size),
                dtype=torch.float32,
                device=device,
            )
            all_predictions = collect_homogeneous_trajectories(
                env,
                initial_obs,
                multi_actor,
                prediction_rnn_states,
                args.episode_length,
            )
            pred_trajectories = all_predictions[: args.num_multi_agents]
        else:
            pred_path = os.path.join(
                args.predictions_dir,
                args.init_predictions,
                "pred_trajectories.npz",
            )
            pred_trajectories = load_prediction_artifact(
                pred_path,
                expected_num_multi_agents=args.num_multi_agents,
                expected_episode_length=args.episode_length,
                expected_seed=args.seed,
                expected_policy_experiment=args.multi_experiment,
                expected_policy_checkpoint=multi_ckpt,
                expected_initial_states=stored_states,
            )

        pred_trajectories = _validate_predictions(
            pred_trajectories,
            args.num_multi_agents,
            args.episode_length,
        )

        # Do not reseed by default: startup has already advanced the legacy stream.
        if args.run_seed is not None:
            set_global_seed(args.run_seed)

        min_radius = env.quad_arm * COLLISION_HITBOX_RADIUS
        if args.update_procedure in ANYTIME_PROCEDURES:
            alpha_bar = np.nan
        else:
            alpha_bar = get_alpha_bar(
                args.alpha,
                args.delta,
                args.num_trajectories,
            )
            if not 0.0 <= alpha_bar < 1.0:
                raise ValueError(
                    "The adjusted alpha is outside [0, 1); increase "
                    "--num_trajectories or adjust --alpha/--delta."
                )

        def collect_for_radius(
            radius: float,
            num_runs: int,
            *,
            solo_action_fn=None,
            deterministic=None,
            first_run_deterministic: bool = True,
            show_progress: bool = True,
        ):
            radii = np.full(
                args.num_multi_agents,
                radius,
                dtype=np.float64,
            )
            if solo_action_fn is None:
                solo_action_fn = make_cbf_filter(
                    radii, cbf_k1=args.cbf_k1, solver_mode=args.cbf_solver,
                )
            if deterministic is None:
                deterministic = args.deterministic
            return collect_rollouts(
                pred_trajectories,
                solo_action_fn,
                num_runs=num_runs,
                deterministic=deterministic,
                first_run_deterministic=first_run_deterministic,
                show_progress=show_progress,
            )

        def calibrate(radius: float) -> float:
            logs = collect_for_radius(radius, args.num_trajectories)
            return joint_conformal_radius(
                logs,
                args.num_multi_agents,
                pred_trajectories,
                alpha_bar,
                args.episode_length,
                min_radius,
            )

        metrics = _empty_metrics()
        initial_predictions = [prediction.copy() for prediction in pred_trajectories]

        def evaluate_and_record(radius: float, qj: float):
            logs = collect_for_radius(radius, args.num_eval_trajs)
            summary = summarize_rollouts(
                logs,
                pred_trajectories,
                radius,
                min_radius,
                args.num_multi_agents,
                args.episode_length,
                args.num_eval_trajs,
            )
            _record_episode(
                metrics,
                qj,
                radius,
                summary,
                pred_trajectories,
                evaluation_scores(logs),
            )
            print(
                f"Cum rew: {summary['cumulative_reward']} "
                f"Tube cov: {summary['tube_coverage']} "
                f"Crashes: {summary['crashes']} "
                f"Bad crashes: {summary['bad_crashes']}"
            )
            return logs

        def update_predictions_from_logs(logs) -> None:
            for agent_id in range(args.num_multi_agents):
                pred_trajectories[agent_id][
                    : args.episode_length,
                    :3,
                ] = logs[agent_id][0]["position"]
                pred_trajectories[agent_id][
                    : args.episode_length,
                    3:6,
                ] = logs[agent_id][0]["velocity"]

        def evaluation_scores(logs):
            return np.asarray([
                min_radius + joint_rollout_error(
                    logs, pred_trajectories, args.num_multi_agents,
                    args.episode_length, run_id,
                )
                for run_id in range(args.num_eval_trajs)
            ], dtype=np.float64)

        def evaluate_anytime_and_record(radius: float):
            radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
            action_filter = make_cbf_filter(
                radii, cbf_k1=args.cbf_k1, solver_mode=args.cbf_solver,
            )
            # This deterministic trajectory is for plots/prediction updates only.
            reference_logs = collect_for_radius(
                radius,
                1,
                solo_action_fn=action_filter,
                deterministic=True,
                first_run_deterministic=False,
                show_progress=False,
            )
            logs = collect_for_radius(
                radius,
                args.num_eval_trajs,
                solo_action_fn=action_filter,
                deterministic=args.deterministic,
                first_run_deterministic=False,
            )
            summary = summarize_rollouts(
                logs,
                pred_trajectories,
                radius,
                min_radius,
                args.num_multi_agents,
                args.episode_length,
                args.num_eval_trajs,
            )
            summary["agent_locs"] = [
                reference_logs[agent_id][0]["position"]
                for agent_id in range(args.num_multi_agents + 1)
            ]
            _record_episode(
                metrics,
                np.nan,
                radius,
                summary,
                pred_trajectories,
                evaluation_scores(logs),
            )
            print(
                f"Cum rew: {summary['cumulative_reward']} "
                f"Joint tube cov: {summary['tube_coverage']} "
                f"Crashes: {summary['crashes']} "
                f"Bad crashes: {summary['bad_crashes']}"
            )
            return reference_logs

        if args.update_procedure in ("robust", "naive"):
            radius = MAX_RADIUS
            episode_ids = np.arange(args.num_episodes)
            for episode in episode_ids:
                print("EPISODE", episode)
                qj = calibrate(radius)
                new_radius = next_radius(
                    args.update_procedure,
                    radius,
                    qj,
                    args.kappa,
                    min_radius,
                    MAX_RADIUS,
                )
                print(
                    "radius",
                    radius,
                    "qj",
                    qj,
                    "new radius",
                    new_radius,
                )
                radius = new_radius
                logs = evaluate_and_record(radius, qj)

                if args.update_predictions and episode == 0:
                    update_predictions_from_logs(logs)
        elif args.update_procedure in ("non-interactive", "calibrate-once-paper"):
            q0 = calibrate(MAX_RADIUS)
            evaluate_and_record(MAX_RADIUS, q0)

            radius = (
                paper_calibrate_once_radius(MAX_RADIUS, q0, args.kappa)
                if args.update_procedure == "calibrate-once-paper"
                else q0
            )
            print("r0", MAX_RADIUS, "q0", q0, "r1", radius)
            q1 = calibrate(radius)
            print("r1", radius, "q1", q1)
            evaluate_and_record(radius, q1)
            episode_ids = np.array([0, 1])
        elif args.update_procedure == "calibrate-once":
            q0 = calibrate(MAX_RADIUS)
            evaluate_and_record(MAX_RADIUS, q0)

            radius = q0
            print("r0", MAX_RADIUS, "q0", q0, "r1", radius)
            q1 = calibrate(radius)
            print("r1", radius, "q1", q1)
            evaluate_and_record(radius, q1)

            radius = calibrate_once_final_radius(
                MAX_RADIUS,
                q0,
                args.kappa,
                args.num_episodes,
            )
            q_final = calibrate(radius)
            print("radius", radius, "qj", q_final)
            evaluate_and_record(radius, q_final)
            episode_ids = np.array([0, 1, args.num_episodes + 1])
        elif args.update_procedure in ANYTIME_PROCEDURES:
            family = args.anytime_family
            radius = MAX_RADIUS
            lower_radius = min_radius
            if radius < lower_radius:
                raise ValueError(
                    "--r_init must be at least the physical minimum radius "
                    f"{lower_radius:.5f}."
                )

            initial_scores = np.empty(0, dtype=np.float64)
            initial_indicators = np.empty(0, dtype=np.int8)
            if args.anytime_assume_initial_safe:
                initial_safety_mode = "assumed"
                effective_delta_init = 0.0
                initial_probability_lower = np.nan
                initial_probability_upper = np.nan
                print(
                    f"Assuming initial radius r0={radius:.5f} is known safe."
                )
            else:
                initial_safety_mode = "clopper-pearson"
                effective_delta_init = args.anytime_delta_init
                print(
                    "Calibrating initial-radius safety with "
                    f"{args.anytime_init_trajectories} independent trajectories"
                )
                initial_logs = collect_for_radius(
                    radius,
                    args.anytime_init_trajectories,
                    deterministic=args.deterministic,
                    first_run_deterministic=False,
                )
                initial_scores = np.asarray(
                    [
                        min_radius
                        + joint_rollout_error(
                            initial_logs,
                            pred_trajectories,
                            args.num_multi_agents,
                            args.episode_length,
                            run_id,
                        )
                        for run_id in range(args.anytime_init_trajectories)
                    ],
                    dtype=np.float64,
                )
                initial_indicators = (initial_scores > radius).astype(np.int8)
                initial_probability_lower, initial_probability_upper = (
                    clopper_pearson_interval(
                        int(np.sum(initial_indicators)),
                        args.anytime_init_trajectories,
                        effective_delta_init,
                    )
                )
                print(
                    "Initial exceedance interval: "
                    f"[{initial_probability_lower:.5f}, "
                    f"{initial_probability_upper:.5f}]"
                )
                if initial_probability_upper > args.alpha:
                    raise RuntimeError(
                        "Initial radius is not certified safe: Clopper-Pearson "
                        f"upper bound {initial_probability_upper:.5f} exceeds "
                        f"alpha={args.alpha:.5f}. Increase --r_init or collect "
                        "more initial trajectories."
                    )

            gamma, zeta = contraction_factors(args.kappa)
            initial_width = radius - lower_radius
            k_epsilon = max_active_episodes(
                initial_width,
                args.anytime_radius_tolerance,
                args.kappa,
            )
            metrics["anytime_config"] = {
                "family": family,
                "kappa": args.kappa,
                "gamma": gamma,
                "zeta": zeta,
                "radius_tolerance": args.anytime_radius_tolerance,
                "k_epsilon": k_epsilon,
                "min_radius": min_radius,
                "initial_width": initial_width,
                "initial_safety_mode": initial_safety_mode,
                "delta_init": effective_delta_init,
                "initial_trajectory_count": len(initial_scores),
                "initial_exceedance_count": int(np.sum(initial_indicators)),
                "initial_probability_lower": initial_probability_lower,
                "initial_probability_upper": initial_probability_upper,
                "initial_scores": initial_scores,
                "initial_indicators": initial_indicators,
            }
            print(
                f"Anytime family={family} kappa={args.kappa} gamma={gamma:.5f} "
                f"zeta={zeta:.5f} initial bracket="
                f"[{lower_radius:.5f}, {radius:.5f}] k_epsilon={k_epsilon}"
            )

            if k_epsilon is None:
                episode_limit = args.num_episodes
            else:
                episode_limit = min(args.num_episodes, max(k_epsilon, 1))
            episode_ids = []
            previous_scores = initial_scores
            previous_radius = None

            for episode in range(episode_limit):
                print("EPISODE", episode)
                prior = None
                if family in INFORMED_FAMILIES:
                    prior = initialize_mixture_prior(
                        previous_scores, radius=radius, previous_radius=previous_radius,
                        kappa=args.kappa, jeffreys_weight=args.anytime_jeffreys_weight,
                        directional_weight=args.anytime_directional_weight,
                    )
                    print(
                        f"  prior a={prior.a:.6g} b={prior.b:.6g} "
                        f"source_n={prior.source_count} sigma={prior.shift:.6g} "
                        f"mu={prior.near_fraction:.6g}"
                    )
                episode_delta = episode_error_budget(
                    args.delta,
                    episode,
                    k_epsilon,
                )
                radii = np.full(
                    args.num_multi_agents,
                    radius,
                    dtype=np.float64,
                )
                action_filter = make_cbf_filter(
                    radii, cbf_k1=args.cbf_k1, solver_mode=args.cbf_solver,
                )
                sample_count = 0

                def sample_joint_score():
                    nonlocal sample_count
                    logs = collect_for_radius(
                        radius,
                        1,
                        solo_action_fn=action_filter,
                        deterministic=args.deterministic,
                        first_run_deterministic=False,
                        show_progress=False,
                    )
                    error = joint_rollout_error(
                        logs,
                        pred_trajectories,
                        args.num_multi_agents,
                        args.episode_length,
                    )
                    sample_count += 1
                    # ECBF radii include physical separation; prediction errors do not.
                    score = min_radius + error
                    if sample_count == 1 or sample_count % 25 == 0:
                        print(
                            f"  sample {sample_count}: score={score:.4f}, "
                            f"exceeds={score > radius}"
                        )
                    return score

                result = run_anytime_episode(
                    sample_joint_score,
                    radius=radius,
                    lower_radius=lower_radius,
                    alpha=args.alpha,
                    delta=episode_delta,
                    kappa=args.kappa,
                    radius_tolerance=args.anytime_radius_tolerance,
                    max_samples=args.num_trajectories,
                    family=family,
                    check_every=args.anytime_check_every,
                    prior=prior,
                )
                print(
                    f"status={result.branch} reason={result.termination_reason} "
                    f"n={result.num_samples} K={result.exceedances} "
                    f"q=[{result.quantile_lower:.5f}, "
                    f"{result.quantile_upper:.5f}] bracket="
                    f"[{result.lower_radius:.5f}, {result.radius:.5f}] -> "
                    f"[{result.next_lower_radius:.5f}, "
                    f"{result.next_radius:.5f}]"
                )

                _record_anytime_episode(
                    metrics,
                    result,
                    episode_delta,
                    min_radius,
                )
                evaluate_anytime_and_record(result.next_radius)
                episode_ids.append(episode)

                previous_scores = result.scores
                previous_radius = result.radius
                radius = result.next_radius
                lower_radius = result.next_lower_radius
                if result.branch == BRANCH_CERTIFICATE:
                    metrics["anytime_run_termination_reason"] = "certificate"
                    print(
                        "Certified bracket width "
                        f"{result.next_width:.5f} <= "
                        f"{args.anytime_radius_tolerance:.5f}; stopping."
                    )
                    break
                if result.branch != BRANCH_CONSERVATIVE:
                    metrics["anytime_run_termination_reason"] = (
                        result.termination_reason
                    )
                    print(
                        "Anytime update terminated unresolved: "
                        f"{result.termination_reason}."
                    )
                    break
            else:
                metrics["anytime_run_termination_reason"] = (
                    "driver-cap"
                    if args.num_episodes <= episode_limit
                    else "episode-budget-exhausted"
                )

            episode_ids = np.asarray(episode_ids, dtype=np.int64)

        if args.update_procedure in ("robust", "naive", *ANYTIME_PROCEDURES):
            # Assess r0 last so plotting data cannot alter any update's RNG stream.
            # Its predictor must also be the original one if predictions were updated.
            print("Collecting initial-policy quantile reference")
            final_predictions = pred_trajectories
            pred_trajectories = initial_predictions
            try:
                initial_reference_logs = collect_for_radius(
                    MAX_RADIUS, args.num_eval_trajs,
                    deterministic=args.deterministic,
                    first_run_deterministic=False,
                )
                metrics["initial_reference_scores"] = evaluation_scores(initial_reference_logs)
            finally:
                pred_trajectories = final_predictions

        if args.update_procedure in ANYTIME_PROCEDURES:
            # Assessment after update j belongs to the bounds for episode j+1.
            metrics["anytime_reference_scores"] = np.asarray([
                metrics["initial_reference_scores"],
                *metrics["evaluation_scores_per_episode"][:-1],
            ], dtype=np.float64)

        metrics_path = _save_metrics(
            experiment_dir,
            args,
            episode_ids,
            metrics,
            alpha_bar,
        )
        print(f"[conformal] Saved per-episode metrics to {metrics_path}")
        print(
            "Number of total crashes:",
            sum(metrics["crashes_per_episode"]),
        )
        print(
            "Number of total crashes outside of conformal tubes:",
            sum(metrics["bad_crashes_per_episode"]),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
