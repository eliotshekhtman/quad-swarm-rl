#!/usr/bin/env python3
"""Run fixed-sample or anytime-valid conformal CBF experiments."""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch

from sample_factory.model.model_utils import get_rnn_size

from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.anytime_utils import (
    BRANCH_CERTIFICATE,
    episode_error_budget,
    probability_slopes,
    run_anytime_episode,
)
from project_utils.conformal_utils import (
    calibrate_once_final_radius,
    get_alpha_bar,
    joint_conformal_radius,
    joint_rollout_error,
    next_radius,
    run_multi_agents,
    summarize_rollouts,
)
from project_utils.full_cbf_utils import make_cbf_filter
from project_utils.prediction_utils import (
    EVALUATION_SCENARIO,
    collect_homogeneous_trajectories,
    load_prediction_artifact,
)
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import latest_checkpoint, load_actor, load_cfg


DEFAULT_INITIAL_RADIUS = 2.0
KAPPA = 0.6
COLLISION_HITBOX_RADIUS = 2.5
UPDATE_PROCEDURES = (
    "robust",
    "naive",
    "non-interactive",
    "calibrate-once",
    "anytime",
)


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
    parser.add_argument("--r_init", type=float, default=DEFAULT_INITIAL_RADIUS)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.1)
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
        help="Replace predictions with the first evaluation run after episode 0.",
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
        help="Radius update rule. Defaults to the legacy conformal.py behavior.",
    )
    parser.add_argument(
        "--anytime-score-density-lower",
        type=float,
        default=1.0,
        help="Fixed uniform lower bound on score density for the anytime mode.",
    )
    parser.add_argument(
        "--anytime-score-density-upper",
        type=float,
        default=1.0,
        help="Fixed uniform upper bound on score density for the anytime mode.",
    )
    parser.add_argument("--anytime-relative-precision", type=float, default=3.0)
    parser.add_argument("--anytime-radius-tolerance", type=float, default=0.15)
    parser.add_argument("--anytime-eta-min", type=float, default=0.2)
    parser.add_argument("--anytime-damping-gamma", type=float, default=0.02)
    parser.add_argument("--anytime-grid-size", type=int, default=601)
    parser.add_argument(
        "--anytime-check-every",
        type=int,
        default=1,
        help="Check stopping conditions every N complete joint rollouts.",
    )
    args = parser.parse_args(argv)

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
    non_iterative_procedures = {"non-interactive", "calibrate-once"}
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
        not np.isfinite(args.anytime_score_density_lower)
        or args.anytime_score_density_lower <= 0.0
    ):
        parser.error("--anytime-score-density-lower must be positive")
    if (
        not np.isfinite(args.anytime_score_density_upper)
        or args.anytime_score_density_upper < args.anytime_score_density_lower
    ):
        parser.error(
            "--anytime-score-density-upper must be at least the lower bound"
        )
    if (
        not np.isfinite(args.anytime_relative_precision)
        or args.anytime_relative_precision <= 0.0
    ):
        parser.error("--anytime-relative-precision must be positive")
    if (
        not np.isfinite(args.anytime_radius_tolerance)
        or args.anytime_radius_tolerance < 0.0
    ):
        parser.error("--anytime-radius-tolerance must be non-negative")
    if (
        not np.isfinite(args.anytime_eta_min)
        or not 0.0 < args.anytime_eta_min <= 1.0
    ):
        parser.error("--anytime-eta-min must lie in (0, 1]")
    if (
        not np.isfinite(args.anytime_damping_gamma)
        or args.anytime_damping_gamma <= 0.0
    ):
        parser.error("--anytime-damping-gamma must be positive")
    if args.anytime_grid_size <= 0:
        parser.error("--anytime-grid-size must be positive")
    if args.anytime_check_every <= 0:
        parser.error("--anytime-check-every must be positive")

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
        "anytime_current_radius_per_episode": [],
        "anytime_next_radius_per_episode": [],
        "anytime_tube_radius_per_episode": [],
        "anytime_lower_probability_per_episode": [],
        "anytime_upper_probability_per_episode": [],
        "anytime_confidence_width_per_episode": [],
        "anytime_trajectory_count_per_episode": [],
        "anytime_exceedance_count_per_episode": [],
        "anytime_empirical_exceedance_per_episode": [],
        "anytime_branch_per_episode": [],
        "anytime_delta_per_episode": [],
        "anytime_eta_per_episode": [],
        "anytime_radius_certificate_per_episode": [],
        "anytime_safe_flag_per_episode": [],
        "anytime_next_safe_flag_per_episode": [],
        "anytime_stopped_forecast_per_episode": [],
        "anytime_log_hedge_conservative_per_episode": [],
        "anytime_log_hedge_under_per_episode": [],
        "anytime_scores_per_episode": [],
        "anytime_indicators_per_episode": [],
        "anytime_sample_radii_per_episode": [],
        "anytime_lower_history_per_episode": [],
        "anytime_upper_history_per_episode": [],
        "anytime_config": None,
    }


def _record_episode(
    metrics: dict,
    qj: float,
    radius: float,
    summary: dict,
    pred_trajectories,
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


def _record_anytime_episode(
    metrics: dict,
    result,
    episode_delta: float,
    min_radius: float,
) -> None:
    metrics["anytime_current_radius_per_episode"].append(result.radius)
    metrics["anytime_next_radius_per_episode"].append(result.next_radius)
    metrics["anytime_tube_radius_per_episode"].append(
        max(result.next_radius - min_radius, 0.0)
    )
    metrics["anytime_lower_probability_per_episode"].append(result.lower)
    metrics["anytime_upper_probability_per_episode"].append(result.upper)
    metrics["anytime_confidence_width_per_episode"].append(result.width)
    metrics["anytime_trajectory_count_per_episode"].append(result.num_samples)
    metrics["anytime_exceedance_count_per_episode"].append(result.exceedances)
    metrics["anytime_empirical_exceedance_per_episode"].append(
        result.exceedances / result.num_samples
    )
    metrics["anytime_branch_per_episode"].append(result.branch)
    metrics["anytime_delta_per_episode"].append(episode_delta)
    metrics["anytime_eta_per_episode"].append(result.eta)
    metrics["anytime_radius_certificate_per_episode"].append(
        result.radius_certificate
    )
    metrics["anytime_safe_flag_per_episode"].append(result.safe)
    metrics["anytime_next_safe_flag_per_episode"].append(result.next_safe)
    metrics["anytime_stopped_forecast_per_episode"].append(
        result.stopped_forecast
    )
    metrics["anytime_log_hedge_conservative_per_episode"].append(
        result.log_hedge_conservative
    )
    metrics["anytime_log_hedge_under_per_episode"].append(
        result.log_hedge_under
    )
    metrics["anytime_scores_per_episode"].append(result.scores.copy())
    metrics["anytime_indicators_per_episode"].append(result.indicators.copy())
    metrics["anytime_sample_radii_per_episode"].append(
        np.full(result.num_samples, result.radius, dtype=np.float64)
    )
    metrics["anytime_lower_history_per_episode"].append(
        result.lower_history.copy()
    )
    metrics["anytime_upper_history_per_episode"].append(
        result.upper_history.copy()
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
        "update_procedure": np.asarray(args.update_procedure),
    }

    if metrics["anytime_trajectory_count_per_episode"]:
        config = metrics["anytime_config"]
        output.update(
            anytime_current_radius_per_episode=np.asarray(
                metrics["anytime_current_radius_per_episode"], dtype=np.float64
            ),
            anytime_next_radius_per_episode=np.asarray(
                metrics["anytime_next_radius_per_episode"], dtype=np.float64
            ),
            tube_radius_per_episode=np.asarray(
                metrics["anytime_tube_radius_per_episode"], dtype=np.float64
            ),
            anytime_lower_probability_per_episode=np.asarray(
                metrics["anytime_lower_probability_per_episode"], dtype=np.float64
            ),
            anytime_upper_probability_per_episode=np.asarray(
                metrics["anytime_upper_probability_per_episode"], dtype=np.float64
            ),
            anytime_confidence_width_per_episode=np.asarray(
                metrics["anytime_confidence_width_per_episode"], dtype=np.float64
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
            anytime_delta_per_episode=np.asarray(
                metrics["anytime_delta_per_episode"], dtype=np.float64
            ),
            anytime_eta_per_episode=np.asarray(
                metrics["anytime_eta_per_episode"], dtype=np.float64
            ),
            anytime_radius_certificate_per_episode=np.asarray(
                metrics["anytime_radius_certificate_per_episode"],
                dtype=np.float64,
            ),
            anytime_safe_flag_per_episode=np.asarray(
                metrics["anytime_safe_flag_per_episode"], dtype=bool
            ),
            anytime_next_safe_flag_per_episode=np.asarray(
                metrics["anytime_next_safe_flag_per_episode"], dtype=bool
            ),
            anytime_stopped_forecast_per_episode=np.asarray(
                metrics["anytime_stopped_forecast_per_episode"], dtype=np.float64
            ),
            anytime_log_hedge_conservative_per_episode=np.asarray(
                metrics["anytime_log_hedge_conservative_per_episode"],
                dtype=np.float64,
            ),
            anytime_log_hedge_under_per_episode=np.asarray(
                metrics["anytime_log_hedge_under_per_episode"], dtype=np.float64
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
            anytime_lower_probability_history=_pad_rows(
                metrics["anytime_lower_history_per_episode"], np.nan, np.float64
            ),
            anytime_upper_probability_history=_pad_rows(
                metrics["anytime_upper_history_per_episode"], np.nan, np.float64
            ),
            anytime_kappa=config["kappa"],
            anytime_density_lower=config["density_lower"],
            anytime_density_upper=config["density_upper"],
            anytime_lower_slope=config["lower_slope"],
            anytime_upper_slope=config["upper_slope"],
            anytime_min_radius=config["min_radius"],
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

        min_radius = env.quad_arm * COLLISION_HITBOX_RADIUS
        if args.update_procedure == "anytime":
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
                solo_action_fn = make_cbf_filter(radii)
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

        def evaluate_anytime_and_record(radius: float):
            radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
            action_filter = make_cbf_filter(radii)
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
                    KAPPA,
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
        elif args.update_procedure == "non-interactive":
            q0 = calibrate(MAX_RADIUS)
            evaluate_and_record(MAX_RADIUS, q0)

            radius = q0
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
                KAPPA,
                args.num_episodes,
            )
            q_final = calibrate(radius)
            print("radius", radius, "qj", q_final)
            evaluate_and_record(radius, q_final)
            episode_ids = np.array([0, 1, args.num_episodes + 1])
        else:
            if args.update_predictions:
                warnings.warn(
                    "Updating predictions changes the radius-indexed policy family; "
                    "the paper's fixed-predictor guarantee no longer applies.",
                    stacklevel=2,
                )

            lower_slope, upper_slope = probability_slopes(
                KAPPA,
                args.anytime_score_density_lower,
                args.anytime_score_density_upper,
            )
            metrics["anytime_config"] = {
                "kappa": KAPPA,
                "density_lower": args.anytime_score_density_lower,
                "density_upper": args.anytime_score_density_upper,
                "lower_slope": lower_slope,
                "upper_slope": upper_slope,
                "min_radius": min_radius,
            }
            print(
                "Anytime fixed assumptions: "
                f"kappa={KAPPA}, "
                f"density=[{args.anytime_score_density_lower}, "
                f"{args.anytime_score_density_upper}], "
                f"m_p={lower_slope}, L_p={upper_slope}"
            )

            radius = MAX_RADIUS
            safe = True
            nominal_forecast = args.alpha / 2.0
            episode_ids = []

            for episode in range(args.num_episodes):
                print("EPISODE", episode)
                episode_delta = episode_error_budget(args.delta, episode)
                radii = np.full(
                    args.num_multi_agents,
                    radius,
                    dtype=np.float64,
                )
                action_filter = make_cbf_filter(radii)
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
                    safe=safe,
                    alpha=args.alpha,
                    delta=episode_delta,
                    lower_slope=lower_slope,
                    upper_slope=upper_slope,
                    relative_precision=args.anytime_relative_precision,
                    radius_tolerance=args.anytime_radius_tolerance,
                    eta_min=args.anytime_eta_min,
                    damping_gamma=args.anytime_damping_gamma,
                    min_radius=min_radius,
                    max_radius=MAX_RADIUS,
                    max_samples=args.num_trajectories,
                    nominal_forecast=nominal_forecast,
                    grid_size=args.anytime_grid_size,
                    check_every=args.anytime_check_every,
                )
                print(
                    f"branch={result.branch} n={result.num_samples} "
                    f"K={result.exceedances} "
                    f"p=[{result.lower:.5f}, {result.upper:.5f}] "
                    f"radius={result.radius:.5f} -> {result.next_radius:.5f}"
                )

                _record_anytime_episode(
                    metrics,
                    result,
                    episode_delta,
                    min_radius,
                )
                reference_logs = evaluate_anytime_and_record(result.next_radius)
                episode_ids.append(episode)

                if args.update_predictions and episode == 0:
                    update_predictions_from_logs(reference_logs)

                radius = result.next_radius
                safe = result.next_safe
                nominal_forecast = result.stopped_forecast
                if result.branch == BRANCH_CERTIFICATE:
                    print(
                        "Certified radius error "
                        f"{result.radius_certificate:.5f} <= "
                        f"{args.anytime_radius_tolerance:.5f}; stopping."
                    )
                    break

            episode_ids = np.asarray(episode_ids, dtype=np.int64)

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
