#!/usr/bin/env python3
"""Rank go_to_goal scenario seeds by planner-induced interaction."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Sequence

import numpy as np
import torch
from sample_factory.model.model_utils import get_rnn_size

from collect_video import _apply_rollout_seed, _build_eval_cfg, _collect_trajectory
from project_utils.full_cbf_utils import make_cbf_filter
from project_utils.prediction_utils import collect_homogeneous_trajectories
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import latest_checkpoint, load_actor, load_cfg
from swarm_rl.env_snapshot import safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import register_swarm_components


DETAIL_FIELDS = [
    "scenario_seed",
    "radius",
    "initial_min_distance",
    "ego_goal_distance",
    "nominal_min_distance",
    "nominal_closest_step",
    "nominal_closest_agent",
    "nominal_close_fraction",
    "baseline_min_distance",
    "baseline_closest_step",
    "baseline_closest_agent",
    "baseline_close_fraction",
    "baseline_prediction_min_distance",
    "filtered_min_distance",
    "filtered_closest_step",
    "filtered_closest_agent",
    "filtered_close_fraction",
    "action_delta_mean",
    "action_delta_max",
    "action_delta_fraction",
    "ego_response_max",
    "other_response_max",
    "other_response_max_step",
    "other_response_max_agent",
    "other_response_mean",
    "other_response_final_max",
    "baseline_length",
    "filtered_length",
    "common_length",
]

SUMMARY_FIELDS = [
    "scenario_seed",
    "interesting",
    "best_radius",
    *DETAIL_FIELDS[2:],
    "encounter_min_distance",
    "passes_initial_clearance",
    "passes_goal_travel",
    "passes_encounter",
    "passes_intervention",
    "passes_other_response",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank go_to_goal scenario seeds using paired deterministic rollouts. "
            "Interesting seeds require a synchronized encounter, ECBF intervention, "
            "and a resulting change in non-ego trajectories."
        )
    )
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", default="multi_ddg_6")
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--solo_experiment", default="solo_ddg")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument(
        "--seed_stop", type=int, default=100, help="Exclusive scenario-seed bound."
    )
    parser.add_argument("--seed_step", type=int, default=1)
    parser.add_argument(
        "--rollout_seed",
        type=int,
        default=0,
        help="Fixed disturbance seed shared by every paired rollout.",
    )
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--num_multi_agents", type=int, default=5)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.35, 0.6, 1.0, 1.4],
        help="Candidate ECBF radii evaluated independently for each scenario seed.",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--proximity_threshold", type=float, default=0.75)
    parser.add_argument("--action_delta_threshold", type=float, default=0.01)
    parser.add_argument("--min_initial_distance", type=float, default=0.75)
    parser.add_argument("--min_ego_goal_distance", type=float, default=2.0)
    parser.add_argument("--max_encounter_distance", type=float, default=0.75)
    parser.add_argument("--min_action_fraction", type=float, default=0.10)
    parser.add_argument("--min_other_response_max", type=float, default=0.15)
    parser.add_argument("--min_other_response_mean", type=float, default=0.02)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_path", default="train_dir/go_to_goal_seed_sweep.csv")
    parser.add_argument("--detail_path", default=None)
    args = parser.parse_args(argv)

    if args.seed_stop <= args.seed_start:
        parser.error("--seed_stop must be greater than --seed_start")
    if args.seed_step <= 0:
        parser.error("--seed_step must be positive")
    if args.episode_length <= 0:
        parser.error("--episode_length must be positive")
    if args.num_multi_agents <= 0:
        parser.error("--num_multi_agents must be positive")
    if not args.radii or any(radius <= 0.0 for radius in args.radii):
        parser.error("--radii must contain only positive values")
    positive_values = {
        "--proximity_threshold": args.proximity_threshold,
        "--max_encounter_distance": args.max_encounter_distance,
    }
    for name, value in positive_values.items():
        if value <= 0.0:
            parser.error(f"{name} must be positive")
    nonnegative_values = {
        "--action_delta_threshold": args.action_delta_threshold,
        "--min_initial_distance": args.min_initial_distance,
        "--min_ego_goal_distance": args.min_ego_goal_distance,
        "--min_other_response_max": args.min_other_response_max,
        "--min_other_response_mean": args.min_other_response_mean,
    }
    for name, value in nonnegative_values.items():
        if value < 0.0:
            parser.error(f"{name} must be non-negative")
    if not 0.0 <= args.min_action_fraction <= 1.0:
        parser.error("--min_action_fraction must lie in [0, 1]")
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    return args


def _csv_path(path: str) -> str:
    return path if path.endswith(".csv") else f"{path}.csv"


def _proximity(
    ego_positions: np.ndarray,
    other_positions: np.ndarray,
    threshold: float,
) -> Dict[str, float | int]:
    """Compute time-aligned proximity, never asynchronous path distance."""
    ego = np.asarray(ego_positions, dtype=np.float64)
    others = np.asarray(other_positions, dtype=np.float64)
    if ego.ndim != 2 or ego.shape[1] != 3:
        raise ValueError("ego_positions must have shape (steps, 3).")
    if others.ndim != 3 or others.shape[2] != 3:
        raise ValueError("other_positions must have shape (agents, steps, 3).")
    if ego.shape[0] == 0 or others.shape[0] == 0:
        raise ValueError("Proximity requires at least one step and one other agent.")
    if others.shape[1] != ego.shape[0]:
        raise ValueError("Ego and other trajectories must have the same horizon.")

    distances = np.linalg.norm(others - ego[None, :, :], axis=2)
    agent, step = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return {
        "min_distance": float(distances[agent, step]),
        "closest_agent": int(agent),
        "closest_step": int(step),
        "close_fraction": float(np.mean(np.min(distances, axis=0) < threshold)),
    }


def _paired_metrics(
    baseline: Dict[str, np.ndarray],
    filtered: Dict[str, np.ndarray],
    predictions: np.ndarray,
    action_deltas: Sequence[float],
    *,
    num_multi_agents: int,
    proximity_threshold: float,
    action_delta_threshold: float,
) -> Dict[str, float | int]:
    """Measure the causal effect of the filter over paired trajectory steps."""
    baseline_positions = np.asarray(baseline["positions"], dtype=np.float64)
    filtered_positions = np.asarray(filtered["positions"], dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    baseline_length = int(baseline["trajectory_length"])
    filtered_length = int(filtered["trajectory_length"])
    common_length = min(baseline_length, filtered_length, predictions.shape[1])
    expected_shape = (num_multi_agents + 1, 3)
    if common_length <= 0:
        raise ValueError("Paired rollouts have no common steps.")
    if baseline_positions.ndim != 3 or baseline_positions.shape[1:] != expected_shape:
        raise ValueError("Baseline positions have an unexpected shape.")
    if filtered_positions.ndim != 3 or filtered_positions.shape[1:] != expected_shape:
        raise ValueError("Filtered positions have an unexpected shape.")
    if predictions.ndim != 3 or predictions.shape[0] != num_multi_agents:
        raise ValueError("Predictions have an unexpected shape.")

    baseline_positions = baseline_positions[:common_length]
    filtered_positions = filtered_positions[:common_length]
    predicted_positions = predictions[:, :common_length, :3]
    baseline_ego = baseline_positions[:, num_multi_agents]
    filtered_ego = filtered_positions[:, num_multi_agents]
    baseline_others = baseline_positions[:, :num_multi_agents]
    filtered_others = filtered_positions[:, :num_multi_agents]

    baseline_proximity = _proximity(
        baseline_ego, np.transpose(baseline_others, (1, 0, 2)), proximity_threshold
    )
    filtered_proximity = _proximity(
        filtered_ego, np.transpose(filtered_others, (1, 0, 2)), proximity_threshold
    )
    prediction_proximity = _proximity(
        baseline_ego, predicted_positions, proximity_threshold
    )
    ego_response = np.linalg.norm(filtered_ego - baseline_ego, axis=1)
    other_response = np.linalg.norm(filtered_others - baseline_others, axis=2)
    other_max_step, other_max_agent = np.unravel_index(
        int(np.argmax(other_response)), other_response.shape
    )
    deltas = np.asarray(action_deltas, dtype=np.float64)
    if deltas.ndim != 1 or deltas.size != filtered_length:
        raise ValueError("Expected one action delta per filtered rollout step.")

    return {
        "baseline_min_distance": baseline_proximity["min_distance"],
        "baseline_closest_step": baseline_proximity["closest_step"],
        "baseline_closest_agent": baseline_proximity["closest_agent"],
        "baseline_close_fraction": baseline_proximity["close_fraction"],
        "baseline_prediction_min_distance": prediction_proximity["min_distance"],
        "filtered_min_distance": filtered_proximity["min_distance"],
        "filtered_closest_step": filtered_proximity["closest_step"],
        "filtered_closest_agent": filtered_proximity["closest_agent"],
        "filtered_close_fraction": filtered_proximity["close_fraction"],
        "action_delta_mean": float(np.mean(deltas)),
        "action_delta_max": float(np.max(deltas)),
        "action_delta_fraction": float(np.mean(deltas > action_delta_threshold)),
        "ego_response_max": float(np.max(ego_response)),
        "other_response_max": float(np.max(other_response)),
        "other_response_max_step": int(other_max_step),
        "other_response_max_agent": int(other_max_agent),
        "other_response_mean": float(np.mean(other_response)),
        "other_response_final_max": float(np.max(other_response[-1])),
        "baseline_length": baseline_length,
        "filtered_length": filtered_length,
        "common_length": common_length,
    }


def _qualification_flags(row: Dict[str, float | int], args) -> Dict[str, int]:
    flags = {
        "passes_initial_clearance": int(
            row["initial_min_distance"] >= args.min_initial_distance
        ),
        "passes_goal_travel": int(
            row["ego_goal_distance"] >= args.min_ego_goal_distance
        ),
        "passes_encounter": int(
            row["encounter_min_distance"] <= args.max_encounter_distance
        ),
        "passes_intervention": int(
            row["action_delta_fraction"] >= args.min_action_fraction
        ),
        "passes_other_response": int(
            row["other_response_max"] >= args.min_other_response_max
            and row["other_response_mean"] >= args.min_other_response_mean
        ),
    }
    flags["interesting"] = int(all(flags.values()))
    return flags


def _rank_summaries(rows: Sequence[Dict[str, float | int]]):
    return sorted(
        rows,
        key=lambda row: (
            -int(row["interesting"]),
            -float(row["other_response_max"]),
            -float(row["other_response_mean"]),
            float(row["encounter_min_distance"]),
            int(row["scenario_seed"]),
        ),
    )


def _identity_action(base_action: np.ndarray, **_unused) -> np.ndarray:
    return np.asarray(base_action, dtype=np.float32).copy()


def _recording_filter(radius: float, num_multi_agents: int):
    cbf_filter = make_cbf_filter(
        np.full(num_multi_agents, radius, dtype=np.float64)
    )
    action_deltas = []

    def apply(base_action, env_state, swarm_state):
        base = np.asarray(base_action, dtype=np.float32)
        result = cbf_filter(
            base_action=base, env_state=env_state, swarm_state=swarm_state
        )
        action_deltas.append(float(np.linalg.norm(result - base)))
        return result

    return apply, action_deltas


def _initial_geometry(states, ego_index: int):
    positions = np.stack([np.asarray(state.position) for state in states])
    pair_distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )
    pair_distances[np.diag_indices_from(pair_distances)] = np.inf
    ego_goal_distance = np.linalg.norm(
        np.asarray(states[ego_index].goal) - positions[ego_index]
    )
    return float(np.min(pair_distances)), float(ego_goal_distance)


def _collect_paired_trajectory(
    snapshot,
    obs,
    args,
    multi_actor,
    multi_rnn_size,
    solo_actor,
    solo_rnn_size,
    solo_obs_dim,
    predictions,
    action_fn,
    device,
):
    return _collect_trajectory(
        snapshot,
        obs,
        num_multi_agents=args.num_multi_agents,
        multi_actor=multi_actor,
        multi_rnn_states=torch.zeros(
            (args.num_multi_agents, multi_rnn_size), device=device
        ),
        solo_actor=solo_actor,
        solo_rnn_states=torch.zeros((1, solo_rnn_size), device=device),
        solo_obs_dim=solo_obs_dim,
        pred_trajectories=predictions,
        solo_action_fn=action_fn,
        max_steps=args.episode_length,
        deterministic=True,
    )


def main() -> None:
    args = parse_args()
    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu was requested, but CUDA is unavailable.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    torch.set_grad_enabled(False)
    register_swarm_components()
    summary_path = _csv_path(args.output_path)
    detail_path = _csv_path(
        args.detail_path or f"{os.path.splitext(summary_path)[0]}_details"
    )
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(detail_path) or ".", exist_ok=True)

    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    eval_cfg = _build_eval_cfg(
        cfg_multi,
        num_agents=args.num_multi_agents + 1,
        enable_render=False,
        view_mode="topdownfollow",
    )
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)
    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    try:
        multi_actor = load_actor(
            cfg_multi,
            env.observation_space,
            env.action_space,
            latest_checkpoint(
                args.multi_train_dir, args.multi_experiment, policy_index=0
            ),
            device,
        )
        solo_actor = load_actor(
            cfg_solo,
            solo_env.observation_space,
            solo_env.action_space,
            latest_checkpoint(
                args.solo_train_dir, args.solo_experiment, policy_index=0
            ),
            device,
        )
        multi_rnn_size = get_rnn_size(cfg_multi)
        solo_rnn_size = get_rnn_size(cfg_solo)
        solo_obs_dim = int(solo_env.observation_space.shape[0])
    finally:
        solo_env.close()

    summaries = []
    seed_range = range(args.seed_start, args.seed_stop, args.seed_step)
    print(
        f"Sweeping {len(seed_range)} scenario seeds at radii {args.radii}; "
        f"paired rollout seed={args.rollout_seed}."
    )
    try:
        with open(summary_path, "w", newline="", encoding="utf-8") as summary_file, open(
            detail_path, "w", newline="", encoding="utf-8"
        ) as detail_file:
            summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
            detail_writer = csv.DictWriter(detail_file, fieldnames=DETAIL_FIELDS)
            summary_writer.writeheader()
            detail_writer.writeheader()

            for scenario_seed in seed_range:
                obs, states = deterministic_reset(env, scenario_seed, None)
                _apply_rollout_seed(env, args.rollout_seed)
                all_predictions = collect_homogeneous_trajectories(
                    env,
                    obs,
                    multi_actor,
                    torch.zeros(
                        (args.num_multi_agents + 1, multi_rnn_size), device=device
                    ),
                    args.episode_length,
                ).astype(np.float64)
                predictions = all_predictions[: args.num_multi_agents]
                initial_min, ego_goal_distance = _initial_geometry(
                    states, args.num_multi_agents
                )
                nominal = _proximity(
                    all_predictions[args.num_multi_agents, :, :3],
                    all_predictions[: args.num_multi_agents, :, :3],
                    args.proximity_threshold,
                )

                obs, _ = deterministic_reset(env, scenario_seed, states)
                _apply_rollout_seed(env, args.rollout_seed)
                snapshot = safe_capture_env_snapshot(env)
                rollout_args = (
                    snapshot,
                    obs,
                    args,
                    multi_actor,
                    multi_rnn_size,
                    solo_actor,
                    solo_rnn_size,
                    solo_obs_dim,
                    [trajectory for trajectory in predictions],
                )
                baseline = _collect_paired_trajectory(
                    *rollout_args, _identity_action, device
                )

                details = []
                for radius in args.radii:
                    action_fn, action_deltas = _recording_filter(
                        radius, args.num_multi_agents
                    )
                    filtered = _collect_paired_trajectory(
                        *rollout_args, action_fn, device
                    )
                    metrics = _paired_metrics(
                        baseline,
                        filtered,
                        predictions,
                        action_deltas,
                        num_multi_agents=args.num_multi_agents,
                        proximity_threshold=args.proximity_threshold,
                        action_delta_threshold=args.action_delta_threshold,
                    )
                    detail = {
                        "scenario_seed": scenario_seed,
                        "radius": radius,
                        "initial_min_distance": initial_min,
                        "ego_goal_distance": ego_goal_distance,
                        "nominal_min_distance": nominal["min_distance"],
                        "nominal_closest_step": nominal["closest_step"],
                        "nominal_closest_agent": nominal["closest_agent"],
                        "nominal_close_fraction": nominal["close_fraction"],
                        **metrics,
                    }
                    details.append(detail)
                    detail_writer.writerow(detail)
                    detail_file.flush()

                best = max(
                    details,
                    key=lambda row: (
                        row["other_response_max"],
                        row["other_response_mean"],
                        row["action_delta_fraction"],
                    ),
                )
                summary = {
                    "scenario_seed": scenario_seed,
                    "best_radius": best["radius"],
                    **{field: best[field] for field in DETAIL_FIELDS[2:]},
                }
                summary["encounter_min_distance"] = min(
                    summary["nominal_min_distance"], summary["baseline_min_distance"]
                )
                summary.update(_qualification_flags(summary, args))
                summaries.append(summary)
                summary_writer.writerow(summary)
                summary_file.flush()
                print(
                    f"seed={scenario_seed} interesting={bool(summary['interesting'])} "
                    f"goal={ego_goal_distance:.3f}m "
                    f"encounter={summary['encounter_min_distance']:.3f}m "
                    f"best_r={summary['best_radius']:.2f} "
                    f"action={summary['action_delta_fraction']:.1%} "
                    f"other_max={summary['other_response_max']:.3f}m "
                    f"other_mean={summary['other_response_mean']:.3f}m"
                )
    finally:
        env.close()

    ranked = _rank_summaries(summaries)
    print(f"\nQualified seeds: {sum(row['interesting'] for row in ranked)}/{len(ranked)}")
    print("Top candidates:")
    for rank, row in enumerate(ranked[: args.top_k], 1):
        print(
            f"{rank:2d}. seed={row['scenario_seed']} "
            f"interesting={bool(row['interesting'])} r={row['best_radius']:.2f} "
            f"encounter={row['encounter_min_distance']:.3f}m "
            f"action={row['action_delta_fraction']:.1%} "
            f"other_max={row['other_response_max']:.3f}m "
            f"other_mean={row['other_response_mean']:.3f}m"
        )
    print(f"Saved summaries to {os.path.abspath(summary_path)}")
    print(f"Saved per-radius details to {os.path.abspath(detail_path)}")


if __name__ == "__main__":
    main()
