#!/usr/bin/env python3
"""Rank go_to_goal seeds by causal ego interactions with multiple agents."""

from __future__ import annotations

import argparse
import csv
import os
from itertools import combinations
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


RADIUS_DETAIL_FIELDS = [
    "scenario_seed",
    "radius",
    "ego_initial_min_distance",
    "ego_goal_distance",
    "nominal_min_distance",
    "nominal_closest_step",
    "nominal_closest_agent",
    "nominal_close_fraction",
    "baseline_min_distance",
    "baseline_closest_step",
    "baseline_closest_agent",
    "baseline_close_fraction",
    "baseline_unique_close_agents",
    "baseline_prediction_min_distance",
    "filtered_min_distance",
    "filtered_closest_step",
    "filtered_closest_agent",
    "filtered_close_fraction",
    "filtered_unique_close_agents",
    "filtered_max_simultaneous_close_agents",
    "action_delta_mean",
    "action_delta_max",
    "action_delta_fraction",
    "ego_response_max",
    "other_response_max",
    "other_response_max_step",
    "other_response_max_agent",
    "other_response_mean",
    "other_response_final_max",
    "baseline_path_distance",
    "filtered_path_distance",
    "baseline_length",
    "filtered_length",
    "common_length",
    "full_completed",
    "num_direct_agents",
    "num_visible_agents",
    "num_causal_agents",
    "causal_agent_ids",
    "second_causal_heading_deg",
    "second_causal_deviation",
    "best_event_gap_frames",
    "has_sequential_pair",
    "has_overlapping_pair",
    "interesting",
]

AGENT_DETAIL_FIELDS = [
    "scenario_seed",
    "radius",
    "agent",
    "ego_initial_distance",
    "actual_min_distance",
    "predicted_min_distance",
    "direct_active_frames",
    "direct_run_start",
    "direct_run_end",
    "marginal_action_delta_max",
    "marginal_action_delta_mean_active",
    "marginal_action_delta_integral",
    "actual_event_min_distance",
    "predicted_event_min_distance",
    "visible_turn_deg",
    "visible_horizontal_turn_deg",
    "direct_agent",
    "visible_agent",
    "leaveout_checked",
    "leaveout_completed",
    "causal_heading_max_deg",
    "causal_heading_frames",
    "causal_deviation_max",
    "causal_agent",
    "other_response_max",
    "other_response_mean",
    "qualification_reason",
]

SUMMARY_FIELDS = [
    "scenario_seed",
    "interesting",
    "best_radius",
    "qualifying_radius_count",
    "qualifying_radii",
    "radius_span_position_max",
    "radius_span_heading_max_deg",
    *[
        field
        for field in RADIUS_DETAIL_FIELDS[2:]
        if field != "interesting"
    ],
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank go_to_goal seeds by visible, per-agent causal effects on the ego "
            "trajectory under paired deterministic rollouts."
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
        default=[0.4, 0.5, 0.6],
        help="Low-radius ECBF values evaluated independently for each seed.",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")

    parser.add_argument("--proximity_threshold", type=float, default=0.75)
    parser.add_argument("--action_delta_threshold", type=float, default=0.01)
    parser.add_argument(
        "--min_agent_action_delta",
        type=float,
        default=0.01,
        help=(
            "Minimum same-state action change caused by including one neighbor "
            "for that neighbor to count as directly active."
        ),
    )
    parser.add_argument("--min_active_frames", type=int, default=10)
    parser.add_argument("--max_active_gap", type=int, default=2)
    parser.add_argument("--min_event_start", type=int, default=20)
    parser.add_argument("--event_end_margin", type=int, default=30)
    parser.add_argument("--event_proximity_margin", type=float, default=0.5)
    parser.add_argument("--heading_window", type=int, default=20)
    parser.add_argument("--min_heading_displacement", type=float, default=0.05)
    parser.add_argument("--min_visible_turn_deg", type=float, default=20.0)
    parser.add_argument("--causal_heading_half_window", type=int, default=10)
    parser.add_argument("--causal_post_frames", type=int, default=30)
    parser.add_argument("--min_causal_heading_deg", type=float, default=10.0)
    parser.add_argument("--min_causal_heading_frames", type=int, default=10)
    parser.add_argument("--min_causal_deviation", type=float, default=0.15)
    parser.add_argument("--min_event_gap", type=int, default=15)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--output_path", default="train_dir/go_to_goal_causal_seed_sweep.csv"
    )
    parser.add_argument("--detail_path", default=None)
    parser.add_argument("--agent_detail_path", default=None)
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
        "--min_active_frames": args.min_active_frames,
        "--event_end_margin": args.event_end_margin,
        "--heading_window": args.heading_window,
        "--min_heading_displacement": args.min_heading_displacement,
        "--causal_heading_half_window": args.causal_heading_half_window,
        "--causal_post_frames": args.causal_post_frames,
        "--min_causal_heading_frames": args.min_causal_heading_frames,
        "--top_k": args.top_k,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"{name} must be positive")

    nonnegative_values = {
        "--action_delta_threshold": args.action_delta_threshold,
        "--min_agent_action_delta": args.min_agent_action_delta,
        "--max_active_gap": args.max_active_gap,
        "--min_event_start": args.min_event_start,
        "--event_proximity_margin": args.event_proximity_margin,
        "--min_visible_turn_deg": args.min_visible_turn_deg,
        "--min_causal_heading_deg": args.min_causal_heading_deg,
        "--min_causal_deviation": args.min_causal_deviation,
        "--min_event_gap": args.min_event_gap,
    }
    for name, value in nonnegative_values.items():
        if value < 0:
            parser.error(f"{name} must be non-negative")

    if args.min_event_start < args.heading_window:
        parser.error("--min_event_start must be at least --heading_window")
    if args.event_end_margin < args.heading_window:
        parser.error("--event_end_margin must be at least --heading_window")
    return args


def _csv_path(path: str) -> str:
    return path if path.endswith(".csv") else f"{path}.csv"


def _proximity(
    ego_positions: np.ndarray,
    other_positions: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    """Compute only synchronized ego-to-other distances."""
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
    close = distances < threshold
    return {
        "min_distance": float(distances[agent, step]),
        "closest_agent": int(agent),
        "closest_step": int(step),
        "close_fraction": float(np.mean(np.any(close, axis=0))),
        "unique_close_agents": int(np.sum(np.any(close, axis=1))),
        "max_simultaneous_close_agents": int(np.max(np.sum(close, axis=0))),
        "per_agent_min_distance": np.min(distances, axis=1),
        "per_agent_closest_step": np.argmin(distances, axis=1),
        "per_agent_close_fraction": np.mean(close, axis=1),
        "distances": distances,
    }


def _ego_state_points(trajectory: Dict[str, np.ndarray], ego_index: int) -> np.ndarray:
    positions = np.asarray(trajectory["positions"], dtype=np.float64)[:, ego_index]
    if "initial_positions" not in trajectory:
        return positions
    initial = np.asarray(trajectory["initial_positions"], dtype=np.float64)[ego_index]
    return np.concatenate([initial[None, :], positions], axis=0)


def _path_distance(trajectory: Dict[str, np.ndarray], ego_index: int) -> float:
    points = _ego_state_points(trajectory, ego_index)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


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
    """Measure paired effects; non-ego response remains descriptive only."""
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
        "baseline_unique_close_agents": baseline_proximity["unique_close_agents"],
        "baseline_prediction_min_distance": prediction_proximity["min_distance"],
        "filtered_min_distance": filtered_proximity["min_distance"],
        "filtered_closest_step": filtered_proximity["closest_step"],
        "filtered_closest_agent": filtered_proximity["closest_agent"],
        "filtered_close_fraction": filtered_proximity["close_fraction"],
        "filtered_unique_close_agents": filtered_proximity["unique_close_agents"],
        "filtered_max_simultaneous_close_agents": filtered_proximity[
            "max_simultaneous_close_agents"
        ],
        "action_delta_mean": float(np.mean(deltas)),
        "action_delta_max": float(np.max(deltas)),
        "action_delta_fraction": float(np.mean(deltas > action_delta_threshold)),
        "ego_response_max": float(np.max(ego_response)),
        "other_response_max": float(np.max(other_response)),
        "other_response_max_step": int(other_max_step),
        "other_response_max_agent": int(other_max_agent),
        "other_response_mean": float(np.mean(other_response)),
        "other_response_final_max": float(np.max(other_response[-1])),
        "baseline_path_distance": _path_distance(baseline, num_multi_agents),
        "filtered_path_distance": _path_distance(filtered, num_multi_agents),
        "baseline_length": baseline_length,
        "filtered_length": filtered_length,
        "common_length": common_length,
    }


def _initial_geometry(states, ego_index: int) -> Dict[str, object]:
    positions = np.stack(
        [np.asarray(state.position, dtype=np.float64) for state in states]
    )
    ego_position = positions[ego_index]
    other_positions = np.delete(positions, ego_index, axis=0)
    ego_distances = np.linalg.norm(other_positions - ego_position[None, :], axis=1)
    ego_goal_distance = np.linalg.norm(
        np.asarray(states[ego_index].goal, dtype=np.float64) - ego_position
    )
    return {
        "ego_initial_min_distance": float(np.min(ego_distances)),
        "ego_initial_distances": ego_distances,
        "ego_goal_distance": float(ego_goal_distance),
    }


def _identity_action(base_action: np.ndarray, **_unused) -> np.ndarray:
    return np.asarray(base_action, dtype=np.float32).copy()


def _cbf_action_filter(
    radius: float, num_multi_agents: int, disabled_agent: int | None = None
):
    radii = np.full(num_multi_agents, radius, dtype=np.float64)
    if disabled_agent is not None:
        radii[disabled_agent] = -1.0
    return make_cbf_filter(radii)


def _recording_filter(
    radius: float,
    num_multi_agents: int,
):
    cbf_filter = _cbf_action_filter(radius, num_multi_agents)
    leaveout_filters = [
        _cbf_action_filter(radius, num_multi_agents, disabled_agent=agent)
        for agent in range(num_multi_agents)
    ]
    records = {
        "action_delta": [],
        "agent_action_delta": [],
        "actual_distance": [],
        "predicted_distance": [],
    }

    def apply(base_action, env_state, swarm_state):
        base = np.asarray(base_action, dtype=np.float32)
        result = cbf_filter(
            base_action=base, env_state=env_state, swarm_state=swarm_state
        )
        action_delta = float(np.linalg.norm(result - base))
        agent_action_delta = np.zeros(num_multi_agents, dtype=np.float64)
        for agent, leaveout_filter in enumerate(leaveout_filters):
            leaveout_action = leaveout_filter(
                base_action=base,
                env_state=env_state,
                swarm_state=swarm_state,
            )
            agent_action_delta[agent] = np.linalg.norm(
                result - leaveout_action
            )

        ego_position = np.asarray(env_state.envs[-1].dynamics.pos, dtype=np.float64)
        actual_positions = np.stack(
            [
                np.asarray(env_state.envs[index].dynamics.pos, dtype=np.float64)
                for index in range(num_multi_agents)
            ]
        )
        predicted_positions = np.asarray(
            swarm_state.positions[:num_multi_agents], dtype=np.float64
        )

        records["action_delta"].append(action_delta)
        records["agent_action_delta"].append(agent_action_delta)
        records["actual_distance"].append(
            np.linalg.norm(actual_positions - ego_position[None, :], axis=1)
        )
        records["predicted_distance"].append(
            np.linalg.norm(predicted_positions - ego_position[None, :], axis=1)
        )
        return result

    return apply, records


def _vector_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return np.nan
    cosine = float(np.clip((left @ right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _event_turn(
    trajectory: Dict[str, np.ndarray],
    ego_index: int,
    start: int,
    end: int,
    *,
    window: int,
    min_displacement: float,
) -> tuple[float, float]:
    points = _ego_state_points(trajectory, ego_index)
    event_start = start
    event_end = end + 1
    if event_start - window < 0 or event_end + window >= len(points):
        return np.nan, np.nan

    incoming = points[event_start] - points[event_start - window]
    outgoing = points[event_end + window] - points[event_end]
    if min(np.linalg.norm(incoming), np.linalg.norm(outgoing)) < min_displacement:
        return np.nan, np.nan

    turn_3d = _vector_angle_degrees(incoming, outgoing)
    if min(np.linalg.norm(incoming[:2]), np.linalg.norm(outgoing[:2])) < min_displacement:
        turn_horizontal = np.nan
    else:
        turn_horizontal = _vector_angle_degrees(incoming[:2], outgoing[:2])
    return turn_3d, turn_horizontal


def _active_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return []
    runs = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index - previous - 1 > max_gap:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def _build_agent_rows(
    scenario_seed: int,
    radius: float,
    filtered: Dict[str, np.ndarray],
    records: Dict[str, list],
    ego_initial_distances: np.ndarray,
    args,
) -> list[Dict[str, object]]:
    agent_action_deltas = np.asarray(
        records["agent_action_delta"], dtype=np.float64
    )
    actual_distances = np.asarray(records["actual_distance"], dtype=np.float64)
    predicted_distances = np.asarray(
        records["predicted_distance"], dtype=np.float64
    )
    rows = []

    for agent in range(args.num_multi_agents):
        marginal_delta = agent_action_deltas[:, agent]
        active = (
            np.isfinite(marginal_delta)
            & (marginal_delta > args.min_agent_action_delta)
        )
        runs = _active_runs(active, args.max_active_gap)
        evaluated = []
        for start, end in runs:
            active_frames = int(np.sum(active[start : end + 1]))
            turn, horizontal_turn = _event_turn(
                filtered,
                args.num_multi_agents,
                start,
                end,
                window=args.heading_window,
                min_displacement=args.min_heading_displacement,
            )
            actual_min = float(np.min(actual_distances[start : end + 1, agent]))
            predicted_min = float(
                np.min(predicted_distances[start : end + 1, agent])
            )
            direct = active_frames >= args.min_active_frames
            timed = (
                start >= args.min_event_start
                and end < len(active) - args.event_end_margin
            )
            close = actual_min <= radius + args.event_proximity_margin
            visible = np.isfinite(turn) and turn >= args.min_visible_turn_deg
            evaluated.append(
                {
                    "start": start,
                    "end": end,
                    "active_frames": active_frames,
                    "actual_min": actual_min,
                    "predicted_min": predicted_min,
                    "turn": float(turn),
                    "horizontal_turn": float(horizontal_turn),
                    "direct": direct,
                    "timed": timed,
                    "close": close,
                    "visible": visible,
                    "eligible": direct and timed and close and visible,
                }
            )

        if evaluated:
            event = max(
                evaluated,
                key=lambda item: (
                    int(item["eligible"]),
                    int(item["direct"]),
                    item["turn"] if np.isfinite(item["turn"]) else -1.0,
                    item["active_frames"],
                    -item["actual_min"],
                ),
            )
        else:
            event = {
                "start": -1,
                "end": -1,
                "active_frames": 0,
                "actual_min": np.nan,
                "predicted_min": np.nan,
                "turn": np.nan,
                "horizontal_turn": np.nan,
                "direct": False,
                "timed": False,
                "close": False,
                "visible": False,
                "eligible": False,
            }

        has_direct_run = any(item["direct"] for item in evaluated)
        if not has_direct_run:
            reason = "no_sustained_agent_action_effect"
        elif not event["timed"]:
            reason = "event_lacks_pre_or_post_context"
        elif not event["close"]:
            reason = "actual_agent_not_close_during_event"
        elif not event["visible"]:
            reason = "ego_turn_not_visible"
        else:
            reason = "pending_leaveout_check"

        active_values = marginal_delta[active]
        rows.append(
            {
                "scenario_seed": scenario_seed,
                "radius": radius,
                "agent": agent,
                "ego_initial_distance": float(ego_initial_distances[agent]),
                "actual_min_distance": float(np.min(actual_distances[:, agent])),
                "predicted_min_distance": float(
                    np.min(predicted_distances[:, agent])
                ),
                "direct_active_frames": int(event["active_frames"]),
                "direct_run_start": int(event["start"]),
                "direct_run_end": int(event["end"]),
                "marginal_action_delta_max": float(np.nanmax(marginal_delta)),
                "marginal_action_delta_mean_active": (
                    float(np.mean(active_values)) if active_values.size else 0.0
                ),
                "marginal_action_delta_integral": float(np.sum(active_values)),
                "actual_event_min_distance": float(event["actual_min"]),
                "predicted_event_min_distance": float(event["predicted_min"]),
                "visible_turn_deg": float(event["turn"]),
                "visible_horizontal_turn_deg": float(event["horizontal_turn"]),
                "direct_agent": int(has_direct_run),
                "visible_agent": int(event["eligible"]),
                "leaveout_checked": 0,
                "leaveout_completed": 0,
                "causal_heading_max_deg": 0.0,
                "causal_heading_frames": 0,
                "causal_deviation_max": 0.0,
                "causal_agent": 0,
                "other_response_max": 0.0,
                "other_response_mean": 0.0,
                "qualification_reason": reason,
            }
        )
    return rows


def _smoothed_tangents(points: np.ndarray, half_window: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    tangents = np.full(points.shape, np.nan, dtype=np.float64)
    if len(points) > 2 * half_window:
        tangents[half_window:-half_window] = (
            points[2 * half_window :] - points[: -2 * half_window]
        )
    return tangents


def _heading_differences(
    first_points: np.ndarray,
    second_points: np.ndarray,
    *,
    half_window: int,
    min_displacement: float,
) -> np.ndarray:
    common = min(len(first_points), len(second_points))
    first = _smoothed_tangents(first_points[:common], half_window)
    second = _smoothed_tangents(second_points[:common], half_window)
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    valid = (
        np.isfinite(first_norm)
        & np.isfinite(second_norm)
        & (first_norm >= min_displacement)
        & (second_norm >= min_displacement)
    )
    angles = np.full(common, np.nan, dtype=np.float64)
    if np.any(valid):
        cosine = np.sum(first[valid] * second[valid], axis=1) / (
            first_norm[valid] * second_norm[valid]
        )
        angles[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angles


def _causal_metrics(
    filtered: Dict[str, np.ndarray],
    leaveout: Dict[str, np.ndarray],
    agent_row: Dict[str, object],
    args,
) -> Dict[str, float | int]:
    filtered_points = _ego_state_points(filtered, args.num_multi_agents)
    leaveout_points = _ego_state_points(leaveout, args.num_multi_agents)
    common = min(len(filtered_points), len(leaveout_points))
    start = int(agent_row["direct_run_start"])
    stop = min(common, int(agent_row["direct_run_end"]) + 2 + args.causal_post_frames)
    angles = _heading_differences(
        filtered_points[:common],
        leaveout_points[:common],
        half_window=args.causal_heading_half_window,
        min_displacement=args.min_heading_displacement,
    )
    event_angles = angles[max(0, start) : stop]
    finite_angles = event_angles[np.isfinite(event_angles)]
    heading_max = float(np.max(finite_angles)) if finite_angles.size else 0.0
    heading_frames = int(
        np.sum(finite_angles >= args.min_causal_heading_deg)
    )
    deviations = np.linalg.norm(
        filtered_points[max(0, start) : stop]
        - leaveout_points[max(0, start) : stop],
        axis=1,
    )
    deviation_max = float(np.max(deviations)) if deviations.size else 0.0
    leaveout_completed = int(
        int(leaveout["trajectory_length"]) == args.episode_length
    )
    causal = int(
        leaveout_completed
        and heading_max >= args.min_causal_heading_deg
        and heading_frames >= args.min_causal_heading_frames
        and deviation_max >= args.min_causal_deviation
    )
    return {
        "leaveout_completed": leaveout_completed,
        "causal_heading_max_deg": heading_max,
        "causal_heading_frames": heading_frames,
        "causal_deviation_max": deviation_max,
        "causal_agent": causal,
    }


def _pair_metrics(
    agent_rows: Sequence[Dict[str, object]], min_event_gap: int
) -> Dict[str, int]:
    causal = [row for row in agent_rows if int(row["causal_agent"])]
    if len(causal) < 2:
        return {
            "best_event_gap_frames": -1,
            "has_sequential_pair": 0,
            "has_overlapping_pair": 0,
        }

    gaps = []
    overlapping = False
    for left, right in combinations(causal, 2):
        left_start = int(left["direct_run_start"])
        left_end = int(left["direct_run_end"])
        right_start = int(right["direct_run_start"])
        right_end = int(right["direct_run_end"])
        if left_end < right_start:
            gap = right_start - left_end - 1
        elif right_end < left_start:
            gap = left_start - right_end - 1
        else:
            overlap = min(left_end, right_end) - max(left_start, right_start) + 1
            gap = -overlap
            overlapping = True
        gaps.append(gap)

    best_gap = max(gaps)
    return {
        "best_event_gap_frames": int(best_gap),
        "has_sequential_pair": int(best_gap >= min_event_gap),
        "has_overlapping_pair": int(overlapping),
    }


def _radius_is_interesting(
    full_completed: int,
    agent_rows: Sequence[Dict[str, object]],
) -> int:
    return int(
        bool(full_completed)
        and sum(int(row["causal_agent"]) for row in agent_rows) >= 2
    )


def _radius_span_metrics(
    filtered_by_radius: Dict[float, Dict[str, np.ndarray]], args
) -> Dict[str, float]:
    if len(filtered_by_radius) < 2:
        return {
            "radius_span_position_max": 0.0,
            "radius_span_heading_max_deg": 0.0,
        }
    low = filtered_by_radius[min(filtered_by_radius)]
    high = filtered_by_radius[max(filtered_by_radius)]
    low_points = _ego_state_points(low, args.num_multi_agents)
    high_points = _ego_state_points(high, args.num_multi_agents)
    common = min(len(low_points), len(high_points))
    position_max = float(
        np.max(np.linalg.norm(low_points[:common] - high_points[:common], axis=1))
    )
    angles = _heading_differences(
        low_points[:common],
        high_points[:common],
        half_window=args.causal_heading_half_window,
        min_displacement=args.min_heading_displacement,
    )
    finite = angles[np.isfinite(angles)]
    heading_max = float(np.max(finite)) if finite.size else 0.0
    return {
        "radius_span_position_max": position_max,
        "radius_span_heading_max_deg": heading_max,
    }


def _radius_rank_key(row: Dict[str, object]):
    return (
        int(row["interesting"]),
        int(row["has_sequential_pair"]),
        int(row["num_causal_agents"]),
        float(row["second_causal_heading_deg"]),
        float(row["second_causal_deviation"]),
        int(row["num_visible_agents"]),
        int(row["num_direct_agents"]),
        -float(row["filtered_min_distance"]),
    )


def _rank_summaries(rows: Sequence[Dict[str, object]]):
    return sorted(
        rows,
        key=lambda row: (
            -int(row["interesting"]),
            -int(row["qualifying_radius_count"]),
            -int(row["has_sequential_pair"]),
            -int(row["num_visible_agents"]),
            -int(row["num_direct_agents"]),
            -float(row["second_causal_heading_deg"]),
            -float(row["second_causal_deviation"]),
            -float(row["radius_span_position_max"]),
            int(row["scenario_seed"]),
        ),
    )


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
    base_path = os.path.splitext(summary_path)[0]
    detail_path = _csv_path(args.detail_path or f"{base_path}_details")
    agent_detail_path = _csv_path(
        args.agent_detail_path or f"{base_path}_agents"
    )
    for path in (summary_path, detail_path, agent_detail_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

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
        ) as detail_file, open(
            agent_detail_path, "w", newline="", encoding="utf-8"
        ) as agent_file:
            summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
            detail_writer = csv.DictWriter(
                detail_file, fieldnames=RADIUS_DETAIL_FIELDS
            )
            agent_writer = csv.DictWriter(
                agent_file, fieldnames=AGENT_DETAIL_FIELDS
            )
            summary_writer.writeheader()
            detail_writer.writeheader()
            agent_writer.writeheader()

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
                initial = _initial_geometry(states, args.num_multi_agents)
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
                filtered_by_radius = {}
                for radius in args.radii:
                    action_fn, records = _recording_filter(
                        radius,
                        args.num_multi_agents,
                    )
                    filtered = _collect_paired_trajectory(
                        *rollout_args, action_fn, device
                    )
                    filtered_by_radius[radius] = filtered
                    metrics = _paired_metrics(
                        baseline,
                        filtered,
                        predictions,
                        records["action_delta"],
                        num_multi_agents=args.num_multi_agents,
                        proximity_threshold=args.proximity_threshold,
                        action_delta_threshold=args.action_delta_threshold,
                    )
                    agent_rows = _build_agent_rows(
                        scenario_seed,
                        radius,
                        filtered,
                        records,
                        initial["ego_initial_distances"],
                        args,
                    )

                    common = int(metrics["common_length"])
                    baseline_positions = np.asarray(baseline["positions"])[
                        :common, : args.num_multi_agents
                    ]
                    filtered_positions = np.asarray(filtered["positions"])[
                        :common, : args.num_multi_agents
                    ]
                    other_response = np.linalg.norm(
                        filtered_positions - baseline_positions, axis=2
                    )
                    for row in agent_rows:
                        agent = int(row["agent"])
                        row["other_response_max"] = float(
                            np.max(other_response[:, agent])
                        )
                        row["other_response_mean"] = float(
                            np.mean(other_response[:, agent])
                        )
                        if not int(row["visible_agent"]):
                            continue
                        leaveout_fn = _cbf_action_filter(
                            radius, args.num_multi_agents, disabled_agent=agent
                        )
                        leaveout = _collect_paired_trajectory(
                            *rollout_args, leaveout_fn, device
                        )
                        row["leaveout_checked"] = 1
                        row.update(
                            _causal_metrics(
                                filtered,
                                leaveout,
                                row,
                                args,
                            )
                        )
                        row["qualification_reason"] = (
                            "qualified_causal_agent"
                            if int(row["causal_agent"])
                            else "leaveout_effect_below_threshold"
                        )

                    causal_rows = [
                        row for row in agent_rows if int(row["causal_agent"])
                    ]
                    causal_headings = sorted(
                        (
                            float(row["causal_heading_max_deg"])
                            for row in causal_rows
                        ),
                        reverse=True,
                    )
                    causal_deviations = sorted(
                        (
                            float(row["causal_deviation_max"])
                            for row in causal_rows
                        ),
                        reverse=True,
                    )
                    pair_metrics = _pair_metrics(agent_rows, args.min_event_gap)
                    full_completed = int(
                        int(filtered["trajectory_length"]) == args.episode_length
                    )
                    interesting = _radius_is_interesting(
                        full_completed, agent_rows
                    )
                    detail = {
                        "scenario_seed": scenario_seed,
                        "radius": radius,
                        "ego_initial_min_distance": initial[
                            "ego_initial_min_distance"
                        ],
                        "ego_goal_distance": initial["ego_goal_distance"],
                        "nominal_min_distance": nominal["min_distance"],
                        "nominal_closest_step": nominal["closest_step"],
                        "nominal_closest_agent": nominal["closest_agent"],
                        "nominal_close_fraction": nominal["close_fraction"],
                        **metrics,
                        "full_completed": full_completed,
                        "num_direct_agents": sum(
                            int(row["direct_agent"]) for row in agent_rows
                        ),
                        "num_visible_agents": sum(
                            int(row["visible_agent"]) for row in agent_rows
                        ),
                        "num_causal_agents": len(causal_rows),
                        "causal_agent_ids": ";".join(
                            str(row["agent"]) for row in causal_rows
                        ),
                        "second_causal_heading_deg": (
                            causal_headings[1] if len(causal_headings) >= 2 else 0.0
                        ),
                        "second_causal_deviation": (
                            causal_deviations[1]
                            if len(causal_deviations) >= 2
                            else 0.0
                        ),
                        **pair_metrics,
                        "interesting": interesting,
                    }
                    details.append(detail)
                    detail_writer.writerow(detail)
                    for row in agent_rows:
                        agent_writer.writerow(row)
                    detail_file.flush()
                    agent_file.flush()

                span_metrics = _radius_span_metrics(filtered_by_radius, args)
                best = max(details, key=_radius_rank_key)
                qualifying = [row for row in details if int(row["interesting"])]
                summary = {
                    "scenario_seed": scenario_seed,
                    "interesting": int(bool(qualifying)),
                    "best_radius": best["radius"],
                    "qualifying_radius_count": len(qualifying),
                    "qualifying_radii": ";".join(
                        f"{float(row['radius']):g}" for row in qualifying
                    ),
                    **span_metrics,
                    **{
                        field: best[field]
                        for field in RADIUS_DETAIL_FIELDS[2:]
                        if field != "interesting"
                    },
                }
                summaries.append(summary)
                summary_writer.writerow(summary)
                summary_file.flush()
                print(
                    f"seed={scenario_seed} interesting={bool(summary['interesting'])} "
                    f"best_r={float(summary['best_radius']):.2f} "
                    f"direct={summary['num_direct_agents']} "
                    f"visible={summary['num_visible_agents']} "
                    f"causal={summary['num_causal_agents']} "
                    f"sequential={bool(summary['has_sequential_pair'])}"
                )
    finally:
        env.close()

    ranked = _rank_summaries(summaries)
    print(f"\nQualified seeds: {sum(row['interesting'] for row in ranked)}/{len(ranked)}")
    print("Top candidates:")
    for rank, row in enumerate(ranked[: args.top_k], 1):
        print(
            f"{rank:2d}. seed={row['scenario_seed']} "
            f"interesting={bool(row['interesting'])} r={float(row['best_radius']):.2f} "
            f"causal={row['num_causal_agents']} "
            f"second_heading={float(row['second_causal_heading_deg']):.1f}deg "
            f"sequential={bool(row['has_sequential_pair'])} "
            f"radius_span={float(row['radius_span_position_max']):.3f}m"
        )
    print(f"Saved summaries to {os.path.abspath(summary_path)}")
    print(f"Saved per-radius details to {os.path.abspath(detail_path)}")
    print(f"Saved per-agent details to {os.path.abspath(agent_detail_path)}")


if __name__ == "__main__":
    main()
