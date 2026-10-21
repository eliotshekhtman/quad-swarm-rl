#!/usr/bin/env python3
"""
Sweep CBF radii and report ego-to-environment-quad minimum distances.
"""

from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch

from sample_factory.model.model_utils import get_rnn_size

from swarm_rl.env_snapshot import safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import register_swarm_components

from collect_video import (
    DEVICE,
    _apply_rollout_seed,
    _build_eval_cfg,
    _collect_trajectory,
    _load_or_generate_pred_trajectories,
    _state_for_frame,
)
from project_utils.restart_utils import deterministic_reset
from project_utils.full_cbf_utils import make_cbf_filter
from project_utils.utils import latest_checkpoint, load_actor, load_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CBF radii and average actual ego-to-env-quad distances.")
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", required=True)
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--predictions_dir", default="train_dir")
    parser.add_argument("--init_predictions", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rollout_seed_start", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--radius_start", type=float, default=8.0)
    parser.add_argument("--radius_stop", type=float, default=0.5)
    parser.add_argument("--num_radii", type=int, default=10)
    parser.add_argument("--output_path", default="train_dir/radius_sweep_runs.csv")
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def _prediction_args(args: argparse.Namespace, rollout_seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        predictions_dir=args.predictions_dir,
        init_predictions=args.init_predictions,
        seed=args.seed,
        rollout_seed=rollout_seed,
        episode_length=args.episode_length,
        multi_experiment=args.multi_experiment,
    )


def _normalize_csv_path(path: str) -> str:
    return path if path.endswith(".csv") else f"{path}.csv"


def _default_summary_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}_summary{ext or '.csv'}"


def _min_ego_env_distance(trajectory, num_multi_agents: int):
    if num_multi_agents <= 0:
        raise ValueError("--num_multi_agents must be positive to measure distance to env quads.")

    min_distance = float("inf")
    min_frame = -1
    min_agent = -1
    max_len = int(trajectory["trajectory_length"])

    for frame_idx in range(max_len + 1):
        positions, _velocities, _rotations, _omegas, _goals = _state_for_frame(trajectory, frame_idx)
        ego_pos = positions[num_multi_agents]
        env_positions = positions[:num_multi_agents]
        distances = np.linalg.norm(env_positions - ego_pos[None, :], axis=1)
        agent_idx = int(np.argmin(distances))
        distance = float(distances[agent_idx])
        if distance < min_distance:
            min_distance = distance
            min_frame = int(frame_idx)
            min_agent = agent_idx

    return min_distance, min_frame, min_agent


def _load_pred_cache(
    args,
    env,
    stored_states,
    rollout_seeds,
    multi_actor,
    multi_rnn_size,
    multi_checkpoint,
):
    pred_cache = {}
    for rollout_seed in rollout_seeds:
        pred_args = _prediction_args(args, rollout_seed)
        pred_trajectories_list = _load_or_generate_pred_trajectories(
            pred_args,
            env,
            stored_states,
            args.num_multi_agents,
            multi_actor,
            torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE),
            multi_checkpoint,
        )
        if len(pred_trajectories_list) != args.num_multi_agents:
            raise ValueError(
                f"Predicted trajectory count ({len(pred_trajectories_list)}) does not match "
                f"--num_multi_agents ({args.num_multi_agents})."
            )
        pred_cache[int(rollout_seed)] = pred_trajectories_list
    return pred_cache


def main() -> None:
    args = parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num_runs must be positive.")
    if args.num_radii <= 0:
        raise ValueError("--num_radii must be positive.")
    if args.radius_start <= 0.0 or args.radius_stop <= 0.0:
        raise ValueError("--radius_start and --radius_stop must be positive.")

    torch.set_grad_enabled(False)
    register_swarm_components()

    output_path = _normalize_csv_path(args.output_path)
    summary_path = _normalize_csv_path(args.summary_path) if args.summary_path else _default_summary_path(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)

    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    if args.num_multi_agents < 0:
        args.num_multi_agents = int(cfg_multi.quads_num_agents)
    num_agents = int(args.num_multi_agents) + 1

    eval_cfg = _build_eval_cfg(cfg_multi, num_agents=num_agents, enable_render=False, view_mode="topdownfollow")
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)

    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    try:
        multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
        multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, DEVICE)
        multi_rnn_size = get_rnn_size(cfg_multi)

        solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
        solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
        solo_obs_dim = int(solo_env.observation_space.shape[0])
        solo_rnn_size = get_rnn_size(cfg_solo)
    finally:
        solo_env.close()

    rollout_seeds = [int(args.rollout_seed_start + idx) for idx in range(args.num_runs)]
    radii = np.linspace(float(args.radius_start), float(args.radius_stop), int(args.num_radii), dtype=np.float64)

    rows = []
    summary_rows = []
    try:
        _obs, stored_states = deterministic_reset(env, args.seed, None)
        pred_cache = _load_pred_cache(
            args,
            env,
            stored_states,
            rollout_seeds,
            multi_actor,
            multi_rnn_size,
            multi_ckpt,
        )

        for radius in radii:
            run_distances = []
            solo_action_fn = make_cbf_filter(
                np.full(args.num_multi_agents, float(radius), dtype=np.float64)
            )

            for rollout_seed in rollout_seeds:
                pred_trajectories_list = pred_cache[int(rollout_seed)]
                obs, _ = deterministic_reset(env, args.seed, stored_states)
                _apply_rollout_seed(env, rollout_seed)
                snapshot = safe_capture_env_snapshot(env)

                trajectory = _collect_trajectory(
                    snapshot,
                    obs,
                    num_multi_agents=args.num_multi_agents,
                    multi_actor=multi_actor,
                    multi_rnn_states=torch.zeros(
                        (args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE
                    ),
                    solo_actor=solo_actor,
                    solo_rnn_states=torch.zeros((1, solo_rnn_size), dtype=torch.float32, device=DEVICE),
                    solo_obs_dim=solo_obs_dim,
                    pred_trajectories=pred_trajectories_list,
                    solo_action_fn=solo_action_fn,
                    max_steps=args.episode_length,
                    deterministic=False,
                )
                min_distance, min_frame, min_agent = _min_ego_env_distance(
                    trajectory,
                    num_multi_agents=args.num_multi_agents,
                )
                run_distances.append(min_distance)
                rows.append(
                    {
                        "radius": float(radius),
                        "rollout_seed": int(rollout_seed),
                        "min_distance": float(min_distance),
                        "min_frame": int(min_frame),
                        "min_agent": int(min_agent),
                        "trajectory_length": int(trajectory["trajectory_length"]),
                    }
                )
                print(
                    f"radius={radius:.6f} rollout_seed={rollout_seed} "
                    f"min_distance={min_distance:.6f} min_frame={min_frame} min_agent={min_agent}"
                )

            run_distances_arr = np.asarray(run_distances, dtype=np.float64)
            summary = {
                "radius": float(radius),
                "num_runs": int(len(run_distances_arr)),
                "avg_min_distance": float(np.mean(run_distances_arr)),
                "std_min_distance": float(np.std(run_distances_arr)),
                "min_min_distance": float(np.min(run_distances_arr)),
                "max_min_distance": float(np.max(run_distances_arr)),
            }
            summary_rows.append(summary)
            print(
                f"SUMMARY radius={radius:.6f} avg_min_distance={summary['avg_min_distance']:.6f} "
                f"std={summary['std_min_distance']:.6f}"
            )
    finally:
        env.close()

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["radius", "rollout_seed", "min_distance", "min_frame", "min_agent", "trajectory_length"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "radius",
            "num_runs",
            "avg_min_distance",
            "std_min_distance",
            "min_min_distance",
            "max_min_distance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-run results to {os.path.abspath(output_path)}")
    print(f"Saved summary results to {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
