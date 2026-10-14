#!/usr/bin/env python3
"""
Sweep rollout seeds for a fixed scenario seed and report ego-vs-prediction CBF margins.
"""

from __future__ import annotations

import argparse
import csv
import importlib
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
    _prediction_step_for_frame,
    _state_for_frame,
)
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import latest_checkpoint, load_actor, load_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find rollout seeds where ego stays outside all predicted CBF balls.")
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", required=True)
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--predictions_dir", default="train_dir")
    parser.add_argument("--init_predictions", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rollout_seed_start", type=int, default=0)
    parser.add_argument("--rollout_seed_stop", type=int, required=True)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--ecbf", action="store_true", help="Use project_utils.full_cbf_utils instead of project_utils.cbf_utils.")
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output_path", default="train_dir/rollout_seed_sweep.csv")
    parser.add_argument("--continue_after_found", action="store_true")
    return parser.parse_args()


def load_cbf_module(use_ecbf: bool):
    module_name = "project_utils.full_cbf_utils" if use_ecbf else "project_utils.cbf_utils"
    return importlib.import_module(module_name)


def _min_ego_prediction_margin(trajectory, pred_trajectories: np.ndarray, radius: float, num_multi_agents: int):
    min_margin = float("inf")
    min_frame = -1
    min_agent = -1
    max_len = int(trajectory["trajectory_length"])

    for frame_idx in range(max_len + 1):
        positions, _velocities, _rotations, _omegas, _goals = _state_for_frame(trajectory, frame_idx)
        ego_pos = positions[num_multi_agents]
        pred_step = _prediction_step_for_frame(frame_idx, pred_trajectories.shape[1])
        predicted_positions = pred_trajectories[:, pred_step, :3]
        margins = np.linalg.norm(predicted_positions - ego_pos[None, :], axis=1) - float(radius)
        agent_idx = int(np.argmin(margins))
        margin = float(margins[agent_idx])
        if margin < min_margin:
            min_margin = margin
            min_frame = int(frame_idx)
            min_agent = agent_idx

    return min_margin, min_frame, min_agent


def _prediction_args(args: argparse.Namespace, rollout_seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        predictions_dir=args.predictions_dir,
        init_predictions=args.init_predictions,
        seed=args.seed,
        rollout_seed=rollout_seed,
        episode_length=args.episode_length,
    )


def _normalize_csv_path(path: str) -> str:
    return path if path.endswith(".csv") else f"{path}.csv"


def main() -> None:
    args = parse_args()
    if args.r <= 0.0:
        raise ValueError("--r must be positive.")
    if args.rollout_seed_stop <= args.rollout_seed_start:
        raise ValueError("--rollout_seed_stop must be greater than --rollout_seed_start.")

    torch.set_grad_enabled(False)
    register_swarm_components()

    output_path = _normalize_csv_path(args.output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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

    cbf_module = load_cbf_module(args.ecbf)
    radii = np.full(args.num_multi_agents, float(args.r), dtype=np.float64)
    solo_action_fn = cbf_module.make_cbf_filter(radii)

    rows = []
    found_seed = None
    try:
        _obs, stored_states = deterministic_reset(env, args.seed, None)

        for rollout_seed in range(args.rollout_seed_start, args.rollout_seed_stop):
            pred_args = _prediction_args(args, rollout_seed)
            multi_rnn_states = torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE)
            solo_rnn_states = torch.zeros((1, solo_rnn_size), dtype=torch.float32, device=DEVICE)
            pred_trajectories_list = _load_or_generate_pred_trajectories(
                pred_args,
                env,
                stored_states,
                args.num_multi_agents,
                multi_actor,
                multi_rnn_states,
                solo_actor,
                solo_rnn_states,
                solo_obs_dim,
            )
            if len(pred_trajectories_list) != args.num_multi_agents:
                raise ValueError(
                    f"Predicted trajectory count ({len(pred_trajectories_list)}) does not match "
                    f"--num_multi_agents ({args.num_multi_agents})."
                )
            pred_trajectories = np.stack(pred_trajectories_list, axis=0).astype(np.float64)

            obs, _ = deterministic_reset(env, args.seed, stored_states)
            _apply_rollout_seed(env, rollout_seed)
            snapshot = safe_capture_env_snapshot(env)

            trajectory = _collect_trajectory(
                snapshot,
                obs,
                num_multi_agents=args.num_multi_agents,
                multi_actor=multi_actor,
                multi_rnn_states=torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE),
                solo_actor=solo_actor,
                solo_rnn_states=torch.zeros((1, solo_rnn_size), dtype=torch.float32, device=DEVICE),
                solo_obs_dim=solo_obs_dim,
                pred_trajectories=pred_trajectories_list,
                solo_action_fn=solo_action_fn,
                max_steps=args.episode_length,
                deterministic=False,
            )
            margin, frame_idx, agent_idx = _min_ego_prediction_margin(
                trajectory,
                pred_trajectories,
                radius=float(args.r),
                num_multi_agents=args.num_multi_agents,
            )
            passed = bool(margin > float(args.tolerance))
            row = {
                "rollout_seed": int(rollout_seed),
                "passed": int(passed),
                "min_margin": float(margin),
                "min_frame": int(frame_idx),
                "min_agent": int(agent_idx),
                "trajectory_length": int(trajectory["trajectory_length"]),
            }
            rows.append(row)
            print(
                f"rollout_seed={rollout_seed} passed={passed} "
                f"min_margin={margin:.6f} min_frame={frame_idx} min_agent={agent_idx}"
            )
            if passed and found_seed is None:
                found_seed = int(rollout_seed)
                print(f"SUCCESS: rollout_seed={found_seed}")
                if not args.continue_after_found:
                    break
    finally:
        env.close()

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["rollout_seed", "passed", "min_margin", "min_frame", "min_agent", "trajectory_length"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if found_seed is None:
        print(f"No passing rollout_seed found in [{args.rollout_seed_start}, {args.rollout_seed_stop}).")
    else:
        print(f"First passing rollout_seed: {found_seed}")
    print(f"Saved sweep results to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
