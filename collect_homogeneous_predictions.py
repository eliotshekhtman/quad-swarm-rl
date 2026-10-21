#!/usr/bin/env python3
"""Collect a fixed-goal prediction rollout using one policy for every quad."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from sample_factory.model.model_utils import get_rnn_size

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.prediction_utils import (
    EVALUATION_SCENARIO,
    collect_homogeneous_trajectories,
    save_prediction_artifact,
)
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import latest_checkpoint, load_actor, load_cfg


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect homogeneous go_to_goal predictions."
    )
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", default="multi_ddg_6")
    parser.add_argument("--train_dir", default="train_dir")
    parser.add_argument("--experiment_name", default="homog_gtg_pred_traj_data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--num_multi_agents", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)
    if args.episode_length <= 0:
        parser.error("--episode_length must be positive")
    if args.num_multi_agents <= 0:
        parser.error("--num_multi_agents must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu was requested, but CUDA is not available.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    output_dir = os.path.join(args.train_dir, args.experiment_name)
    prediction_path = os.path.join(output_dir, "pred_trajectories.npz")
    args_path = os.path.join(output_dir, "conformal_args.json")
    if os.path.exists(prediction_path) or os.path.exists(args_path):
        raise FileExistsError(
            f"Refusing to overwrite an existing prediction artifact in {output_dir}."
        )
    os.makedirs(output_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    register_swarm_components()
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    num_policy_agents = args.num_multi_agents + 1
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        f"--device={args.device}",
        f"--quads_mode={EVALUATION_SCENARIO}",
        f"--quads_num_agents={num_policy_agents}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_collision_reward=8.0",
        "--quads_collision_hitbox_radius=2.5",
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
        checkpoint = latest_checkpoint(
            args.multi_train_dir,
            args.multi_experiment,
            policy_index=0,
        )
        actor = load_actor(
            cfg_multi,
            env.observation_space,
            env.action_space,
            checkpoint,
            device,
        )
        obs, initial_states = deterministic_reset(env, args.seed, None)
        rnn_states = torch.zeros(
            (num_policy_agents, get_rnn_size(cfg_multi)),
            dtype=torch.float32,
            device=device,
        )
        all_trajectories = collect_homogeneous_trajectories(
            env,
            obs,
            actor,
            rnn_states,
            args.episode_length,
        )
        pred_trajectories = all_trajectories[: args.num_multi_agents].copy()
        save_prediction_artifact(
            prediction_path,
            pred_trajectories,
            all_trajectories,
            initial_states,
            policy_experiment=args.multi_experiment,
            policy_checkpoint=checkpoint,
            seed=args.seed,
        )
    finally:
        env.close()

    metadata = {
        **vars(args),
        "evaluation_scenario": EVALUATION_SCENARIO,
        "homogeneous": True,
        "num_policy_agents": num_policy_agents,
        "policy_checkpoint": os.path.basename(checkpoint),
    }
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"Saved homogeneous predictions to {prediction_path}")


if __name__ == "__main__":
    main()
