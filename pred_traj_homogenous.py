#!/usr/bin/env python3
"""
Open-loop conformal

Supercharged predictor: true collected trajectories from a run without interaction.
    Because of intense variation, predictions re-collected from last run's interaction.
Collect trajectories for qj using last episode's ego agent policy/radius.
Adjust new radius, but use a kappa that we set.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List
import numpy as np
import torch

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size
from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_snapshot import *

from project_utils.utils import *
from project_utils.restart_utils import (
    deterministic_reset,
    extract_positions_velocities,
)

DEVICE = torch.device("cpu")



# ---------------------------------------------------------------------------
# Conformal utilities
# ---------------------------------------------------------------------------

def run_multi_agents(env, obs, num_multi_agents, actor, rnn_states, max_steps=1500):
    """
    Run a single deterministic episode using the same policy for all agents.
    Returns positions and velocities for each agent (including the former solo).
    """
    snapshot = safe_capture_env_snapshot(env)
    env_run = clone_env_from_snapshot(snapshot)
    obs_run = np.array(obs, copy=True, dtype=np.float32)
    done = False
    step_num = 0
    run_rnn_states = rnn_states.clone()
    logs = {agent_id: {"position": [], "velocity": []} for agent_id in range(num_multi_agents + 1)}

    while not done and step_num < max_steps:
        obs_dict = {OBS_KEY: obs_run}
        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(actor, obs_dict)
            policy_output = actor(normalized_obs, run_rnn_states)
        run_rnn_states = policy_output["new_rnn_states"]
        actions = argmax_actions(actor.action_distribution())
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        actions = actions.detach().cpu().numpy()

        obs_run, rewards, dones, infos = env_run.step(actions)
        obs_run = np.array(obs_run, dtype=np.float32)

        pos, vel = extract_positions_velocities(env_run.unwrapped)
        for agent_id in range(num_multi_agents + 1):
            logs[agent_id]["position"].append(pos[agent_id])
            logs[agent_id]["velocity"].append(vel[agent_id])
        done = np.all(dones)
        step_num += 1

    env_run.close()
    return logs

# record distance to goal position / reward / max reward/mindist over episodes?
# what happens if you don't update, if you do aggressive adversarial CP: want to highlight that all the steps are important
# ablations ^^ 
# full explanation of what each step does currently, what's my plan for next teps, what's their feedback

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic patrol_dual_goal data collection.")
    parser.add_argument("--multi_train_dir", default='train_dir', help="Directory containing the trained multi-agent policy.")
    parser.add_argument("--multi_experiment", required=True, help="Experiment name for the multi-agent policy.")
    parser.add_argument("--train_dir", default='train_dir', help="Base directory to store the new conformal experiment.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed applied before every reset to reproduce goals.")
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    return parser.parse_args()


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    register_swarm_components()

    experiment_dir = ensure_experiment_dir(args.train_dir, args.experiment_name)
    args_path = os.path.join(experiment_dir, "conformal_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Load multi config early since it has some useful info
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    if args.num_multi_agents < 0:
        args.num_multi_agents = cfg_multi.quads_num_agents

    quads_collision_hitbox_radius = 2.5
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={args.num_multi_agents + 1}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_collision_reward=8.0",
        f"--quads_collision_hitbox_radius={quads_collision_hitbox_radius}",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=True",
        "--quads_view_mode=topdown",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    render_mode = "rgb_array"

    # Load in multi-agents (used for all agents, including former solo)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=render_mode)
    multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
    multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, DEVICE)
    multi_rnn_size = get_rnn_size(cfg_multi)
    multi_rnn_states = torch.zeros((args.num_multi_agents + 1, multi_rnn_size), dtype=torch.float32, device=DEVICE)

    # Save initial state so we can return to it later
    obs, stored_states = deterministic_reset(env, args.seed, None)

    # Collect predicted trajectory for a single deterministic episode
    print('Collecting trajectories')
    obs, stored_states = deterministic_reset(env, args.seed, stored_states)
    snapshot = safe_capture_env_snapshot(env)
    temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
    logs = run_multi_agents(temp_env, obs, args.num_multi_agents, multi_actor, multi_rnn_states, max_steps=args.episode_length)
    temp_env.close()

    pred_trajectories = []
    for agent_id in range(args.num_multi_agents):  # exclude former solo agent
        positions = np.asarray(logs[agent_id]["position"], dtype=np.float32)  # episode_length x 3
        velocities = np.asarray(logs[agent_id]["velocity"], dtype=np.float32)
        pred_trajectories.append(np.concatenate([positions, velocities], axis=1))

    pred_trajectories = np.asarray(pred_trajectories, dtype=np.float32)
    save_path = os.path.join(experiment_dir, "pred_trajectories.npz")
    np.savez(save_path, pred_trajectories=pred_trajectories)
    print(f"[conformal] Saved predicted trajectories to {save_path}")

    # Save minimal conformal_metrics.npz compatible with plot_no_interaction.py
    agent_locs = [np.asarray(logs[agent_id]["position"], dtype=np.float32) for agent_id in range(args.num_multi_agents + 1)]
    agent_locs_per_episode = np.asarray([agent_locs], dtype=np.float32)  # 1 x (num_agents) x steps x 3
    predicted_traj_per_episode = np.asarray([pred_trajectories], dtype=np.float32)  # 1 x (num_multi_agents) x steps x 6

    episodes = np.arange(1, dtype=np.int32)
    zeros1 = np.zeros_like(episodes, dtype=np.float32)
    metrics_path = os.path.join(experiment_dir, "conformal_metrics.npz")
    np.savez(
        metrics_path,
        episodes=episodes,
        qj_per_episode=zeros1,
        radius_per_episode=zeros1,
        tube_coverage_per_episode=zeros1,
        crashes_per_episode=zeros1,
        bad_crashes_per_episode=zeros1,
        safety_per_episode=zeros1,
        cumulative_reward_per_episode=zeros1,
        cumulative_reward_per_run=np.zeros((1, 0), dtype=np.float32),
        agent_locs_per_episode=agent_locs_per_episode,
        predicted_traj_per_episode=predicted_traj_per_episode,
        alpha=0.0,
        bar_alpha=0.0,
    )
    print(f"[conformal] Saved minimal metrics (for plotting) to {metrics_path}")

    env.close()

    



if __name__ == "__main__":
    main()
