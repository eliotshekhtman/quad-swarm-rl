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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.huggingface.huggingface_utils import generate_replay_video

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_snapshot import *

from project_utils.conformal_utils import *
from project_utils.utils import *
from project_utils.cbf_utils import (
    make_cbf_filter, 
    CBF_K0, 
    CBF_K1,
)
from project_utils.restart_utils import (
    deterministic_reset,
    extract_positions_velocities,
)

DEVICE = torch.device("cpu")
DELTA_T = 0.015
MIN_RADIUS = 0
MAX_RADIUS = 8



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
    parser.add_argument("--solo_train_dir", default='train_dir', help="Directory containing the trained single-agent policy.")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", default='train_dir', help="Base directory to store the new conformal experiment.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--seed", type=int, default=1, help="Seed applied before every reset to reproduce goals.")
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--num_threads", type=int, default=1, help="Max worker threads for parallel rollouts (default: CPU count).")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    register_swarm_components()
    max_threads = args.num_threads if args.num_threads and args.num_threads > 0 else (os.cpu_count() or 1)


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

    # Load in multi-agents
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=render_mode)
    multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
    multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, DEVICE)
    multi_rnn_size = get_rnn_size(cfg_multi)
    multi_rnn_states = torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE)

    # Add in ego agent
    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    multi_rnn_states = torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE)
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)

    # Save initial state so we can return to it later
    obs, stored_states = deterministic_reset(env, args.seed, None)

    # Collect initial predicted trajectory
    print('Predicting trajectories')
    obs, stored_states = deterministic_reset(env, args.seed, stored_states)
    snapshot = safe_capture_env_snapshot(env)
    temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
    dummy_pred_traj = np.zeros((args.num_multi_agents, args.episode_length, 6))
    logs = run_multi_agents(temp_env, obs, args.num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories=dummy_pred_traj,
                    solo_action_fn=fall_down,
                    deterministic=True, max_steps=args.episode_length,
                    num_threads=max_threads)
    temp_env.close()
    pred_trajectories = []
    for agent_id in range(args.num_multi_agents):
        positions = logs[agent_id][0]["position"] # episode_length x 3
        velocities = logs[agent_id][0]["velocity"]
        pred_trajectories.append(np.concatenate([positions, velocities], axis=1))
    

    # Collect arm length for default radius and dt for time btn steps
    arm_len = env.quad_arm
    DELTA_T = env.control_dt
    MIN_RADIUS = arm_len * quads_collision_hitbox_radius # Internally used for detecting crashes
    KAPPA = 0.6 # Tune to desired

    ##### EPISODE 0 #####
    # Init r0 to some large value that ought to be safe
    radius = MAX_RADIUS
    radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
    filter = make_cbf_filter(radii)

    ### COLLECTING DATA FOR EPISODE 0 ###
    episode = 0
    # Do a full reset before collecting rollouts
    multi_rnn_states = torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE)
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
    obs, stored_states = deterministic_reset(env, args.seed, stored_states)
    snapshot = safe_capture_env_snapshot(env)
    temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
    # Collect actual rollouts to compare against (with current radius)
    logs = run_multi_agents(temp_env, obs, args.num_multi_agents, 
                multi_actor, multi_rnn_states, 
                solo_actor, solo_rnn_states, solo_obs_dim, 
                pred_trajectories, filter,
                max_steps=args.episode_length, 
                num_runs=1, 
                deterministic=args.deterministic,
                num_threads=max_threads)
    temp_env.close()

    ##### SET UP PRED_TRAJECTORIES FOR SAVING #####
    pred_trajectories = []
    for agent_id in range(args.num_multi_agents):
        positions = logs[agent_id][0]["position"] # episode_length x 3
        velocities = logs[agent_id][0]["velocity"]
        pred_trajectories.append(np.concatenate([positions, velocities], axis=1))

    pred_trajectories = np.asarray(pred_trajectories, dtype=np.float32)
    save_path = os.path.join(experiment_dir, "pred_trajectories.npz")
    np.savez(save_path, pred_trajectories=pred_trajectories)
    print(f"[conformal] Saved predicted trajectories to {save_path}")
    env.close()

    



if __name__ == "__main__":
    main()
