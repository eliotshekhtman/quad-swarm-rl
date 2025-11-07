#!/usr/bin/env python3
"""
Collect quadrotor trajectories in the patrol_dual_goal scenario with deterministic resets.

This script mirrors the evaluation flow of ``mixed_enjoy.py`` while enforcing:
    * A user-specified random seed that is re-applied before every environment reset so the
      patrol scenario regenerates the same goals each episode.
    * Manual restoration of the initial quad states (position, velocity, rotation, body rates)
      after every reset, ensuring agents respawn exactly where they started on the first episode.
    * Per-timestep logging of each quad's position and velocity to an ``NPZ`` archive.
    * Loading of a pretrained recurrent predictor (unused for control, but instantiated and ready).

Initial states detected on the first episode are persisted to ``initial_states.json`` inside the
output experiment directory so subsequent runs can reproduce the exact spawn configuration.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from tqdm import tqdm

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_snapshot import *

from pretrain_rnn_predictor import RNNPredictor, load_rnn_checkpoint

from utils import *
from cbf_utils import make_cbf_filter
from restart_utils import (
    QuadState,
    deterministic_reset,
    extract_positions_velocities,
    quad_state_from_dict,
    quad_state_to_serialisable,
)



# ---------------------------------------------------------------------------
# Conformal utilities
# ---------------------------------------------------------------------------

def fall_down(action_solo):
    return np.array([-1., -1., -1., -1.], dtype=np.float64)

def run_multi_agents(env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, solo_action_fn,
                    max_steps=1600, num_runs=1, deterministic=False):
    '''
    Run the environment for [max_steps] steps, where the multi agents act like normal
    but the solo agent plays a fixed action, and return positions and velocities.
    '''
    device = torch.device("cpu")
    snapshot = capture_env_snapshot(env)

    logs = {}
    for i in range(num_multi_agents):
        logs[i] = []
    for run in range(num_runs):
        # Start logging the new run
        for i in range(num_multi_agents):
            logs[i].append({ "position" : [], "velocity" : [] })
        # Make sure I spin up a new env and obs
        env_run = clone_env_from_snapshot(snapshot)
        obs_run = np.array(obs, copy=True, dtype=np.float32)
        done = False
        step_num = 0
        run_multi_rnn_states = multi_rnn_states.clone()
        run_solo_rnn_states = solo_rnn_states.clone()
        while not done and step_num < max_steps:
            obs_multi_dict = {OBS_KEY: obs_run[:num_multi_agents]}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_output = multi_actor(normalized_obs, run_multi_rnn_states)
            actions_multi = policy_output["actions"]
            run_multi_rnn_states = policy_output["new_rnn_states"]
            if deterministic:
                actions_multi = argmax_actions(multi_actor.action_distribution())
            if actions_multi.dim() == 1:
                actions_multi = actions_multi.unsqueeze(-1)
            actions_multi = actions_multi.detach().cpu().numpy()

            obs_solo_self = obs_run[-1, :solo_obs_dim]
            obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
            with torch.no_grad():
                normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
                policy_solo = solo_actor(normalized_solo, run_solo_rnn_states)
            run_solo_rnn_states = policy_solo["new_rnn_states"]
            # Running the actual execution as deterministic
            action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]
            action_solo = solo_action_fn(action_solo) # May need more params

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs_run, rewards, terminated, truncated, infos = env_run.step(actions)
            obs_run = np.array(obs_run, dtype=np.float32)

            pos, vel = extract_positions_velocities(env_run.unwrapped)
            for i in range(num_multi_agents):
                logs[i][-1]["position"].append(pos[i])
                logs[i][-1]["velocity"].append(vel[i])

            terminated = np.asarray(terminated)
            truncated = np.asarray(truncated)
            done = np.all(np.logical_or(terminated, truncated))
            step_num += 1
        env_run.close()

    return logs

def finetune_rnn(logs, num_multi_agents, predictor_checkpoint):
    device = torch.device("cpu")

    # Fine-tune it just a little for the actual path
    predictors = []
    for agent_id in range(num_multi_agents):
        predictor = load_rnn_checkpoint(predictor_checkpoint, device)
        deterministic_run = np.concatenate(
            [logs[agent_id][-1]["position"], logs[agent_id][-1]["velocity"]],
            axis=-1,
        ).astype(np.float32)

        predictor.train()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=5e-4)
        criterion = nn.MSELoss()
        print(f'Training agent {agent_id}')
        with torch.enable_grad():
            for _ in tqdm(range(3)):
                sequence_tensor = torch.from_numpy(deterministic_run).to(
                    device=device, dtype=torch.float32
                )
                inputs = sequence_tensor[:-1].unsqueeze(0)
                targets = sequence_tensor[1:].unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                preds = predictor(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                optimizer.step()
        predictor.eval()
        predictors.append(predictor)
    return predictors

def conformal_radii(logs, num_multi_agents, predictors, alpha):
    for agent_id in range(num_multi_agents):
        print('Collecting trajectory-level nonconformity scores')
        # Collect trajectory-level nonconformity scores
        scores = []
        progress = tqdm(logs[agent_id][:-1])
        for run_log in progress:
            score = 0  # Lowerbound on possible nonconformity score
            # Warm it up with the first 50 steps, and collect predictions for the rest
            run = np.concatenate(
                [run_log["position"], run_log["velocity"]],
                axis=-1,
            ).astype(np.float32)
            warmup = min(50, len(run))
            predictions = [run[idx].copy() for idx in range(warmup)]
            for i in range(warmup, len(run)):
                history_arr = np.asarray(predictions, dtype=np.float32)
                history = torch.from_numpy(history_arr).unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                pred = predictor(history)[0, -1].detach().cpu().numpy()
                predictions.append(pred.astype(np.float32))
                # L2 norm difference between prediction and what really happened
                score = max(score, np.linalg.norm(pred - run[i]))
            scores.append(score)
        scores.sort() # Make sure the index isn't equal to len
        # Just want to visually check that this makes sense
        print(f'Scores for agent {agent_id}: ', scores)
        conformal_radius = scores[np.ceil(len(scores) * (1 - alpha)) - 1]


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic patrol_dual_goal data collection.")
    parser.add_argument("--multi_train_dir", required=True, help="Directory containing the trained multi-agent policy.")
    parser.add_argument("--multi_experiment", required=True, help="Experiment name for the multi-agent policy.")
    parser.add_argument("--solo_train_dir", required=True, help="Directory containing the trained single-agent policy.")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", required=True, help="Base directory to store the new conformal experiment.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--predictor_checkpoint", required=True, help="Path to a pretrained RNN predictor checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Seed applied before every reset to reproduce goals.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Desired probability of conformal error")
    parser.add_argument("--video_name", default="conformal_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=10)
    parser.add_argument("--num_trajectories", type=int, default=20)
    return parser.parse_args()


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_initial_states(path: str, states: List[QuadState]) -> None:
    payload = [quad_state_to_serialisable(s) for s in states]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_initial_states(path: str) -> Optional[List[QuadState]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [quad_state_from_dict(entry) for entry in payload]


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    register_swarm_components()

    if os.path.isabs(args.video_name):
        video_dir = os.path.dirname(args.video_name) or "."
        video_file = os.path.basename(args.video_name)
    else:
        video_dir = os.path.join(args.multi_train_dir, args.multi_experiment)
        video_file = args.video_name
    os.makedirs(video_dir, exist_ok=True)
    video_frames: List[np.ndarray] = []

    experiment_dir = ensure_experiment_dir(args.train_dir, args.experiment_name)

    device = torch.device("cpu")

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={cfg_multi.quads_num_agents + 1}",
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
    render_mode = "rgb_array"
    
    num_multi_agents = int(eval_cfg.quads_num_agents)

    # Load in multi-agents
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=render_mode)
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
    multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, device)
    multi_rnn_size = get_rnn_size(cfg_multi)
    multi_rnn_states = torch.zeros((num_multi_agents, multi_rnn_size), dtype=torch.float32, device=device)

    # Add in ego agent
    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, device)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=device)

    # Save initial state so we can return to it later
    obs, stored_states = deterministic_reset(env, args.seed, None)

    # Finetune a predictor for each multi-agent
    shapshot = capture_env_snapshot(env)
    temp_env = clone_env_from_snapshot(shapshot)
    logs = run_multi_agents(temp_env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, fall_down,
                    deterministic=True)
    predictors = finetune_rnn(logs, num_multi_agents, args.predictor_checkpoint)
    temp_env.close()

    # Make sure no resets are needed for the actual run
    num_episodes = 1500 // args.episode_length - 1
    progress_bar = tqdm(range(num_episodes))
    radii = np.full(num_multi_agents, 0, dtype=np.float64)
    # Collect histories for each agent to pass to their respective predictors
    for episode in progress_bar:
        progress_bar.set_postfix_str("Setting radii")
        filter = make_cbf_filter(radii)
        # Set conformal radii
        snapshot = capture_env_snapshot(env)
        temp_env = clone_env_from_snapshot(shapshot)
        logs = run_multi_agents(temp_env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, filter,
                    max_steps=args.episode_length, 
                    num_runs=args.num_trajectories, 
                    deterministic=False)
        # conformal radii and actual prediction traj's
        # solver stuff for other radii
        temp_env.close()
        for step in range(args.episode_length):
            # Actually run the normal execution
            obs_np = np.asarray(obs)

            obs_multi_dict = {OBS_KEY: obs_np[:num_multi_agents]}
            with torch.no_grad():
                normalized_multi = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_multi = multi_actor(normalized_multi, multi_rnn_states)
            actions_multi = policy_multi["actions"]
            multi_rnn_states = policy_multi["new_rnn_states"]
            # Running the actual execution as deterministic
            actions_multi = argmax_actions(multi_actor.action_distribution())
            if actions_multi.dim() == 1:
                actions_multi = actions_multi.unsqueeze(-1)
            actions_multi = actions_multi.detach().cpu().numpy()

            obs_solo_self = obs_np[-1, :solo_obs_dim]
            obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
            with torch.no_grad():
                normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
                policy_solo = solo_actor(normalized_solo, solo_rnn_states)
            action_solo = policy_solo["actions"]
            solo_rnn_states = policy_solo["new_rnn_states"]
            # Running the actual execution as deterministic
            action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]

            swarm_state = get_swarm_state(env.unwrapped)
            # Apply CBF to the ego agent based on radius determined earlier
            action_solo = filter (
                base_action=action_solo,
                env_state=env.unwrapped,
                swarm_state=swarm_state
            )

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs, rewards, terminated, truncated, infos = env.step(actions)

            solo_info_rewards = infos[-1].get("rewards", {})
            if solo_info_rewards.get("rewraw_quadcol", 0.0) < 0.0:
                solo_collision_count += 1
                progress_bar.set_postfix_str(f"crashes={solo_collision_count}")

            terminated = np.asarray(terminated)
            truncated = np.asarray(truncated)
            dones = np.logical_or(terminated, truncated)

            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    env.close()

    if len(video_frames) > 0:
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_dir, video_file))
        print(f"[conformal_enjoy] Video saved to {final_path}")

    print(f"[conformal_enjoy] Solo drone collisions: {solo_collision_count}")



if __name__ == "__main__":
    main()
