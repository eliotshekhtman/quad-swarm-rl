#!/usr/bin/env python3
"""
Closed-loop episodes
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
from sample_factory.huggingface.huggingface_utils import generate_replay_video

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_snapshot import *

from pretrain_rnn_predictor import RNNPredictor, load_rnn_checkpoint

from utils import *
from cbf_utils import (
    make_cbf_filter, 
    CBF_K0, 
    CBF_K1,
)
from restart_utils import (
    QuadState,
    deterministic_reset,
    extract_positions_velocities,
    quad_state_from_dict,
    quad_state_to_serialisable,
)
from lipschitz_utils import (
    calculate_beta_T,
    closed_form_estimate_LXx,
    estimate_LU,
    estimate_LXu,
    estimate_LXx,
    estimate_LYx,
    estimate_LYy,
)

DEVICE = torch.device("cpu")
DELTA_T = 0.015
MIN_RADIUS = 0
MAX_RADIUS = 2




# ---------------------------------------------------------------------------
# Conformal utilities
# ---------------------------------------------------------------------------

def fall_down(base_action, env_state, swarm_state):
    return np.array([-1., -1., -1., -1.], dtype=np.float64)

def run_multi_agents(env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories, solo_action_fn,
                    max_steps=1600, num_runs=1, deterministic=False):
    '''
    Run the environment for [max_steps] steps, where the multi agents act like normal
    but the solo agent plays a fixed action, and return positions and velocities.
    Does not log the initial state.
    '''
    snapshot = safe_capture_env_snapshot(env)

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
            swarm_state = get_swarm_state(env_run.unwrapped)
            # We care about where we think they'll be next timestep: that's what
            # the conformal radius is built on
            for agent_id in range(num_multi_agents):
                swarm_state.positions[agent_id, :] = pred_trajectories[agent_id][step_num][:3]
                swarm_state.velocities[agent_id, :] = pred_trajectories[agent_id][step_num][3:]
            # Apply CBF to the ego agent based on radius determined earlier
            action_solo = solo_action_fn(
                base_action=action_solo,
                env_state=env_run.unwrapped,
                swarm_state=swarm_state
            )

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs_run, rewards, dones, infos = env_run.step(actions)
            obs_run = np.array(obs_run, dtype=np.float32)

            pos, vel = extract_positions_velocities(env_run.unwrapped)
            for i in range(num_multi_agents):
                logs[i][-1]["position"].append(pos[i])
                logs[i][-1]["velocity"].append(vel[i])
            done = np.all(dones)
            step_num += 1
        env_run.close()

    return logs



def finetune_rnn(logs, num_multi_agents, predictor_checkpoint):

    # Fine-tune it just a little for the actual path
    predictors = []
    for agent_id in range(num_multi_agents):
        predictor = load_rnn_checkpoint(predictor_checkpoint, DEVICE)
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
                    device=DEVICE, dtype=torch.float32
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

def roll_out_predictor(histories, predictors, agent_id, rollout_length):
    history_log = histories[agent_id]
    # Original true history array
    history_array = np.concatenate(
        [history_log["position"], history_log["velocity"]], axis=-1
    ).astype(np.float32)
    # Track history as a Python list so we can append predictions
    history = [entry.copy() for entry in history_array]
    history_len = len(history)
    # Roll out a predicted trajectory
    for i in range(rollout_length):
        # Predictor wasn't doing that well so just replaced with predicting from velocity
        # New history array generated from appended-to history
        # history_np = np.asarray(history, dtype=np.float32)
        # history_tensor = torch.from_numpy(history_np).unsqueeze(0).to(
        #     device = DEVICE, dtype = torch.float32)
        # pred = predictor(history_tensor)[0, -1].detach().cpu().numpy()
        pred = np.concatenate([
            history[-1][:3] + history[-1][3:] * DELTA_T, history[-1][3:]], 
            axis=-1)
        history.append(pred.astype(np.float32))
    return history[history_len:]

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
    parser.add_argument("--alpha", type=float, default=0.1, help="Desired probability of conformal error")
    parser.add_argument("--delta", type=float, default=0.1, help="Desired probability of a bad draw")
    parser.add_argument("--video_name", default="conformal_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=10)
    parser.add_argument("--num_trajectories", type=int, default=200)
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

    if os.path.isabs(args.video_name):
        video_dir = os.path.dirname(args.video_name) or "."
        video_file = os.path.basename(args.video_name)
    else:
        video_dir = os.path.join(args.multi_train_dir, args.multi_experiment)
        video_file = args.video_name
    os.makedirs(video_dir, exist_ok=True)
    video_frames: List[np.ndarray] = []

    experiment_dir = ensure_experiment_dir(args.train_dir, args.experiment_name)

    # Load multi config early since it has some useful info
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    if args.num_multi_agents < 0:
        args.num_multi_agents = cfg_multi.quads_num_agents

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={args.num_multi_agents + 1}",
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
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)

    # Save initial state so we can return to it later
    obs, stored_states = deterministic_reset(env, args.seed, None)

    # Finetune a predictor for each multi-agent
    # snapshot = safe_capture_env_snapshot(env)
    # temp_env = clone_env_from_snapshot(snapshot)
    # logs = run_multi_agents(temp_env, obs, args.num_multi_agents, 
    #                 multi_actor, multi_rnn_states, 
    #                 solo_actor, solo_rnn_states, solo_obs_dim, fall_down,
    #                 deterministic=True)
    # predictors = finetune_rnn(logs, args.num_multi_agents, args.predictor_checkpoint)
    # temp_env.close()
    predictors = [None] * args.num_multi_agents

    # Collect arm length for default radius and dt for time btn steps
    arm_len = env.quad_arm
    DELTA_T = env.control_dt
    MIN_RADIUS = arm_len
    bar_alpha = get_alpha_bar(args.alpha, args.delta, args.num_trajectories)

    # Make sure no resets are needed for the actual run
    num_episodes = 1500 // args.episode_length - 1
    progress_bar = tqdm(range(num_episodes))
    # Init r0 to some large value that ought to be safe
    radius = 2
    radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
    filter = make_cbf_filter(radii)
    # Collect histories for each agent to pass to their respective predictors
    pos, vel = extract_positions_velocities(env.unwrapped)
    histories = [] # list of pos,vel histories, each entry is a quad
    solo_collision_count = 0
    for i in range(args.num_multi_agents + 1):
        histories.append({ "position" : [pos[i]], "velocity" : [vel[i]] })
    for episode in progress_bar:
        # progress_bar.set_postfix_str("Setting radii")
        # Find qj using old pi_j
        snapshot = safe_capture_env_snapshot(env)
        temp_env = clone_env_from_snapshot(snapshot)
        # Collect predictions of where we think they'll go using our naive predictor
        pred_trajectories = [roll_out_predictor(histories, predictors, agent_id, args.episode_length) for agent_id in range(args.num_multi_agents)]
        # Collect actual rollouts to compare against
        logs = run_multi_agents(temp_env, obs, args.num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories, filter,
                    max_steps=args.episode_length, 
                    num_runs=args.num_trajectories, 
                    deterministic=False)
        # Set radius depending on how bad our prediction was
        perturbation_radius = 0.1
        # Empirically, joint qj seems to be around 2x the largest old qj
        # old_qj = conformal_radii(logs, args.num_multi_agents, pred_trajectories, bar_alpha, args.episode_length)
        qj = joint_conformal_radii(logs, args.num_multi_agents, pred_trajectories, bar_alpha, args.episode_length, args.num_trajectories)
        L_U = estimate_LU(temp_env, args.num_multi_agents)
        L_Xx_cf = closed_form_estimate_LXx(temp_env)
        L_Xx = estimate_LXx(temp_env, obs, args.num_multi_agents,
            solo_actor, solo_rnn_states, solo_obs_dim, perturbation_radius,)
        L_Xu = estimate_LXu(temp_env, obs, args.num_multi_agents, 
            solo_actor, solo_rnn_states, solo_obs_dim)
        L_Yx = estimate_LYx(temp_env, obs, args.num_multi_agents, 
            multi_actor, multi_rnn_states, 
            solo_actor, solo_rnn_states, solo_obs_dim, perturbation_radius)
        L_Yy = estimate_LYy(temp_env, obs, args.num_multi_agents, 
            multi_actor, multi_rnn_states,
            solo_actor, solo_rnn_states, solo_obs_dim, perturbation_radius)
        # L_Yu is zero, definitively
        # L_Yu = estimate_LYu(temp_env, obs, args.num_multi_agents, 
        #     multi_actor, multi_rnn_states,
        #     solo_actor, solo_rnn_states, solo_obs_dim, perturbation_radius)
        beta_T = calculate_beta_T(L_Xx, L_Xu, L_Yy, L_Yx, 0, args.episode_length)
        print('L_Xx', L_Xx, 'L_Xu', L_Xu, 'closed form L_Xx', L_Xx_cf)
        print('L_Yy', L_Yy, 'L_Yx', L_Yx)
        print('L_U', L_U, 'beta_T', beta_T)

        # Just because it gets way too big right now
        beta_threshold = 100_000_000
        if not np.isfinite(beta_T) or beta_T > beta_threshold:
            beta_T = beta_threshold
        radius = explicit_radius_update(radius, qj, L_U * beta_T)
        radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
        print('radius', radius, 'qj', qj) # , 'old qj', np.max(old_qj)
        filter = make_cbf_filter(radii) # pi_{j+1}
        temp_env.close()
        # progress_bar.set_postfix_str(f"crashes={solo_collision_count}")
        for step in range(args.episode_length):
            # Actually run the normal execution
            obs_np = np.asarray(obs)

            obs_multi_dict = {OBS_KEY: obs_np[:args.num_multi_agents]}
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
            # We care about where we think they'll be next timestep: that's what
            # the conformal radius is built on
            for agent_id in range(args.num_multi_agents):
                swarm_state.positions[agent_id, :] = pred_trajectories[agent_id][step][:3]
                swarm_state.velocities[agent_id, :] = pred_trajectories[agent_id][step][3:]
            # Apply CBF to the ego agent based on radius determined earlier
            action_solo = filter(
                base_action=action_solo,
                env_state=env.unwrapped,
                swarm_state=swarm_state
            )
            closest_dist = 100 # arbitrarily big
            solo_pos = swarm_state.positions[-1]
            for teammate_pos in swarm_state.positions[:-1]:
                dist = np.linalg.norm(solo_pos - teammate_pos)
                closest_dist = min(dist, closest_dist)
            progress_bar.set_postfix_str(f"dist>={closest_dist:.2f}")

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs, rewards, terminated, truncated, infos = env.step(actions)

            solo_info_rewards = infos[-1].get("rewards", {})
            if solo_info_rewards.get("rewraw_quadcol", 0.0) < 0.0:
                solo_collision_count += 1
                progress_bar.set_postfix_str(f"crashes={solo_collision_count}")
            
            pos, vel = extract_positions_velocities(env.unwrapped)
            for i in range(args.num_multi_agents + 1): # Save for all quads
                histories[i]["position"].append(pos[i])
                histories[i]["velocity"].append(vel[i])

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
