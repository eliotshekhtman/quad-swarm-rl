#!/usr/bin/env python3
"""
Actually run this open-loop
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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

from conformal_utils import *
from utils import *
from cbf_utils import (
    make_cbf_filter, 
    CBF_K0, 
    CBF_K1,
)
from restart_utils import (
    deterministic_reset,
    extract_positions_velocities,
)

DEVICE = torch.device("cpu")
DELTA_T = 0.015
MIN_RADIUS = 0
MAX_RADIUS = 5



# ---------------------------------------------------------------------------
# Conformal utilities
# ---------------------------------------------------------------------------

def fall_down(base_action, env_state, swarm_state):
    return np.array([-1., -1., -1., -1.], dtype=np.float64)

def run_multi_agents(env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories, solo_action_fn,
                    max_steps=1500, num_runs=1, deterministic=False):
    '''
    Run the environment for [max_steps] steps, where the multi agents act like normal
    but the solo agent plays a fixed action, and return positions and velocities.
    Does not log the initial state.
    '''
    snapshot = safe_capture_env_snapshot(env)
    # rng_backup = snapshot_rng_state() # restore_rng_state(rng_backup)

    # logs: num_multi_agents x num_runs x [pos or vel] x max_steps x 3
    logs = {}
    for i in range(num_multi_agents):
        logs[i] = []
    
    max_dist = 0
    progress_bar = tqdm(range(num_runs))
    for run in progress_bar:
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
            # scenario = env_run.unwrapped.scenario
            # print("Active goals:", scenario.goals)
            # print("Goal pairs:", scenario.goal_pairs)
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
            # Running the ego agent as deterministic, as would happen in practice
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
                # How far is i from where we predicted
                max_dist = max(max_dist, np.linalg.norm(pos[i] - swarm_state.positions[i, :]))
            progress_bar.set_postfix_str(f"max dist={max_dist:.3f}")
            done = np.all(dones)
            step_num += 1
        env_run.close()

    return logs




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

    multi_rnn_states = torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE)
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)

    # Save initial state so we can return to it later
    obs, stored_states = deterministic_reset(env, args.seed, None)

    # Collect predicted trajectory
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
                    deterministic=True, max_steps=args.episode_length)
    temp_env.close()
    pred_trajectories = []
    for agent_id in range(args.num_multi_agents):
        positions = logs[agent_id][0]["position"] # episode_length x 3
        velocities = logs[agent_id][0]["velocity"]
        pred_trajectories.append(np.concatenate([positions, velocities], axis=1))

    # Collect arm length for default radius and dt for time btn steps
    arm_len = env.quad_arm
    DELTA_T = env.control_dt
    MIN_RADIUS = arm_len
    KAPPA = 0.6 # Tune to desired
    alpha = get_alpha_bar(args.alpha, args.delta, args.num_trajectories)

    # Init r0 to some large value that ought to be safe
    delta_r = 2 # Difference between this radius and last radius
    radius = MAX_RADIUS
    radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
    filter = make_cbf_filter(radii)

    # Running list: every entry is how many env agents left their tubes that episode
    left_tubes_per_episode = []
    delta_r_per_episode = []
    num_crashes_per_episode = [] # All crashes: including when CBF fails
    num_bad_crashes_per_episode = [] # Crash outside of a tube
    prev_actions = [np.array([0.0, 0.0, 0.0, 0.0])] * args.episode_length
    max_action_diff = 4 # Instantiating to a "big" number
    max_action_diff_per_episode = []
    qj_per_episode: List[float] = []
    radius_per_episode: List[float] = []

    # While the radius hasn't converged
    for episode in range(20):
        print('EPISODE', episode)
        # Whether an agent left the tube this episode
        left_tube = [False] * args.num_multi_agents
        num_crashes = 0
        num_bad_crashes = 0

        # Find qj using old pi_j
        # Make sure the environment is reset for rollout collection
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
                    num_runs=args.num_trajectories, 
                    deterministic=True)
        # Set radius depending on how bad our prediction was
        # qj = joint_conformal_radii(logs, args.num_multi_agents, pred_trajectories, alpha, args.episode_length, args.num_trajectories)
        qj = conformal_radii(logs, args.num_multi_agents, pred_trajectories, alpha, args.episode_length)
        qj = np.max(qj)
        new_radius = explicit_radius_update(radius, qj, KAPPA)
        delta_r = np.abs(new_radius - radius) # How different is it this time
        print('radius', radius, 'qj', qj, 'new radius', new_radius)
        radius = new_radius
        radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
        qj_per_episode.append(qj)
        radius_per_episode.append(radius)
        filter = make_cbf_filter(radii) # pi_{j+1}
        temp_env.close()

        # Doublechecking that everything's reset
        obs, stored_states = deterministic_reset(env, args.seed, stored_states)
        max_action_diff = 0
        episode_pred_positions = np.zeros((args.num_multi_agents, args.episode_length, 3), dtype=np.float32)
        episode_pred_velocities = np.zeros_like(episode_pred_positions)

        progress_bar = tqdm(range(args.episode_length))
        for step in progress_bar:
            # scenario = env.unwrapped.scenario
            # print("Active goals:", scenario.goals)
            # print("Goal pairs:", scenario.goal_pairs)
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

            # Biggest change in action across timesteps in this episode
            max_action_diff = max(max_action_diff, np.linalg.norm(action_solo - prev_actions[step]))
            prev_actions[step] = action_solo # Next episode, comparing against this action

            # Checking how far I am from the nearest env quad
            # closest_dist = 100 # arbitrarily big
            # solo_pos = swarm_state.positions[-1]
            # for env_pos in swarm_state.positions[:-1]:
            #     dist = np.linalg.norm(solo_pos - env_pos)
            #     closest_dist = min(dist, closest_dist)
            # progress_bar.set_postfix_str(f"dist>={closest_dist:.2f}")

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs, rewards, terminated, truncated, infos = env.step(actions)

            # After a step, check if everyone's in their tubes and cache actual rollouts
            swarm_state = get_swarm_state(env.unwrapped)
            for agent_id in range(args.num_multi_agents):
                env_pos = swarm_state.positions[agent_id] # Actual next pos
                env_vel = swarm_state.velocities[agent_id]
                episode_pred_positions[agent_id, step, :] = env_pos
                episode_pred_velocities[agent_id, step, :] = env_vel
                pred_pos = pred_trajectories[agent_id][step][:3] # Predicted next pos
                distance = np.linalg.norm(env_pos - pred_pos)
                if distance > radius:
                    left_tube[agent_id] = True
            progress_bar.set_postfix_str(f"left={sum(left_tube)}")

            solo_info_rewards = infos[-1].get("rewards", {})
            if solo_info_rewards.get("rewraw_quadcol", 0.0) < 0.0:
                num_crashes += 1 
                # Want to check if the agent we crashed into was in their tube
                solo_pos = swarm_state.positions[-1]
                closest_dist = 100
                closest_oot = False
                for agent_id in range(args.num_multi_agents):
                    env_pos = swarm_state.positions[agent_id]
                    dist = np.linalg.norm(env_pos - solo_pos)
                    if dist <= closest_dist:
                        closest_dist = dist 
                        closest_oot = left_tube[agent_id]
                if closest_oot: # Register if closest env quad was out of tube
                    num_bad_crashes += 1

            terminated = np.asarray(terminated)
            truncated = np.asarray(truncated)
            dones = np.logical_or(terminated, truncated)

            # frame = env.render()
            # if frame is not None:
            #     video_frames.append(frame.copy())
        
        pred_trajectories = [
            np.concatenate([episode_pred_positions[agent_id], episode_pred_velocities[agent_id]], axis=1)
            for agent_id in range(args.num_multi_agents)
        ]
        # Things I'm recording every episode
        left_tubes_per_episode.append(sum(left_tube))
        delta_r_per_episode.append(delta_r)
        num_crashes_per_episode.append(num_crashes)
        num_bad_crashes_per_episode.append(num_bad_crashes)
        max_action_diff_per_episode.append(max_action_diff)
        print(f'Episode {episode}: qj={qj:.3f}, rj={radius:.3f}, delta_r={delta_r:.3f}, max_action_diff={max_action_diff:.3f}')
        print('how many left', sum(left_tube), 'num crashes', num_crashes, 'num crashes outside of traj', num_bad_crashes)
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    episodes = np.arange(len(qj_per_episode))
    plot_paths = {}
    if len(episodes) > 0:
        # Plot 1: Radius convergence
        fig, ax = plt.subplots()
        ax.plot(episodes, radius_per_episode, label="rj")
        ax.plot(episodes, qj_per_episode, label="qj")
        ax.set_title("Radius Convergence")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Radius")
        ax.legend()
        radius_plot_path = os.path.join(plots_dir, "radius_convergence.png")
        fig.savefig(radius_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["radius_convergence"] = radius_plot_path

        # Plot 2: Empirical safety coverage
        coverage_pct = (1 - (np.array(left_tubes_per_episode) / args.num_multi_agents)) * 100.0
        target_pct = (1 - args.alpha) * 100.0
        fig, ax = plt.subplots()
        ax.plot(episodes, coverage_pct, label="Coverage")
        ax.axhline(target_pct, linestyle="--", color="gray", label="Target=1-alpha%")
        ax.set_title("Empirical Safety Coverage (per Episode)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Safety Coverage (%)")
        ax.legend()
        coverage_plot_path = os.path.join(plots_dir, "empirical_safety_coverage.png")
        fig.savefig(coverage_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["coverage"] = coverage_plot_path

        # Plot 3: Safety performance
        fig, ax = plt.subplots()
        ax.plot(episodes, num_bad_crashes_per_episode, label="Bad Crashes")
        ax.set_title("Empirical Safety Performance")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Number of crashes due to poor coverage")
        ax.legend()
        safety_plot_path = os.path.join(plots_dir, "empirical_safety_performance.png")
        fig.savefig(safety_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["safety_performance"] = safety_plot_path

        # Plot 4: Convergence metrics
        fig, ax = plt.subplots()
        ax.plot(episodes, max_action_diff_per_episode, label="max_action_diff")
        ax.plot(episodes, delta_r_per_episode, label="delta_r")
        ax.set_title("Convergence")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.legend()
        convergence_plot_path = os.path.join(plots_dir, "convergence_metrics.png")
        fig.savefig(convergence_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["convergence"] = convergence_plot_path

    for plot_name, plot_path in plot_paths.items():
        print(f"[conformal] Saved {plot_name} plot to {plot_path}")

    env.close()

    if len(video_frames) > 0:
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_dir, video_file))
        print(f"[conformal_enjoy] Video saved to {final_path}")

    



if __name__ == "__main__":
    main()
