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



# ---------------------------------------------------------------------------
# Conformal utilities
# ---------------------------------------------------------------------------

def fall_down(base_action, env_state, swarm_state):
    return np.array([-1., -1., -1., -1.], dtype=np.float64)

def run_multi_agents(env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories, solo_action_fn,
                    max_steps=1500, num_runs=1, deterministic=False,
                    num_threads=1):
    '''
    Run the environment for [max_steps] steps, where the multi agents act like normal
    but the solo agent plays a fixed action, and return positions and velocities.
    Does not log the initial state.
    '''
    snapshot = safe_capture_env_snapshot(env)
    # logs: num_multi_agents + 1 x num_runs x [pos or vel] x max_steps x 3
    logs = {i: [None] * num_runs for i in range(num_multi_agents + 1)}
    max_workers = max(1, min(num_threads, num_runs))
    
    def _run_single_episode(run_idx: int):
        run_logs = {agent_id: {"position": [], "velocity": [], "goal_dist": [], "goal_swap": []} for agent_id in range(num_multi_agents + 1)}
        env_run = clone_env_from_snapshot(snapshot)
        obs_run = np.array(obs, copy=True, dtype=np.float32)
        done = False
        step_num = 0
        run_multi_rnn_states = multi_rnn_states.clone()
        run_solo_rnn_states = solo_rnn_states.clone()
        run_max_dist = 0.0

        scenario = env_run.unwrapped.scenario
        goals = []
        for agent_id in range(num_multi_agents + 1):
            active = scenario.active_goal_index[agent_id]
            target = scenario.goal_pairs[agent_id, active]
            goals.append(target)
        # print("Active starting goals:", goals)
        # print(env_run.envs[0].sense_noise.bypass, env_run.envs[0].dynamics.thrust_noise_ratio)
        # print(env_run.envs[-1].sense_noise.bypass, env_run.envs[-1].dynamics.thrust_noise_ratio)

        while not done and step_num < max_steps:
            obs_multi_dict = {OBS_KEY: obs_run[:num_multi_agents]}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_output = multi_actor(normalized_obs, run_multi_rnn_states)
            actions_multi = policy_output["actions"]
            run_multi_rnn_states = policy_output["new_rnn_states"]
            if deterministic or run_idx == 0: # First run always deterministic
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
            # Conformal radius uses predicted next timestep states
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
            for agent_id in range(num_multi_agents + 1):
                active = scenario.active_goal_index[agent_id]
                target = scenario.goal_pairs[agent_id, active]
                dist = np.linalg.norm(pos[agent_id] - target)
                run_logs[agent_id]["position"].append(pos[agent_id])
                run_logs[agent_id]["velocity"].append(vel[agent_id])
                run_logs[agent_id]["goal_dist"].append(dist)
                if not np.allclose(target, goals[agent_id]):
                    run_logs[agent_id]["goal_swap"].append(True)
                    goals[agent_id] = target 
                else:
                    run_logs[agent_id]["goal_swap"].append(False)
                if agent_id < num_multi_agents: # Don't care how far solo quad deviates
                    run_max_dist = max(run_max_dist, np.linalg.norm(pos[agent_id] - swarm_state.positions[agent_id, :]))
            done = np.all(dones)
            step_num += 1
        env_run.close()
        return run_idx, run_logs, run_max_dist

    max_dist = 0.0
    progress_bar = tqdm(total=num_runs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_episode, run_idx) for run_idx in range(num_runs)]
        for future in as_completed(futures):
            run_idx, run_logs, run_max_dist = future.result()
            for agent_id in range(num_multi_agents + 1):
                logs[agent_id][run_idx] = run_logs[agent_id]
            max_dist = max(max_dist, run_max_dist)
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"max dist={max_dist:.3f}")
    progress_bar.close()

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
    parser.add_argument("--solo_train_dir", default='train_dir', help="Directory containing the trained single-agent policy.")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", default='train_dir', help="Base directory to store the new conformal experiment.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed applied before every reset to reproduce goals.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Desired probability of conformal error")
    parser.add_argument("--delta", type=float, default=0.1, help="Desired probability of a bad draw")
    parser.add_argument("--video_name", default="conformal_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--num_eval_trajs", type=int, default=100)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--update_predictions", action="store_true", help="Whether or not to update predictions every episode.")
    parser.add_argument("--num_threads", type=int, default=None, help="Max worker threads for parallel rollouts (default: CPU count).")
    parser.add_argument("--num_episodes", type=int, default=10)
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

    if os.path.isabs(args.video_name):
        video_dir = os.path.dirname(args.video_name) or "."
        video_file = os.path.basename(args.video_name)
    else:
        video_dir = os.path.join(args.multi_train_dir, args.multi_experiment)
        video_file = args.video_name
    os.makedirs(video_dir, exist_ok=True)
    video_frames: List[np.ndarray] = []

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
    alpha = get_alpha_bar(args.alpha, args.delta, args.num_trajectories)

    # Init r0 to some large value that ought to be safe
    radius = MAX_RADIUS
    radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
    filter = make_cbf_filter(radii)

    # Running list: every entry is how many env agents left their tubes that episode
    tube_coverage_per_episode = []
    safety_per_episode = []
    crashes_per_episode = [] # All crashes: including when CBF fails
    bad_crashes_per_episode = [] # Crash outside of a tube
    qj_per_episode = []
    radius_per_episode = []
    cumulative_reward_runs_per_episode = []
    cumulative_reward_per_episode = []
    agent_locs_per_episode = [] # Save positions during one run per episode for plotting
    predicted_traj_per_episode = [] # Store predicted trajectories used each episode (agents x steps x 6)


    ##### SETTING THE RADIUS FOR EPISODE 0 #####
    episode = 0
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
                pred_trajectories, fall_down, # qj is to be based on non-interactive dynamics
                max_steps=args.episode_length, 
                num_runs=args.num_trajectories, 
                deterministic=args.deterministic,
                num_threads=max_threads)
    # Set radius depending on how bad our prediction was
    qj = conformal_radii(logs, args.num_multi_agents, pred_trajectories, alpha, args.episode_length)
    qj = np.max(qj)
    new_radius = qj # Run with normal conformal radius
    delta_r = np.abs(new_radius - radius)
    print('radius', radius, 'qj', qj, 'new radius', new_radius)
    radius = new_radius
    radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
    filter = make_cbf_filter(radii) # pi_{j+1}
    temp_env.close()

    # Find q0 using r0
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
                deterministic=args.deterministic,
                num_threads=max_threads)
    # Set radius depending on how bad our prediction was
    qj = conformal_radii(logs, args.num_multi_agents, pred_trajectories, alpha, args.episode_length)
    qj = np.max(qj) # Not actually updating radius based on qj, just wanted to know
    print('radius', radius, 'qj', qj, 'new radius', new_radius)
    temp_env.close()

    ##### COLLECTING THE DATA FOR THIS EPISODE #####
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
                num_runs=args.num_eval_trajs, 
                deterministic=args.deterministic,
                num_threads=max_threads)
    temp_env.close()

    ##### DIGEST DATA FOR THIS EPISODE #####
    cumulative_reward_per_run = []
    tube_coverage_per_run = []
    num_crashes_per_run = []
    num_bad_crashes_per_run = []
    solo_agent_id = args.num_multi_agents
    for run_id in range(args.num_eval_trajs):
        num_nonswap_steps = 0
        cumulative_reward = 0
        in_tube = [True] * args.num_multi_agents
        num_crashes = 0
        num_bad_crashes = 0
        # Skip first step so we can do comparisons with the earlier step
        # Also no chance of crashes or leaving tube in the first step
        for step in range(1, args.episode_length):
            # If, at this step, the goal didn't change wrt last step
            if not logs[solo_agent_id][run_id]['goal_swap'][step]:
                num_nonswap_steps += 1
                # Should have a smaller distance this step than last step if same goal
                delta_distance = logs[solo_agent_id][run_id]['goal_dist'][step - 1] - logs[solo_agent_id][run_id]['goal_dist'][step]
                # if run_id == 0:
                #     print(logs[solo_agent_id][run_id]['goal_dist'][step - 1], logs[solo_agent_id][run_id]['goal_dist'][step])
                if delta_distance > 0:
                    cumulative_reward += delta_distance
            solo_loc = logs[solo_agent_id][run_id]['position'][step]
            for agent_id in range(args.num_multi_agents):
                agent_loc = logs[agent_id][run_id]['position'][step]
                pred_loc = pred_trajectories[agent_id][step][:3]
                # If outside of tube, set in_tube to False: can't become True
                if np.linalg.norm(agent_loc - pred_loc) > radius:
                    in_tube[agent_id] = False
                # Record a crash if below the minimum radius
                if np.linalg.norm(solo_loc - agent_loc) <= MIN_RADIUS:
                    num_crashes += 1
                    # If crashed and that agent wasn't in their tube, presumably avoidable
                    if not in_tube[agent_id]:
                        num_bad_crashes += 1
        # Don't want to be penalized for the number of times the goal swaps since that's good
        cumulative_reward_per_run.append(cumulative_reward / num_nonswap_steps * (args.episode_length - 1))
        # Make agnostic to the number of multi agents
        tube_coverage_per_run.append(sum(in_tube) / args.num_multi_agents)
        num_crashes_per_run.append(num_crashes)
        num_bad_crashes_per_run.append(num_bad_crashes)
    # Take average over evaluation runs
    cumulative_reward_per_episode.append(sum(cumulative_reward_per_run) / args.num_eval_trajs)
    tube_coverage_per_episode.append(sum(tube_coverage_per_run) / args.num_eval_trajs)
    crashes_per_episode.append(sum(num_crashes_per_run) / args.num_eval_trajs)
    bad_crashes_per_episode.append(sum(num_bad_crashes_per_run) / args.num_eval_trajs)
    safety = 1 - sum([crashes > 0 for crashes in num_bad_crashes_per_run]) / args.num_eval_trajs
    safety_per_episode.append(safety)
    cumulative_reward_runs_per_episode.append(np.asarray(cumulative_reward_per_run, dtype=np.float32))
    # Radius updates don't change across eval runs so just append normally
    qj_per_episode.append(qj)
    radius_per_episode.append(radius)
    # Cache predicted trajectories used this episode
    predicted_traj_per_episode.append(np.asarray(pred_trajectories, dtype=np.float32))

    ##### SET UP PRED_TRAJECTORIES FOR NEXT EPISODE #####
    agent_locs = []
    for agent_id in range(args.num_multi_agents + 1):
        for step in range(args.episode_length):
            if args.update_predictions and agent_id < solo_agent_id:
                pred_trajectories[agent_id][step][:3] = logs[agent_id][0]['position'][step]
                pred_trajectories[agent_id][step][3:] = logs[agent_id][0]['velocity'][step]
        agent_locs.append(logs[agent_id][0]['position'])
    agent_locs_per_episode.append(agent_locs) # episodes x agents x steps x 3
    print(f'Cum rew: {cumulative_reward_per_episode[-1]} Tube cov: {tube_coverage_per_episode[-1]} Crashes: {crashes_per_episode[-1]} Bad crashes: {bad_crashes_per_episode[-1]}')

    ##### SETTING THE RADIUS FOR EPISODE num_episodes #####
    for i in range(args.num_episodes):
        new_radius = qj + KAPPA * delta_r 
        delta_r = np.abs(new_radius - radius) # How different is it this time
        radius = new_radius
    radii = np.full(args.num_multi_agents, radius, dtype=np.float64)
    filter = make_cbf_filter(radii) # pi_{j+1}
    
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
                deterministic=args.deterministic,
                num_threads=max_threads)
    # Set radius depending on how bad our prediction was
    qj = conformal_radii(logs, args.num_multi_agents, pred_trajectories, alpha, args.episode_length)
    qj = np.max(qj) # Not actually updating radius based on qj, just wanted to know
    print('radius', radius, 'qj', qj, 'new radius', new_radius)
    temp_env.close()

    ##### COLLECTING THE DATA FOR THIS EPISODE #####
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
                num_runs=args.num_eval_trajs, 
                deterministic=args.deterministic,
                num_threads=max_threads)
    temp_env.close()

    ##### DIGEST DATA FOR THIS EPISODE #####
    cumulative_reward_per_run = []
    tube_coverage_per_run = []
    num_crashes_per_run = []
    num_bad_crashes_per_run = []
    solo_agent_id = args.num_multi_agents
    for run_id in range(args.num_eval_trajs):
        num_nonswap_steps = 0
        cumulative_reward = 0
        in_tube = [True] * args.num_multi_agents
        num_crashes = 0
        num_bad_crashes = 0
        # Skip first step so we can do comparisons with the earlier step
        # Also no chance of crashes or leaving tube in the first step
        for step in range(1, args.episode_length):
            # If, at this step, the goal didn't change wrt last step
            if not logs[solo_agent_id][run_id]['goal_swap'][step]:
                num_nonswap_steps += 1
                # Should have a smaller distance this step than last step if same goal
                delta_distance = logs[solo_agent_id][run_id]['goal_dist'][step - 1] - logs[solo_agent_id][run_id]['goal_dist'][step]
                # if run_id == 0:
                #     print(logs[solo_agent_id][run_id]['goal_dist'][step - 1], logs[solo_agent_id][run_id]['goal_dist'][step])
                if delta_distance > 0:
                    cumulative_reward += delta_distance
            solo_loc = logs[solo_agent_id][run_id]['position'][step]
            for agent_id in range(args.num_multi_agents):
                agent_loc = logs[agent_id][run_id]['position'][step]
                pred_loc = pred_trajectories[agent_id][step][:3]
                # If outside of tube, set in_tube to False: can't become True
                if np.linalg.norm(agent_loc - pred_loc) > radius:
                    in_tube[agent_id] = False
                # Record a crash if below the minimum radius
                if np.linalg.norm(solo_loc - agent_loc) <= MIN_RADIUS:
                    num_crashes += 1
                    # If crashed and that agent wasn't in their tube, presumably avoidable
                    if not in_tube[agent_id]:
                        num_bad_crashes += 1
        # Don't want to be penalized for the number of times the goal swaps since that's good
        cumulative_reward_per_run.append(cumulative_reward / num_nonswap_steps * (args.episode_length - 1))
        # Make agnostic to the number of multi agents
        tube_coverage_per_run.append(sum(in_tube) / args.num_multi_agents)
        num_crashes_per_run.append(num_crashes)
        num_bad_crashes_per_run.append(num_bad_crashes)
    # Take average over evaluation runs
    cumulative_reward_per_episode.append(sum(cumulative_reward_per_run) / args.num_eval_trajs)
    tube_coverage_per_episode.append(sum(tube_coverage_per_run) / args.num_eval_trajs)
    crashes_per_episode.append(sum(num_crashes_per_run) / args.num_eval_trajs)
    bad_crashes_per_episode.append(sum(num_bad_crashes_per_run) / args.num_eval_trajs)
    safety = 1 - sum([crashes > 0 for crashes in num_bad_crashes_per_run]) / args.num_eval_trajs
    safety_per_episode.append(safety)
    cumulative_reward_runs_per_episode.append(np.asarray(cumulative_reward_per_run, dtype=np.float32))
    # Radius updates don't change across eval runs so just append normally
    qj_per_episode.append(qj)
    radius_per_episode.append(radius)
    # Cache predicted trajectories used this episode
    predicted_traj_per_episode.append(np.asarray(pred_trajectories, dtype=np.float32))

    ##### SET UP PRED_TRAJECTORIES FOR NEXT EPISODE #####
    agent_locs = []
    for agent_id in range(args.num_multi_agents + 1):
        for step in range(args.episode_length):
            if args.update_predictions and agent_id < solo_agent_id:
                pred_trajectories[agent_id][step][:3] = logs[agent_id][0]['position'][step]
                pred_trajectories[agent_id][step][3:] = logs[agent_id][0]['velocity'][step]
        agent_locs.append(logs[agent_id][0]['position'])
    agent_locs_per_episode.append(agent_locs) # episodes x agents x steps x 3
    print(f'Cum rew: {cumulative_reward_per_episode[-1]} Tube cov: {tube_coverage_per_episode[-1]} Crashes: {crashes_per_episode[-1]} Bad crashes: {bad_crashes_per_episode[-1]}')

    # Persist per-episode metrics for offline plotting
    metrics_path = os.path.join(experiment_dir, "conformal_metrics.npz")
    np.savez(
        metrics_path,
        episodes=np.array([0, args.num_episodes]),
        qj_per_episode=np.asarray(qj_per_episode, dtype=np.float32),
        radius_per_episode=np.asarray(radius_per_episode, dtype=np.float32),
        tube_coverage_per_episode=np.asarray(tube_coverage_per_episode, dtype=np.float32),
        crashes_per_episode=np.asarray(crashes_per_episode, dtype=np.float32),
        bad_crashes_per_episode=np.asarray(bad_crashes_per_episode, dtype=np.float32),
        safety_per_episode=np.asarray(safety_per_episode, dtype=np.float32),
        cumulative_reward_per_episode=np.asarray(cumulative_reward_per_episode, dtype=np.float32),
        cumulative_reward_per_run=np.asarray(cumulative_reward_runs_per_episode, dtype=np.float32),
        agent_locs_per_episode=np.asarray(agent_locs_per_episode, dtype=np.float32),
        predicted_traj_per_episode=np.asarray(predicted_traj_per_episode, dtype=np.float32),
        alpha=args.alpha,
        delta=args.delta,
        bar_alpha=alpha,
    )
    print(f"[conformal] Saved per-episode metrics to {metrics_path}")
    env.close()

    if len(video_frames) > 0:
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_dir, video_file))
        print(f"[conformal] Video saved to {final_path}")
    
    print('Number of total crashes:', sum(crashes_per_episode))
    print('Number of total crashes outside of conformal tubes:', sum(bad_crashes_per_episode))

    



if __name__ == "__main__":
    main()
