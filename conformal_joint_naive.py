#!/usr/bin/env python3
"""
Conformal-style joint CBF evaluation for multi-agent patrol.

Each agent runs the same solo policy. A joint CBF filter (full_cbf_utils) modifies
the stacked actions for all agents simultaneously. The conformal radius is updated
from trajectory-level mismatch between cbf_dynamics and real_dynamics on the
concatenated full swarm state.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.huggingface.huggingface_utils import generate_replay_video
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.conformal_utils import explicit_radius_update, get_alpha_bar
from project_utils.full_cbf_utils import (
    CBF_K0,
    CBF_K1,
    apply_cbf_filter,
    cbf_dynamics,
    real_dynamics,
)
from project_utils.restart_utils import deterministic_reset, extract_positions_velocities
from project_utils.utils import OBS_KEY, load_actor, load_cfg, latest_checkpoint

DEVICE = torch.device("cpu")
MAX_R = 8.0
COLLISION_FAR_DISTANCE = 10000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint conformal CBF evaluation on patrol_dual_goal.")
    parser.add_argument("--solo_train_dir", default="train_dir", help="Directory containing the trained single-agent policy.")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", default="train_dir", help="Base directory to store outputs.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed applied before every reset.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Desired probability of conformal error.")
    parser.add_argument("--delta", type=float, default=0.1, help="Desired probability of a bad draw.")
    parser.add_argument("--video_name", default="conformal_joint_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--num_eval_trajs", type=int, default=100)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_agents", type=int, default=8, help="Number of patrol agents.")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--initial_r", type=float, default=2.0, help="Initial conformal safety radius.")
    parser.add_argument("--separation_radius", type=float, default=0.5, help="Desired pairwise separation distance enforced by CBF.")
    parser.add_argument("--gamma", type=float, default=0.8, help="CBF gamma in (0, 1].")
    parser.add_argument("--disable_boundary_collision", action="store_true", help="Move room boundaries far enough to effectively disable wall/ceiling/floor collisions.")
    return parser.parse_args()


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def _configure_far_boundary_geometry(env_unwrapped, far_distance: float = COLLISION_FAR_DISTANCE) -> None:
    far_distance = float(far_distance)
    if far_distance <= 0.0:
        raise ValueError("--disable_boundary_collision requires a positive far distance")
    far_box = np.array([[-far_distance, -far_distance, -far_distance], [far_distance, far_distance, far_distance]])
    env_unwrapped.room_box = far_box.copy()
    for quad in env_unwrapped.envs:
        quad.room_box = far_box.copy()
        quad.dynamics.room_box = far_box.copy()
        quad.dynamics.floor_threshold = -far_distance


def _pack_state_tuple(pos: np.ndarray, vel: np.ndarray, rot: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.concatenate([pos.reshape(-1), vel.reshape(-1), rot.reshape(-1), omega.reshape(-1)], axis=0).astype(np.float64)


def _joint_state_model_mismatch(actions: np.ndarray, env_unwrapped) -> float:
    """
    One-step full-state mismatch on concatenated swarm state:
    || concat_i x_i^cbf(next) - concat_i x_i^real(next) ||_2
    """
    actions = np.asarray(actions, dtype=np.float64)
    dt = float(env_unwrapped.control_dt)
    states_cbf = []
    states_real = []
    for agent_id, quad in enumerate(env_unwrapped.envs):
        dynamics = quad.dynamics
        normalized = np.clip(0.5 * (actions[agent_id] + 1.0), 0.0, 1.0)
        states_cbf.append(_pack_state_tuple(*cbf_dynamics(normalized, dynamics, dt)))
        states_real.append(_pack_state_tuple(*real_dynamics(normalized, dynamics, dt)))
    concat_cbf = np.concatenate(states_cbf, axis=0)
    concat_real = np.concatenate(states_real, axis=0)
    return float(np.linalg.norm(concat_cbf - concat_real))


def make_joint_cbf_filter(r_mismatch: float, separation_radius: float, gamma: float):
    def _filter(base_action: np.ndarray, env_state):
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_state.unwrapped,
            r=float(r_mismatch),
            separation_radius=float(separation_radius),
            gamma=float(gamma),
        )

    return _filter


def conformal_qj(logs: List[Dict[str, np.ndarray]], alpha: float, episode_length: int) -> float:
    """
    Trajectory-level score is max full-state mismatch across timesteps.
    """
    scores = []
    for run_log in logs:
        max_mismatch = 0.0
        mismatch = run_log["model_mismatch_state"]
        for step in range(min(episode_length, len(mismatch))):
            max_mismatch = max(max_mismatch, float(mismatch[step]))
        scores.append(max_mismatch)
    if len(scores) == 0:
        return 0.0
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    idx = int(np.ceil(len(scores) * (1 - alpha)) - 1)
    idx = int(np.clip(idx, 0, len(scores) - 1))
    return float(scores[idx])


def run_joint_agents(
    env,
    obs,
    solo_actor,
    init_rnn_states,
    solo_obs_dim,
    joint_action_fn,
    num_agents: int,
    max_steps=1500,
    num_runs=1,
    deterministic=False,
    disable_boundary_collision=False,
    boundary_far_distance: float = COLLISION_FAR_DISTANCE,
):
    """
    Run environment for max_steps and return run-level logs.
    """
    snapshot = safe_capture_env_snapshot(env)
    logs = [None] * num_runs
    running_max_mismatch = 0.0
    progress_bar = tqdm(range(num_runs))

    for run_idx in progress_bar:
        run_logs = {
            "positions": [],
            "velocities": [],
            "goal_dist": [],
            "goal_swap": [],
            "model_mismatch_state": [],
        }
        env_run = clone_env_from_snapshot(snapshot)
        if disable_boundary_collision:
            _configure_far_boundary_geometry(env_run.unwrapped, boundary_far_distance)
        obs_run = np.array(obs, copy=True, dtype=np.float32)
        done = False
        step_num = 0
        run_rnn_states = init_rnn_states.clone()

        scenario = env_run.unwrapped.scenario
        goals = []
        for agent_id in range(num_agents):
            active = scenario.active_goal_index[agent_id]
            target = scenario.goal_pairs[agent_id, active]
            goals.append(np.asarray(target, dtype=np.float64).copy())

        while not done and step_num < max_steps:
            obs_self = obs_run[:, :solo_obs_dim]
            obs_dict = {OBS_KEY: obs_self}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(solo_actor, obs_dict)
                policy_out = solo_actor(normalized_obs, run_rnn_states)
            run_rnn_states = policy_out["new_rnn_states"]
            actions = policy_out["actions"]
            if deterministic or run_idx == 0:
                actions = argmax_actions(solo_actor.action_distribution())
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
            actions = actions.detach().cpu().numpy()

            actions = joint_action_fn(base_action=actions, env_state=env_run)
            mismatch = _joint_state_model_mismatch(actions, env_run.unwrapped)

            obs_run, rewards, dones, infos = env_run.step(actions)
            obs_run = np.array(obs_run, dtype=np.float32)
            pos, vel = extract_positions_velocities(env_run.unwrapped)

            step_goal_dist = []
            step_goal_swap = []
            for agent_id in range(num_agents):
                active = scenario.active_goal_index[agent_id]
                target = scenario.goal_pairs[agent_id, active]
                dist = np.linalg.norm(pos[agent_id] - target)
                step_goal_dist.append(dist)
                goal_changed = not np.allclose(target, goals[agent_id])
                step_goal_swap.append(goal_changed)
                if goal_changed:
                    goals[agent_id] = np.asarray(target, dtype=np.float64).copy()

            run_logs["positions"].append(pos.copy())
            run_logs["velocities"].append(vel.copy())
            run_logs["goal_dist"].append(np.asarray(step_goal_dist, dtype=np.float64))
            run_logs["goal_swap"].append(np.asarray(step_goal_swap, dtype=np.bool_))
            run_logs["model_mismatch_state"].append(mismatch)

            done = np.all(dones)
            step_num += 1

        for key in run_logs:
            run_logs[key] = np.asarray(run_logs[key])
        if run_logs["model_mismatch_state"].size > 0:
            running_max_mismatch = max(running_max_mismatch, float(np.max(run_logs["model_mismatch_state"])))
        progress_bar.set_postfix_str(f"max mismatch={running_max_mismatch:.6f}")
        logs[run_idx] = run_logs
        env_run.close()

    return logs


def _collision_indicator_from_positions(positions_t: np.ndarray, min_r: float) -> int:
    for i, j in combinations(range(positions_t.shape[0]), 2):
        if np.linalg.norm(positions_t[i] - positions_t[j]) <= min_r:
            return 1
    return 0


def _pairwise_h_min(positions_t: np.ndarray, separation_radius: float) -> float:
    min_h = float("inf")
    for i, j in combinations(range(positions_t.shape[0]), 2):
        margin = float(np.linalg.norm(positions_t[i] - positions_t[j]) - separation_radius)
        min_h = min(min_h, margin)
    return min_h


def main() -> None:
    args = parse_args()
    if args.num_agents < 2:
        raise ValueError("--num_agents must be >= 2 for joint CBF.")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must satisfy 0 < gamma <= 1")
    if args.separation_radius <= 0.0:
        raise ValueError("--separation_radius must be > 0")

    torch.set_grad_enabled(False)
    register_swarm_components()

    if os.path.isabs(args.video_name):
        video_dir = os.path.dirname(args.video_name) or "."
        video_file = os.path.basename(args.video_name)
    else:
        video_dir = os.path.join(args.solo_train_dir, args.solo_experiment)
        video_file = args.video_name
    os.makedirs(video_dir, exist_ok=True)
    video_frames: List[np.ndarray] = []

    experiment_dir = ensure_experiment_dir(args.train_dir, args.experiment_name)
    args_path = os.path.join(experiment_dir, "conformal_joint_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={args.num_agents}",
        "--quads_neighbor_visible_num=0",
        "--quads_neighbor_obs_type=none",
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
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode="rgb_array")
    if args.disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)

    solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    solo_rnn_states = torch.zeros((args.num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
    obs, stored_states = deterministic_reset(env, args.seed, None)

    arm_len = env.quad_arm
    min_r = arm_len * 2.5
    alpha = get_alpha_bar(args.alpha, args.delta, args.num_trajectories)
    r_mismatch = float(np.clip(args.initial_r, 0.0, MAX_R))
    filter_fn = make_joint_cbf_filter(r_mismatch, args.separation_radius, args.gamma)

    qj_per_episode = []
    r_mismatch_per_episode = []
    safety_per_episode = []
    crashes_per_episode = []
    cumulative_reward_per_episode = []
    cumulative_reward_runs_per_episode = []
    mismatch_per_episode = []
    mismatch_q10_per_episode = []
    mismatch_q90_per_episode = []
    mismatch_runs_per_episode = []
    agent_locs_per_episode = []
    h_violation_per_episode = []
    h_violation_runs_per_episode = []
    h_min_per_episode = []

    for episode in range(args.num_episodes):
        print("EPISODE", episode)

        solo_rnn_states = torch.zeros((args.num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
        obs, stored_states = deterministic_reset(env, args.seed, stored_states)
        snapshot = safe_capture_env_snapshot(env)
        temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
        cal_logs = run_joint_agents(
            temp_env,
            obs,
            solo_actor,
            solo_rnn_states,
            solo_obs_dim,
            filter_fn,
            num_agents=args.num_agents,
            max_steps=args.episode_length,
            num_runs=args.num_trajectories,
            deterministic=args.deterministic,
            disable_boundary_collision=args.disable_boundary_collision,
        )
        qj = conformal_qj(cal_logs, alpha, args.episode_length)
        new_r = float(np.clip(qj, 0.0, MAX_R))
        print("r_mismatch", r_mismatch, "qj", qj, "new_r_mismatch", new_r, "separation_radius", args.separation_radius)
        r_mismatch = float(new_r)
        filter_fn = make_joint_cbf_filter(r_mismatch, args.separation_radius, args.gamma)
        temp_env.close()

        solo_rnn_states = torch.zeros((args.num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
        obs, stored_states = deterministic_reset(env, args.seed, stored_states)
        snapshot = safe_capture_env_snapshot(env)
        temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
        logs = run_joint_agents(
            temp_env,
            obs,
            solo_actor,
            solo_rnn_states,
            solo_obs_dim,
            filter_fn,
            num_agents=args.num_agents,
            max_steps=args.episode_length,
            num_runs=args.num_eval_trajs,
            deterministic=args.deterministic,
            disable_boundary_collision=args.disable_boundary_collision,
        )
        temp_env.close()

        cumulative_reward_per_run = []
        crash_indicator_per_run = []
        max_mismatch_per_run = []
        h_violation_per_run = []
        h_min_per_run = []
        for run_id in range(args.num_eval_trajs):
            run = logs[run_id]
            goal_dist = run["goal_dist"]        # (T, N)
            goal_swap = run["goal_swap"]        # (T, N)
            positions = run["positions"]        # (T, N, 3)

            cumulative_reward = 0.0
            nonswap_steps = 0
            for t in range(1, goal_dist.shape[0]):
                for agent_id in range(args.num_agents):
                    if not bool(goal_swap[t, agent_id]):
                        nonswap_steps += 1
                        delta = float(goal_dist[t - 1, agent_id] - goal_dist[t, agent_id])
                        if delta > 0.0:
                            cumulative_reward += delta
            if nonswap_steps > 0:
                cumulative_reward = cumulative_reward / nonswap_steps * max(1, goal_dist.shape[0] - 1)

            had_crash = 0
            min_h = float("inf")
            for t in range(positions.shape[0]):
                step_h = _pairwise_h_min(positions[t], args.separation_radius)
                min_h = min(min_h, step_h)
                had_crash = max(had_crash, _collision_indicator_from_positions(positions[t], min_r))

            cumulative_reward_per_run.append(cumulative_reward)
            crash_indicator_per_run.append(had_crash)
            max_mismatch_per_run.append(float(np.max(run["model_mismatch_state"])) if run["model_mismatch_state"].size > 0 else 0.0)
            h_violation_per_run.append(1.0 if min_h <= 0.0 else 0.0)
            h_min_per_run.append(min_h)

        cumulative_reward_per_episode.append(float(np.mean(cumulative_reward_per_run)))
        cumulative_reward_runs_per_episode.append(np.asarray(cumulative_reward_per_run, dtype=np.float32))
        crashes_per_episode.append(float(np.mean(crash_indicator_per_run)))
        safety_per_episode.append(1.0 - float(np.mean(crash_indicator_per_run)))
        mismatch_per_episode.append(float(np.mean(max_mismatch_per_run)))
        mismatch_q10_per_episode.append(float(np.quantile(max_mismatch_per_run, 0.10)))
        mismatch_q90_per_episode.append(float(np.quantile(max_mismatch_per_run, 0.90)))
        mismatch_runs_per_episode.append(np.asarray(max_mismatch_per_run, dtype=np.float32))
        h_violation_per_episode.append(float(np.mean(h_violation_per_run)))
        h_violation_runs_per_episode.append(np.asarray(h_violation_per_run, dtype=np.float32))
        h_min_per_episode.append(float(np.mean(h_min_per_run)))
        qj_per_episode.append(float(qj))
        r_mismatch_per_episode.append(float(r_mismatch))
        agent_locs_per_episode.append(logs[0]["positions"])
        print(
            f"Cum rew: {cumulative_reward_per_episode[-1]} "
            f"Crash rate: {crashes_per_episode[-1]} "
            f"Max state mismatch: {mismatch_per_episode[-1]}"
        )

    metrics_path = os.path.join(experiment_dir, "conformal_joint_metrics.npz")
    np.savez(
        metrics_path,
        episodes=np.arange(args.num_episodes),
        qj_per_episode=np.asarray(qj_per_episode, dtype=np.float32),
        r_mismatch_per_episode=np.asarray(r_mismatch_per_episode, dtype=np.float32),
        crashes_per_episode=np.asarray(crashes_per_episode, dtype=np.float32),
        safety_per_episode=np.asarray(safety_per_episode, dtype=np.float32),
        cumulative_reward_per_episode=np.asarray(cumulative_reward_per_episode, dtype=np.float32),
        cumulative_reward_per_run=np.asarray(cumulative_reward_runs_per_episode, dtype=np.float32),
        h_violation_per_episode=np.asarray(h_violation_per_episode, dtype=np.float32),
        h_violation_per_run=np.asarray(h_violation_runs_per_episode, dtype=np.float32),
        h_min_per_episode=np.asarray(h_min_per_episode, dtype=np.float32),
        mismatch_per_episode=np.asarray(mismatch_per_episode, dtype=np.float32),
        mismatch_q10_per_episode=np.asarray(mismatch_q10_per_episode, dtype=np.float32),
        mismatch_q90_per_episode=np.asarray(mismatch_q90_per_episode, dtype=np.float32),
        mismatch_per_run=np.asarray(mismatch_runs_per_episode, dtype=np.float32),
        agent_locs_first_run=np.asarray(agent_locs_per_episode, dtype=np.float32),
        alpha=args.alpha,
        delta=args.delta,
        bar_alpha=alpha,
        cbf_k0=CBF_K0,
        cbf_k1=CBF_K1,
    )
    print(f"[conformal_joint] Saved per-episode metrics to {metrics_path}")
    env.close()

    if len(video_frames) > 0:
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_dir, video_file))
        print(f"[conformal_joint] Video saved to {final_path}")

    print("Number of total crashes:", int(np.sum(np.asarray(crashes_per_episode) * args.num_eval_trajs)))


if __name__ == "__main__":
    main()
