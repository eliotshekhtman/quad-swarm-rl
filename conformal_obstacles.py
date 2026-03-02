#!/usr/bin/env python3
"""
Conformal-style single-agent obstacle avoidance evaluation.

This mirrors conformal.py structure, but replaces multi-agent tube constraints
with obstacle-aware CBF constraints.
"""

from __future__ import annotations

import argparse
import json
import os
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
from project_utils.cbf_utils import CBF_K0, CBF_K1, apply_cbf_filter, cbf_dynamics, real_dynamics
from project_utils.restart_utils import deterministic_reset, extract_positions_velocities
from project_utils.utils import OBS_KEY, load_actor, load_cfg, latest_checkpoint

DEVICE = torch.device("cpu")
MAX_RADIUS = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conformal obstacle-avoidance evaluation.")
    parser.add_argument("--solo_train_dir", default="train_dir", help="Directory containing the trained single-agent policy.")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", default="train_dir", help="Base directory to store the new conformal experiment.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed applied before every reset.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Desired probability of conformal error.")
    parser.add_argument("--delta", type=float, default=0.1, help="Desired probability of a bad draw.")
    parser.add_argument("--video_name", default="conformal_obstacles_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--num_eval_trajs", type=int, default=100)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--quads_mode", default="o_static_same_goal", choices=["o_static_same_goal", "o_random", "o_dynamic_same_goal"])
    parser.add_argument("--quads_obst_spawn_area", nargs=2, type=float, default=[8.0, 8.0])
    parser.add_argument("--quads_obst_density", type=float, default=0.2)
    parser.add_argument("--quads_obst_size", type=float, default=0.6)

    parser.add_argument("--kappa", type=float, default=0.6, help="Radius update aggressiveness.")
    parser.add_argument("--initial_radius", type=float, default=2.0, help="Initial conformal safety radius.")
    parser.add_argument("--obstacle_radius_margin", type=float, default=0.05, help="Extra radius added to each obstacle for CBF constraints.")
    parser.add_argument("--gamma", type=float, default=0.8, help="CBF gamma in (0, 1].")
    return parser.parse_args()


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def _pack_state_tuple(pos: np.ndarray, vel: np.ndarray, rot: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.concatenate([pos.reshape(-1), vel.reshape(-1), rot.reshape(-1), omega.reshape(-1)], axis=0).astype(np.float64)


def _capture_environment_geometry(env) -> Dict[str, object]:
    """Capture the current single-agent obstacle environment geometry."""
    env_unwrapped = env.unwrapped
    return {
        "start_point": np.asarray(env_unwrapped.envs[0].dynamics.pos, dtype=np.float64).tolist(),
        "goal_point": np.asarray(env_unwrapped.envs[0].goal, dtype=np.float64).tolist(),
        "obstacle_radius": float(env_unwrapped.obstacles.obstacle_radius),
        "obstacle_positions": np.asarray(env_unwrapped.obstacles.pos_arr, dtype=np.float64).tolist(),
    }


def _print_initial_environment_geometry(env, obstacle_radius_margin: float) -> None:
    """Print the solo quad start/goal and obstacle geometry before rollouts start."""
    env_unwrapped = env.unwrapped
    start_point = np.asarray(env_unwrapped.envs[0].dynamics.pos, dtype=np.float64)
    goal_point = np.asarray(env_unwrapped.envs[0].goal, dtype=np.float64)
    obstacle_positions = np.asarray(env_unwrapped.obstacles.pos_arr, dtype=np.float64)
    base_radius = float(env_unwrapped.obstacles.obstacle_radius)
    inflated_radius = base_radius + float(obstacle_radius_margin)

    print("[conformal_obstacles] Initial geometry before episode 0")
    print(f"  start_point: {start_point.tolist()}")
    print(f"  goal_point: {goal_point.tolist()}")
    print(
        f"  obstacle_radius: {base_radius:.6f} "
        f"(inflated_for_cbf: {inflated_radius:.6f})"
    )
    if obstacle_positions.size == 0:
        print("  obstacles: []")
        return
    for idx, center in enumerate(obstacle_positions):
        print(
            f"  obstacle[{idx}]: position={np.asarray(center, dtype=np.float64).tolist()}, "
            f"radius={base_radius:.6f} (inflated_for_cbf={inflated_radius:.6f})"
        )


def make_obstacle_cbf_filter(radius: float, gamma: float, obstacle_radius_margin: float):
    def _filter(base_action: np.ndarray, env_state, _unused_swarm_state=None):
        env_unwrapped = env_state.unwrapped
        if not getattr(env_unwrapped, "use_obstacles", False) or env_unwrapped.obstacles is None:
            return np.asarray(base_action, dtype=np.float32)

        obstacle_centers = np.asarray(env_unwrapped.obstacles.pos_arr, dtype=np.float64)
        if obstacle_centers.size == 0:
            return np.asarray(base_action, dtype=np.float32)

        quad = env_unwrapped.envs[0]
        dynamics = quad.dynamics
        quad_state = _pack_state_tuple(
            np.asarray(dynamics.pos, dtype=np.float64),
            np.asarray(dynamics.vel, dtype=np.float64),
            np.asarray(dynamics.rot, dtype=np.float64),
            np.asarray(dynamics.omega, dtype=np.float64),
        )
        inflated_radius = float(env_unwrapped.obstacles.obstacle_radius) + float(obstacle_radius_margin)
        obstacles = [
            {
                "position": np.asarray(center, dtype=np.float64),
                "velocity": np.zeros(3, dtype=np.float64),
                "radius": inflated_radius,
            }
            for center in obstacle_centers
        ]
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_unwrapped,
            quad_state=quad_state,
            obstacles=obstacles,
            r=float(radius),
            gamma=float(gamma),
        )

    return _filter


def _state_model_mismatch(
    action: np.ndarray,
    dynamics,
    dt: float,
) -> float:
    """
    One-step full-state mismatch between cbf_dynamics and real_dynamics.
    """
    action = np.asarray(action, dtype=np.float64)
    normalized = np.clip(0.5 * (action + 1.0), 0.0, 1.0)
    state_cbf = _pack_state_tuple(*cbf_dynamics(normalized, dynamics, dt))
    state_real = _pack_state_tuple(*real_dynamics(normalized, dynamics, dt))
    return float(np.linalg.norm(state_cbf - state_real))


def conformal_qj(logs: List[Dict[str, np.ndarray]], alpha: float, episode_length: int) -> float:
    """
    Trajectory-level score is maximum full-state mismatch across timesteps.
    We use the upper-tail quantile to get a high-probability upper bound.
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


def run_single_agent(
    env,
    obs,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim,
    solo_action_fn,
    max_steps=1500,
    num_runs=1,
    deterministic=False,
):
    """
    Run the environment for max_steps steps and return solo-agent logs.
    Does not log the initial state.
    """
    snapshot = safe_capture_env_snapshot(env)
    logs = [None] * num_runs
    running_max_mismatch = 0.0

    progress_bar = tqdm(range(num_runs))
    for run_idx in progress_bar:
        run_logs = {
            "position": [],
            "velocity": [],
            "goal_dist": [],
            "collision_obstacle": [],
            "boundary_dist": [],
            "clearance": [],
            "model_mismatch_state": [],
        }
        env_run = clone_env_from_snapshot(snapshot)
        obs_run = np.array(obs, copy=True, dtype=np.float32)
        done = False
        step_num = 0
        run_solo_rnn_states = solo_rnn_states.clone()

        while not done and step_num < max_steps:
            obs_solo_self = obs_run[0, :solo_obs_dim]
            obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
            with torch.no_grad():
                normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
                policy_solo = solo_actor(normalized_solo, run_solo_rnn_states)
            run_solo_rnn_states = policy_solo["new_rnn_states"]
            action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]

            action_solo = solo_action_fn(
                base_action=action_solo,
                env_state=env_run,
                _unused_swarm_state=None,
            )
            mismatch_state = _state_model_mismatch(
                action_solo,
                env_run.unwrapped.envs[0].dynamics,
                float(env_run.unwrapped.control_dt),
            )

            actions = action_solo[None, :]
            obs_run, rewards, dones, infos = env_run.step(actions)
            obs_run = np.array(obs_run, dtype=np.float32)

            pos, vel = extract_positions_velocities(env_run.unwrapped)
            solo_pos = pos[0]
            goal = np.asarray(env_run.unwrapped.envs[0].goal, dtype=np.float64)
            goal_dist = float(np.linalg.norm(solo_pos - goal))

            obstacle_centers = np.asarray(env_run.unwrapped.obstacles.pos_arr, dtype=np.float64)
            obstacle_radius = float(env_run.unwrapped.obstacles.obstacle_radius)
            center_dists = np.linalg.norm(obstacle_centers - solo_pos[None, :], axis=1)
            boundary_dist = float(np.min(center_dists - obstacle_radius))
            clearance = boundary_dist

            rew_obst_raw = infos[0]["rewards"].get("rewraw_quadcol_obstacle", 0.0)
            collision_obstacle = float(rew_obst_raw) < 0.0

            run_logs["position"].append(pos[0])
            run_logs["velocity"].append(vel[0])
            run_logs["goal_dist"].append(goal_dist)
            run_logs["collision_obstacle"].append(collision_obstacle)
            run_logs["boundary_dist"].append(boundary_dist)
            run_logs["clearance"].append(clearance)
            run_logs["model_mismatch_state"].append(mismatch_state)

            done = np.all(dones)
            step_num += 1

        for key in run_logs:
            run_logs[key] = np.asarray(run_logs[key], dtype=np.float32)
        if len(run_logs["model_mismatch_state"]) > 0:
            running_max_mismatch = max(running_max_mismatch, float(np.max(run_logs["model_mismatch_state"])))
        progress_bar.set_postfix_str(f"max mismatch={running_max_mismatch:.6f}")
        logs[run_idx] = run_logs
        env_run.close()

    return logs


def main() -> None:
    args = parse_args()
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must satisfy 0 < gamma <= 1")

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
    args_path = os.path.join(experiment_dir, "conformal_obstacles_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_use_obstacles=True",
        f"--quads_mode={args.quads_mode}",
        "--quads_num_agents=1",
        "--quads_neighbor_visible_num=0",
        "--quads_neighbor_obs_type=none",
        "--quads_obstacle_obs_type=octomap",
        f"--quads_obst_density={args.quads_obst_density}",
        f"--quads_obst_size={args.quads_obst_size}",
        "--quads_obst_spawn_area",
        str(args.quads_obst_spawn_area[0]),
        str(args.quads_obst_spawn_area[1]),
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

    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=render_mode)
    solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
    # Match conformal.py behavior: load actor with its original training observation space,
    # then slice runtime observations to that dimension.
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)

    obs, stored_states = deterministic_reset(env, args.seed, None)
    _print_initial_environment_geometry(env, args.obstacle_radius_margin)
    initial_geometry = _capture_environment_geometry(env)
    env_json_path = os.path.join(experiment_dir, "conformal_obstacles_environment.json")
    with open(env_json_path, "w", encoding="utf-8") as f:
        json.dump(initial_geometry, f, indent=2)
    print(f"[conformal_obstacles] Saved initial environment geometry to {env_json_path}")
    alpha = get_alpha_bar(args.alpha, args.delta, args.num_trajectories)

    radius = float(np.clip(args.initial_radius, 0.0, MAX_RADIUS))
    filter_fn = make_obstacle_cbf_filter(radius, args.gamma, args.obstacle_radius_margin)

    qj_per_episode = []
    radius_per_episode = []
    crashes_per_episode = []
    safety_per_episode = []
    cumulative_reward_per_episode = []
    cumulative_reward_runs_per_episode = []
    min_clearance_per_episode = []
    mismatch_per_episode = []
    mismatch_runs_per_episode = []
    run_logs_per_episode = []
    for episode in range(args.num_episodes):
        print("EPISODE", episode)

        solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
        obs, stored_states = deterministic_reset(env, args.seed, stored_states)
        snapshot = safe_capture_env_snapshot(env)
        temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
        cal_logs = run_single_agent(
            temp_env,
            obs,
            solo_actor,
            solo_rnn_states,
            solo_obs_dim,
            filter_fn,
            max_steps=args.episode_length,
            num_runs=args.num_trajectories,
            deterministic=args.deterministic,
        )
        qj = conformal_qj(cal_logs, alpha, args.episode_length)
        new_radius = explicit_radius_update(radius, qj, args.kappa, 0.0, MAX_RADIUS)
        print("radius", radius, "qj", qj, "new radius", new_radius)
        radius = float(new_radius)
        filter_fn = make_obstacle_cbf_filter(radius, args.gamma, args.obstacle_radius_margin)
        temp_env.close()

        solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
        obs, stored_states = deterministic_reset(env, args.seed, stored_states)
        snapshot = safe_capture_env_snapshot(env)
        temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
        logs = run_single_agent(
            temp_env,
            obs,
            solo_actor,
            solo_rnn_states,
            solo_obs_dim,
            filter_fn,
            max_steps=args.episode_length,
            num_runs=args.num_eval_trajs,
            deterministic=args.deterministic,
        )
        temp_env.close()

        cumulative_reward_per_run = []
        num_crashes_per_run = []
        min_clearance_per_run = []
        max_mismatch_per_run = []
        for run_id in range(args.num_eval_trajs):
            goal_dists = logs[run_id]["goal_dist"]
            goal_delta = goal_dists[:-1] - goal_dists[1:]
            positive_progress = np.maximum(goal_delta, 0.0)
            cumulative_reward = float(np.sum(positive_progress))
            had_crash = bool(np.any(logs[run_id]["collision_obstacle"]))
            min_clearance = float(np.min(logs[run_id]["clearance"]))
            max_mismatch = float(np.max(logs[run_id]["model_mismatch_state"]))

            cumulative_reward_per_run.append(cumulative_reward)
            num_crashes_per_run.append(1 if had_crash else 0)
            min_clearance_per_run.append(min_clearance)
            max_mismatch_per_run.append(max_mismatch)

        cumulative_reward_per_episode.append(float(np.mean(cumulative_reward_per_run)))
        crashes_per_episode.append(float(np.mean(num_crashes_per_run)))
        safety_per_episode.append(1.0 - float(np.mean(num_crashes_per_run)))
        min_clearance_per_episode.append(float(np.mean(min_clearance_per_run)))
        mismatch_per_episode.append(float(np.mean(max_mismatch_per_run)))
        cumulative_reward_runs_per_episode.append(np.asarray(cumulative_reward_per_run, dtype=np.float32))
        mismatch_runs_per_episode.append(np.asarray(max_mismatch_per_run, dtype=np.float32))
        qj_per_episode.append(float(qj))
        radius_per_episode.append(float(radius))
        run_logs_per_episode.append(logs[0]["position"])
        print(
            f"Cum rew: {cumulative_reward_per_episode[-1]} "
            f"Crash rate: {crashes_per_episode[-1]} "
            f"Min clearance: {min_clearance_per_episode[-1]} "
            f"Max mismatch: {mismatch_per_episode[-1]}"
        )

    metrics_path = os.path.join(experiment_dir, "conformal_obstacles_metrics.npz")
    np.savez(
        metrics_path,
        episodes=np.arange(args.num_episodes),
        qj_per_episode=np.asarray(qj_per_episode, dtype=np.float32),
        radius_per_episode=np.asarray(radius_per_episode, dtype=np.float32),
        crashes_per_episode=np.asarray(crashes_per_episode, dtype=np.float32),
        safety_per_episode=np.asarray(safety_per_episode, dtype=np.float32),
        cumulative_reward_per_episode=np.asarray(cumulative_reward_per_episode, dtype=np.float32),
        cumulative_reward_per_run=np.asarray(cumulative_reward_runs_per_episode, dtype=np.float32),
        min_clearance_per_episode=np.asarray(min_clearance_per_episode, dtype=np.float32),
        mismatch_per_episode=np.asarray(mismatch_per_episode, dtype=np.float32),
        mismatch_per_run=np.asarray(mismatch_runs_per_episode, dtype=np.float32),
        solo_positions_first_run=np.asarray(run_logs_per_episode, dtype=np.float32),
        alpha=args.alpha,
        delta=args.delta,
        bar_alpha=alpha,
        cbf_k0=CBF_K0,
        cbf_k1=CBF_K1,
    )
    print(f"[conformal_obstacles] Saved per-episode metrics to {metrics_path}")

    env.close()

    if len(video_frames) > 0:
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_dir, video_file))
        print(f"[conformal_obstacles] Video saved to {final_path}")

    print("Number of total crashes:", int(np.sum(np.asarray(crashes_per_episode) * args.num_eval_trajs)))


if __name__ == "__main__":
    main()
