#!/usr/bin/env python3
"""
Conformal-style joint CBF evaluation for multi-agent patrol.

Each agent runs the same solo policy. A joint CBF filter (joint_cbf_utils) modifies
the stacked actions for all agents simultaneously. The conformal radius is updated
from trajectory-level mismatch between the CBF next-state model and the realized
post-step simulator state on the concatenated full swarm state.
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

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.conformal_utils import explicit_radius_update, get_alpha_bar
from project_utils.joint_cbf_utils import (
    CBF_K0,
    CBF_K1,
    CBF_K4,
    CBF_RELINEARIZATION_PASSES,
    CBF_SLACK_WEIGHT,
    apply_cbf_filter,
    cbf_dynamics,
)
from project_utils.restart_utils import extract_positions_velocities, set_global_seed
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
    parser.add_argument("--seed", type=int, default=42, help="Seed used to generate the canonical first-reset patrol layout.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Desired probability of conformal error.")
    parser.add_argument("--delta", type=float, default=0.1, help="Desired probability of a bad draw.")
    parser.add_argument("--video_name", default="conformal_joint_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--num_eval_trajs", type=int, default=100)
    parser.add_argument("--num_agents", type=int, default=8, help="Number of patrol agents.")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--separation_radius", type=float, default=0.5, help="Desired pairwise separation distance enforced by CBF.")
    parser.add_argument("--gamma", type=float, default=0.8, help="CBF gamma in (0, 1].")
    parser.add_argument("--disable_boundary_collision", action="store_true", help="Move room boundaries far enough to effectively disable wall/ceiling/floor collisions.")
    parser.add_argument("--use_downwash", action="store_true", help="Enable simulator downwash in the rollout environment.")
    parser.add_argument(
        "--environment_layout",
        default=None,
        help="Optional path to a canonical joint environment JSON with saved start_points and goal_pairs.",
    )
    parser.add_argument("--spawn_ball_radius", type=float, default=1.0, help="Radius of the 3D ball used to resample each quad around its saved canonical patrol start.")
    parser.add_argument("--spawn_ball_max_tries", type=int, default=1000, help="Maximum number of joint ball-spawn attempts before failing.")
    parser.add_argument("--policy_refresh_interval", type=int, default=1, help="Call the policy every N control steps (N=1 keeps existing behavior).")
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


def _make_yaw_towards_goal_rotation(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    direction_xy = np.asarray(goal[:2] - pos[:2], dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction_xy))
    if direction_norm <= 1e-9:
        return np.eye(3, dtype=np.float64)

    x_axis = np.array([direction_xy[0] / direction_norm, direction_xy[1] / direction_norm, 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-9:
        return np.eye(3, dtype=np.float64)
    y_axis /= y_norm
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)


def _sample_point_in_ball(center: np.ndarray, radius: float) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    if radius <= 0.0:
        return center.copy()
    direction = np.random.normal(size=3)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        return center.copy()
    direction = direction / direction_norm
    distance = radius * (np.random.uniform(0.0, 1.0) ** (1.0 / 3.0))
    return center + distance * direction


def _capture_initial_patrol_layout(env) -> Dict[str, np.ndarray]:
    scenario = env.unwrapped.scenario
    goal_pairs = np.asarray(scenario.goal_pairs, dtype=np.float64).copy()
    start_points = np.asarray(scenario.start_points, dtype=np.float64).copy()
    if goal_pairs.ndim != 3 or goal_pairs.shape[1:] != (2, 3):
        raise ValueError(f"Unexpected goal_pairs shape: {goal_pairs.shape}")
    if start_points.ndim != 2 or start_points.shape[1] != 3:
        raise ValueError(f"Unexpected start_points shape: {start_points.shape}")
    if goal_pairs.shape[0] != start_points.shape[0]:
        raise ValueError("Saved patrol goal_pairs and start_points disagree on agent count")
    return {
        "goal_pairs": goal_pairs,
        "start_points": start_points,
    }


def _load_patrol_layout(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    goal_pairs = np.asarray(data["goal_pairs"], dtype=np.float64)
    start_points = np.asarray(data["start_points"], dtype=np.float64)
    if goal_pairs.ndim != 3 or goal_pairs.shape[1:] != (2, 3):
        raise ValueError(f"Saved goal_pairs must have shape (num_agents, 2, 3), got {goal_pairs.shape}")
    if start_points.ndim != 2 or start_points.shape[1] != 3:
        raise ValueError(f"Saved start_points must have shape (num_agents, 3), got {start_points.shape}")
    if goal_pairs.shape[0] != start_points.shape[0]:
        raise ValueError("Saved patrol goal_pairs and start_points disagree on agent count")
    return {
        "goal_pairs": goal_pairs,
        "start_points": start_points,
    }


def _serializable_patrol_layout(saved_layout: Dict[str, np.ndarray]) -> Dict[str, object]:
    goal_pairs = np.asarray(saved_layout["goal_pairs"], dtype=np.float64)
    start_points = np.asarray(saved_layout["start_points"], dtype=np.float64)
    return {
        "num_agents": int(start_points.shape[0]),
        "start_points": start_points.tolist(),
        "goal_pairs": goal_pairs.tolist(),
        "initial_goals": goal_pairs[:, 0, :].tolist(),
        "alternate_goals": goal_pairs[:, 1, :].tolist(),
    }


def _sample_joint_ball_spawn_positions(
    saved_start_points: np.ndarray,
    radius: float,
    min_pairwise_distance: float,
    max_tries: int,
) -> np.ndarray:
    saved_start_points = np.asarray(saved_start_points, dtype=np.float64)
    radius = float(radius)
    min_pairwise_distance = float(min_pairwise_distance)
    if saved_start_points.ndim != 2 or saved_start_points.shape[1] != 3:
        raise ValueError(f"saved_start_points must have shape (num_agents, 3), got {saved_start_points.shape}")
    if radius < 0.0:
        raise ValueError("spawn ball radius must be non-negative")
    if max_tries <= 0:
        raise ValueError("spawn_ball_max_tries must be positive")

    if radius == 0.0:
        return saved_start_points.copy()

    num_agents = saved_start_points.shape[0]
    for _ in range(int(max_tries)):
        sampled = np.zeros_like(saved_start_points, dtype=np.float64)
        valid = True
        for agent_id in range(num_agents):
            sampled[agent_id] = _sample_point_in_ball(saved_start_points[agent_id], radius)
            for other_id in range(agent_id):
                if np.linalg.norm(sampled[agent_id] - sampled[other_id]) < min_pairwise_distance:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return sampled

    raise RuntimeError(
        "Failed to sample pairwise-valid joint spawn positions inside the requested balls. "
        "Try reducing --spawn_ball_radius or increasing --spawn_ball_max_tries."
    )


def _apply_initialized_patrol_environment(
    env,
    saved_layout: Dict[str, np.ndarray],
    spawn_ball_radius: float,
    spawn_ball_max_tries: int,
    min_pairwise_distance: float,
) -> np.ndarray:
    env_unwrapped = env.unwrapped
    scenario = env_unwrapped.scenario
    goal_pairs = np.asarray(saved_layout["goal_pairs"], dtype=np.float64).copy()
    start_points = np.asarray(saved_layout["start_points"], dtype=np.float64).copy()

    if goal_pairs.shape != (len(env_unwrapped.envs), 2, 3):
        raise ValueError(
            f"Saved goal_pairs shape {goal_pairs.shape} does not match expected ({len(env_unwrapped.envs)}, 2, 3)"
        )
    if start_points.shape != (len(env_unwrapped.envs), 3):
        raise ValueError(
            f"Saved start_points shape {start_points.shape} does not match expected ({len(env_unwrapped.envs)}, 3)"
        )

    sampled_positions = _sample_joint_ball_spawn_positions(
        start_points,
        spawn_ball_radius,
        min_pairwise_distance,
        spawn_ball_max_tries,
    )

    scenario.goal_pairs = goal_pairs.copy()
    scenario.start_points = start_points.copy()
    scenario.active_goal_index = np.zeros(len(env_unwrapped.envs), dtype=np.int64)
    scenario.steps_since_switch = np.zeros(len(env_unwrapped.envs), dtype=np.int64)
    scenario.goals = goal_pairs[:, 0].copy()
    scenario.spawn_points = sampled_positions.copy()

    for agent_id, quad in enumerate(env_unwrapped.envs):
        goal = goal_pairs[agent_id, 0].copy()
        pos = sampled_positions[agent_id].copy()
        vel = np.zeros(3, dtype=np.float64)
        omega = np.zeros(3, dtype=np.float64)
        rotation = _make_yaw_towards_goal_rotation(pos, goal)

        quad.goal = goal.copy()
        quad.spawn_point = pos.copy()
        quad.dynamics.set_state(pos, vel, rotation, omega)
        quad.dynamics.reset()
        quad.dynamics.on_floor = False
        quad.dynamics.crashed_floor = False
        quad.dynamics.crashed_wall = False
        quad.dynamics.crashed_ceiling = False
        quad.tick = 0
        quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]

        env_unwrapped.pos[agent_id, :] = quad.dynamics.pos
        env_unwrapped.vel[agent_id, :] = quad.dynamics.vel

    obs = [quad.state_vector(quad) for quad in env_unwrapped.envs]
    if env_unwrapped.num_use_neighbor_obs > 0:
        obs = env_unwrapped.add_neighborhood_obs(obs)
    return np.asarray(obs, dtype=np.float32)


def _unpack_step_result(step_result):
    if not isinstance(step_result, tuple):
        raise TypeError(f"Expected step() to return tuple, got {type(step_result)!r}")
    if len(step_result) == 5:
        obs, rewards, terminated, truncated, infos = step_result
        dones = np.logical_or(terminated, truncated)
    elif len(step_result) == 4:
        obs, rewards, dones, infos = step_result
    else:
        raise ValueError(f"Unexpected step() return length: {len(step_result)}")
    return obs, rewards, np.asarray(dones), infos


def _joint_cbf_predicted_next_state(actions: np.ndarray, env_unwrapped) -> np.ndarray:
    """
    Predict the next concatenated swarm state under the simplified CBF model.
    """
    actions = np.asarray(actions, dtype=np.float64).copy()
    num_agents = len(env_unwrapped.envs)
    if actions.shape != (num_agents, 4):
        raise ValueError(f"actions shape {actions.shape} does not match expected ({num_agents}, 4)")

    dt = float(env_unwrapped.control_dt)
    predicted_states = []
    for agent_id, quad in enumerate(env_unwrapped.envs):
        normalized = np.clip(0.5 * (actions[agent_id] + 1.0), 0.0, 1.0)
        predicted_states.append(_pack_state_tuple(*cbf_dynamics(normalized, quad.dynamics, dt)))
    return np.concatenate(predicted_states, axis=0)


def _joint_actual_swarm_state(env_unwrapped) -> np.ndarray:
    """
    Read the realized post-step simulator state in the same packed layout used by the CBF prediction.
    """
    actual_states = []
    for quad in env_unwrapped.envs:
        dynamics = quad.dynamics
        actual_states.append(_pack_state_tuple(dynamics.pos, dynamics.vel, dynamics.rot, dynamics.omega))
    return np.concatenate(actual_states, axis=0)


def make_joint_cbf_filter(r_mismatch: float, separation_radius: float, gamma: float):
    def _filter(base_action: np.ndarray, env_state):
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_state.unwrapped,
            r=float(r_mismatch),
            separation_radius=float(separation_radius),
            gamma=float(gamma),
            use_repeated_linearization=True,
        )

    return _filter


def conformal_qj(logs: List[Dict[str, np.ndarray]], alpha: float, episode_length: int) -> float:
    """
    Trajectory-level score is max full-state mismatch across timesteps.
    """
    scores = []
    for run_log in logs:
        max_mismatch = 0.0
        mismatch = run_log.get("model_mismatch_state", np.array([]))
        mismatch_arr = np.asarray(mismatch)
        if mismatch_arr.ndim == 0:
            max_mismatch = float(mismatch_arr)
        else:
            for step in range(min(episode_length, len(mismatch_arr))):
                max_mismatch = max(max_mismatch, float(mismatch_arr[step]))
        scores.append(max_mismatch)
    if len(scores) == 0:
        return 0.0
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    idx = int(np.ceil(len(scores) * (1 - alpha)) - 1)
    idx = int(np.clip(idx, 0, len(scores) - 1))
    return float(scores[idx])


def run_joint_agents(
    env,
    solo_actor,
    init_rnn_states,
    solo_obs_dim,
    joint_action_fn,
    num_agents: int,
    saved_layout: Dict[str, np.ndarray],
    spawn_ball_radius: float,
    spawn_ball_max_tries: int,
    min_spawn_pairwise_distance: float,
    max_steps=1500,
    num_runs=1,
    deterministic=False,
    policy_refresh_interval: int = 1,
    disable_boundary_collision=False,
    boundary_far_distance: float = COLLISION_FAR_DISTANCE,
    separation_radius: float | None = None,
    min_collision_distance: float = 0.0,
    return_first_run_trajectory=False,
):
    """
    Run environment for max_steps and return run-level summaries.
    """
    logs = [None] * num_runs
    policy_refresh_interval = int(policy_refresh_interval)
    if policy_refresh_interval < 1:
        raise ValueError("--policy_refresh_interval must be >= 1")
    running_max_mismatch = 0.0
    progress_bar = tqdm(range(num_runs))

    for run_idx in progress_bar:
        if disable_boundary_collision:
            _configure_far_boundary_geometry(env.unwrapped, boundary_far_distance)
        env.reset()
        if disable_boundary_collision:
            _configure_far_boundary_geometry(env.unwrapped, boundary_far_distance)
        obs_run = _apply_initialized_patrol_environment(
            env,
            saved_layout,
            spawn_ball_radius,
            spawn_ball_max_tries,
            min_spawn_pairwise_distance,
        )
        done = False
        step_num = 0
        sep_radius = float(separation_radius) if separation_radius is not None else 0.0
        run_rnn_states = init_rnn_states.clone()
        cached_nominal_action = None
        cumulative_reward = 0.0
        nonswap_steps = 0
        run_max_mismatch = 0.0
        crash_indicator = 0
        h_min = float("inf")
        pairwise_min_dist = float("inf")
        prev_goal_dist = None
        run_trajectory_positions = []
        return_trajectory = bool(return_first_run_trajectory) and run_idx == 0

        scenario = env.unwrapped.scenario
        goals = []
        for agent_id in range(num_agents):
            active = scenario.active_goal_index[agent_id]
            target = scenario.goal_pairs[agent_id, active]
            goals.append(np.asarray(target, dtype=np.float64).copy())

        while not done and step_num < max_steps:
            if (step_num % policy_refresh_interval == 0) or cached_nominal_action is None:
                obs_self = obs_run[:, :solo_obs_dim]
                obs_dict = {OBS_KEY: obs_self}
                with torch.no_grad():
                    normalized_obs = prepare_and_normalize_obs(solo_actor, obs_dict)
                    policy_out = solo_actor(normalized_obs, run_rnn_states)
                run_rnn_states = policy_out["new_rnn_states"]
                cached_nominal_action = policy_out["actions"]
                if deterministic:
                    cached_nominal_action = argmax_actions(solo_actor.action_distribution())
                if cached_nominal_action.dim() == 1:
                    cached_nominal_action = cached_nominal_action.unsqueeze(-1)
                cached_nominal_action = cached_nominal_action.detach().cpu().numpy()

            actions = cached_nominal_action

            actions = joint_action_fn(base_action=actions, env_state=env)
            predicted_next = _joint_cbf_predicted_next_state(actions, env.unwrapped)

            obs_run, rewards, dones, infos = _unpack_step_result(env.step(actions))
            actual_next = _joint_actual_swarm_state(env.unwrapped)
            pos_vel = True # TODO: integrate better
            dt = env.unwrapped.control_dt
            if pos_vel:
                # Just positions and velocities
                proj_posvel = np.concatenate([[1.0] * 6 + [0.0] * 12] * num_agents)
            else:
                proj_posvel = np.array([1.0] * (18 * num_agents))
            mismatch = float(np.linalg.norm(proj_posvel * (predicted_next - actual_next))) / dt
            run_max_mismatch = max(run_max_mismatch, mismatch)
            obs_run = np.array(obs_run, dtype=np.float32)
            pos, vel = extract_positions_velocities(env.unwrapped)
            step_h = _pairwise_h_min(pos, sep_radius)
            h_min = min(h_min, float(step_h))
            crash_indicator = max(crash_indicator, _collision_indicator_from_positions(pos, min_collision_distance))
            pairwise_min_dist = min(pairwise_min_dist, _pairwise_min_dist(pos))
            if return_trajectory:
                run_trajectory_positions.append(pos.copy())

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

            step_goal_dist = np.asarray(step_goal_dist, dtype=np.float64)
            if prev_goal_dist is None:
                prev_goal_dist = step_goal_dist.copy()
            else:
                for agent_id in range(num_agents):
                    if not bool(step_goal_swap[agent_id]):
                        nonswap_steps += 1
                        delta = float(prev_goal_dist[agent_id] - step_goal_dist[agent_id])
                        if delta > 0.0:
                            cumulative_reward += delta
                prev_goal_dist = step_goal_dist

            done = np.all(dones)
            step_num += 1

        if nonswap_steps > 0:
            cumulative_reward = cumulative_reward / nonswap_steps * max(1, step_num - 1)
        has_rollout_data = step_num > 0

        run_logs = {
            "model_mismatch_state": np.float64(run_max_mismatch),
            "cumulative_reward": float(cumulative_reward),
            "crash_indicator": int(crash_indicator),
            "quad_crash_flag": int(crash_indicator),
            "pairwise_min_dist": float(pairwise_min_dist),
            "h_violation": 1.0 if has_rollout_data and h_min <= 0.0 else 0.0,
            "h_min": float(h_min),
        }
        if return_trajectory:
            run_logs["positions"] = np.asarray(run_trajectory_positions, dtype=np.float32)

        if step_num > 0:
            running_max_mismatch = max(running_max_mismatch, run_max_mismatch)
        progress_bar.set_postfix_str(f"max mismatch={running_max_mismatch:.6f}")
        logs[run_idx] = run_logs

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


def _pairwise_min_dist(positions_t: np.ndarray) -> float:
    min_dist = float("inf")
    for i, j in combinations(range(positions_t.shape[0]), 2):
        dist = float(np.linalg.norm(positions_t[i] - positions_t[j]))
        min_dist = min(min_dist, dist)
    return min_dist


def main() -> None:
    args = parse_args()
    if args.num_agents < 2:
        raise ValueError("--num_agents must be >= 2 for joint CBF.")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must satisfy 0 < gamma <= 1")
    if args.separation_radius <= 0.0:
        raise ValueError("--separation_radius must be > 0")
    if args.policy_refresh_interval < 1:
        raise ValueError("--policy_refresh_interval must be >= 1")
    if args.spawn_ball_radius < 0.0:
        raise ValueError("--spawn_ball_radius must be non-negative")
    if args.spawn_ball_max_tries <= 0:
        raise ValueError("--spawn_ball_max_tries must be positive")

    torch.set_grad_enabled(False)
    register_swarm_components()
    set_global_seed(args.seed)

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
        f"--quads_use_downwash={args.use_downwash}",
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
    arm_len = env.quad_arm
    min_r = arm_len * 2.5
    min_spawn_pairwise_distance = min_r + float(args.separation_radius)

    if args.environment_layout is not None:
        saved_layout = _load_patrol_layout(args.environment_layout)
        if int(saved_layout["start_points"].shape[0]) != args.num_agents:
            raise ValueError("--environment_layout agent count does not match --num_agents")
    else:
        if args.disable_boundary_collision:
            _configure_far_boundary_geometry(env.unwrapped)
        env.reset()
        if args.disable_boundary_collision:
            _configure_far_boundary_geometry(env.unwrapped)
        saved_layout = _capture_initial_patrol_layout(env)
    patrol_json_path = os.path.join(experiment_dir, "conformal_joint_environment.json")
    with open(patrol_json_path, "w", encoding="utf-8") as f:
        json.dump(_serializable_patrol_layout(saved_layout), f, indent=2)
    print(f"[conformal_joint] Saved canonical patrol layout to {patrol_json_path}")

    alpha = get_alpha_bar(args.alpha, args.delta, args.num_trajectories)
    r_mismatch = 0.0
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

    print("EPISODE", 0)

    solo_rnn_states = torch.zeros((args.num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
    cal_logs = run_joint_agents(
        env,
        solo_actor,
        solo_rnn_states,
        solo_obs_dim,
        filter_fn,
        num_agents=args.num_agents,
        saved_layout=saved_layout,
        spawn_ball_radius=args.spawn_ball_radius,
        spawn_ball_max_tries=args.spawn_ball_max_tries,
        min_spawn_pairwise_distance=min_spawn_pairwise_distance,
        max_steps=args.episode_length,
        num_runs=args.num_trajectories,
        deterministic=args.deterministic,
        policy_refresh_interval=args.policy_refresh_interval,
        disable_boundary_collision=args.disable_boundary_collision,
        separation_radius=args.separation_radius,
        min_collision_distance=min_r,
    )
    qj = conformal_qj(cal_logs, alpha, args.episode_length)
    # Do not update rj or filter

    solo_rnn_states = torch.zeros((args.num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
    logs = run_joint_agents(
        env,
        solo_actor,
        solo_rnn_states,
        solo_obs_dim,
        filter_fn,
        num_agents=args.num_agents,
        saved_layout=saved_layout,
        spawn_ball_radius=args.spawn_ball_radius,
        spawn_ball_max_tries=args.spawn_ball_max_tries,
        min_spawn_pairwise_distance=min_spawn_pairwise_distance,
        max_steps=args.episode_length,
        num_runs=args.num_eval_trajs,
        deterministic=args.deterministic,
        policy_refresh_interval=args.policy_refresh_interval,
        disable_boundary_collision=args.disable_boundary_collision,
        separation_radius=args.separation_radius,
        min_collision_distance=min_r,
        return_first_run_trajectory=True,
    )
    cumulative_reward_per_run = []
    crash_indicator_per_run = []
    max_mismatch_per_run = []
    h_violation_per_run = []
    h_min_per_run = []
    pairwise_min_dist_per_run = []
    quad_crash_flags = []
    for run_id in range(args.num_eval_trajs):
        run = logs[run_id]
        cumulative_reward_per_run.append(float(run["cumulative_reward"]))
        crash_indicator_per_run.append(float(run["crash_indicator"]))
        quad_crash_flags.append(float(run["quad_crash_flag"]))
        max_mismatch_per_run.append(float(run["model_mismatch_state"]))
        h_violation_per_run.append(float(run["h_violation"]))
        h_min_per_run.append(float(run["h_min"]))
        pairwise_min_dist_per_run.append(float(run["pairwise_min_dist"]))

    episode_min_pair_dist = float(np.min(np.asarray(pairwise_min_dist_per_run, dtype=np.float32)))
    episode_quad_crashes = int(np.sum(np.asarray(quad_crash_flags, dtype=np.float32)))
    episode_h_violations = int(np.sum(np.asarray(h_violation_per_run, dtype=np.float32)))

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
        f"Crash rate: {crashes_per_episode[-1]} "
        f"Episode min pairwise dist: {episode_min_pair_dist} "
        f"Quad-crash runs: {episode_quad_crashes}/{args.num_eval_trajs} "
        f"H-violation runs: {episode_h_violations}/{args.num_eval_trajs}"
    )

    metrics_path = os.path.join(experiment_dir, "conformal_joint_metrics.npz")
    np.savez(
        metrics_path,
        episodes=np.arange(1),
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
        cbf_k4=CBF_K4,
        cbf_relinearization_passes=CBF_RELINEARIZATION_PASSES,
        cbf_slack_weight=CBF_SLACK_WEIGHT,
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
