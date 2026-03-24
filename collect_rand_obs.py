#!/usr/bin/env python3
"""Collect fixed-radius obstacle-avoidance rollouts for plotting."""

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
from sample_factory.model.model_utils import get_rnn_size

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.cbf_utils import apply_cbf_filter, cbf_dynamics, real_dynamics
from project_utils.restart_utils import extract_positions_velocities, set_global_seed
from project_utils.utils import OBS_KEY, load_actor, load_cfg, latest_checkpoint


DEVICE = torch.device("cpu")
COLLISION_FAR_DISTANCE = 10000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fixed-radius obstacle trajectories.")
    parser.add_argument(
        "--r_mismatch",
        type=float,
        required=True,
        help="Conformal mismatch radius to pass into the obstacle filter.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=100,
        help="Number of trajectories to collect.",
    )
    parser.add_argument(
        "--output_path",
        default="rand_obs_trajectories.npz",
        help="Path to save the collected rollout dataset.",
    )
    parser.add_argument(
        "--conformal_obstacles_environment",
        required=True,
        help="Path to geometry JSON containing authoritative start/goal/obstacle data.",
    )
    parser.add_argument(
        "--conf_rand_obs_args",
        required=True,
        help="Path to conf_rand_obs args JSON used to rebuild policy/runtime settings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used once before rollout collection (not reset per trajectory).",
    )
    parser.add_argument(
        "--point_towards_goal",
        action="store_true",
        help="If set, initialize each trajectory with a level yaw pointing toward the goal.",
    )
    parser.add_argument(
        "--disable_4step_sampled",
        dest="use_4step_sampled",
        action="store_false",
        help="Disable the 4-step sampled CBF refinement that is enabled by default.",
    )
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=1,
        help="Hold each chosen action fixed for this many environment timesteps before recomputing it.",
    )
    parser.set_defaults(use_4step_sampled=True)
    return parser.parse_args()


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_environment_geometry(path: str) -> Dict[str, np.ndarray | float]:
    data = _load_json(path)
    start_point = np.asarray(data["start_point"], dtype=np.float64)
    goal_point = np.asarray(data["goal_point"], dtype=np.float64)
    obstacle_positions = np.asarray(data.get("obstacles", []), dtype=np.float64).reshape(-1, 3)
    obstacle_radius = float(data["radius"])

    if start_point.shape != (3,):
        raise ValueError("Environment start_point must have shape (3,).")
    if goal_point.shape != (3,):
        raise ValueError("Environment goal_point must have shape (3,).")
    if obstacle_positions.ndim != 2 or obstacle_positions.shape[1] != 3:
        raise ValueError("Environment obstacles must have shape (num_obstacles, 3).")
    if obstacle_radius < 0.0:
        raise ValueError("Environment radius must be non-negative.")

    return {
        "start_point": start_point,
        "goal_point": goal_point,
        "obstacle_positions": obstacle_positions,
        "obstacle_radius": obstacle_radius,
    }


def _configure_far_boundary_geometry(env_unwrapped, far_distance: float = COLLISION_FAR_DISTANCE) -> None:
    far_distance = float(far_distance)
    if far_distance <= 0.0:
        raise ValueError("--disable_boundary_collision requires a positive far distance.")
    far_box = np.array(
        [
            [-far_distance, -far_distance, -far_distance],
            [far_distance, far_distance, far_distance],
        ]
    )
    env_unwrapped.room_box = far_box.copy()
    for quad in env_unwrapped.envs:
        quad.room_box = far_box.copy()
        quad.dynamics.room_box = far_box.copy()
        quad.dynamics.floor_threshold = -far_distance


def _pack_state_tuple(pos: np.ndarray, vel: np.ndarray, rot: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.concatenate([pos.reshape(-1), vel.reshape(-1), rot.reshape(-1), omega.reshape(-1)], axis=0).astype(np.float64)


def make_obstacle_cbf_filter(
    r_mismatch: float,
    gamma: float,
    obstacle_radius_margin: float,
    use_4step_sampled: bool,
):
    def _filter(base_action: np.ndarray, env_state, _unused_swarm_state=None):
        env_unwrapped = env_state.unwrapped
        if not getattr(env_unwrapped, "use_obstacles", False) or env_unwrapped.obstacles is None:
            return np.asarray(base_action, dtype=np.float32)

        obstacle_centers = np.asarray(env_unwrapped.obstacles.pos_arr, dtype=np.float64).reshape(-1, 3)
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
        cbf_obstacle_radius = (
            float(env_unwrapped.obstacles.obstacle_radius)
            + float(env_unwrapped.quad_arm)
            + float(obstacle_radius_margin)
        )
        obstacles = [
            {
                "position": np.asarray(center, dtype=np.float64),
                "velocity": np.zeros(3, dtype=np.float64),
                "radius": cbf_obstacle_radius,
            }
            for center in obstacle_centers
        ]
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_unwrapped,
            quad_state=quad_state,
            obstacles=obstacles,
            r=float(r_mismatch),
            gamma=float(gamma),
            use_4step_sampled=use_4step_sampled,
        )

    return _filter


def _state_model_mismatch(action: np.ndarray, dynamics, dt: float) -> float:
    action = np.asarray(action, dtype=np.float64)
    normalized = np.clip(0.5 * (action + 1.0), 0.0, 1.0)
    state_cbf = _pack_state_tuple(*cbf_dynamics(normalized, dynamics, dt))
    state_real = _pack_state_tuple(*real_dynamics(normalized, dynamics, dt))
    return float(np.linalg.norm(state_cbf - state_real))


def _reset_env(env) -> np.ndarray:
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result
    return np.asarray(obs, dtype=np.float32)


def _step_env(env, actions: np.ndarray):
    step_result = env.step(actions)
    if not isinstance(step_result, tuple):
        raise TypeError(f"Expected env.step() to return tuple, got {type(step_result)!r}")
    if len(step_result) == 5:
        obs, rewards, terminated, truncated, infos = step_result
        dones = np.logical_or(np.asarray(terminated), np.asarray(truncated))
    elif len(step_result) == 4:
        obs, rewards, dones, infos = step_result
        dones = np.asarray(dones)
    else:
        raise ValueError(f"Unexpected env.step() return length: {len(step_result)}")
    return np.asarray(obs, dtype=np.float32), rewards, dones, infos


def _make_yaw_towards_goal_rotation(pos: np.ndarray, goal: np.ndarray, fallback_rot: np.ndarray) -> np.ndarray:
    direction_xy = np.asarray(goal[:2] - pos[:2], dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction_xy))
    if direction_norm <= 1e-9:
        return np.asarray(fallback_rot, dtype=np.float64).copy()

    x_axis = np.array([direction_xy[0] / direction_norm, direction_xy[1] / direction_norm, 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-9:
        return np.asarray(fallback_rot, dtype=np.float64).copy()
    y_axis /= y_norm
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)


def _apply_authoritative_obstacle_environment(env, geometry: Dict[str, np.ndarray | float], point_towards_goal: bool):
    env_unwrapped = env.unwrapped
    if not getattr(env_unwrapped, "use_obstacles", False):
        raise ValueError("collect_rand_obs requires an obstacle-enabled environment.")
    if len(env_unwrapped.envs) != 1:
        raise ValueError("collect_rand_obs currently supports exactly one quadrotor.")

    quad = env_unwrapped.envs[0]
    dynamics = quad.dynamics

    pos = np.asarray(geometry["start_point"], dtype=np.float64).copy()
    vel = np.zeros(3, dtype=np.float64)
    omega = np.zeros(3, dtype=np.float64)
    goal = np.asarray(geometry["goal_point"], dtype=np.float64).copy()

    reset_rotation = np.asarray(dynamics.rot, dtype=np.float64).copy()
    rotation = reset_rotation.copy()
    if point_towards_goal:
        rotation = _make_yaw_towards_goal_rotation(pos, goal, reset_rotation)

    quad.goal = goal.copy()
    quad.spawn_point = pos.copy()
    dynamics.set_state(pos, vel, rotation, omega)
    dynamics.reset()
    dynamics.on_floor = False
    dynamics.crashed_floor = False
    dynamics.crashed_wall = False
    dynamics.crashed_ceiling = False
    quad.tick = 0
    quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]

    env_unwrapped.pos[0, :] = dynamics.pos
    env_unwrapped.vel[0, :] = dynamics.vel

    obstacle_positions = np.asarray(geometry["obstacle_positions"], dtype=np.float64).reshape(-1, 3)
    obstacle_radius = float(geometry["obstacle_radius"])
    env_unwrapped.num_obstacles = int(obstacle_positions.shape[0])
    env_unwrapped.obst_size = 2.0 * obstacle_radius
    env_unwrapped.obstacles.pos_arr = obstacle_positions.copy()
    env_unwrapped.obstacles.obstacle_radius = obstacle_radius
    env_unwrapped.obstacles.size = 2.0 * obstacle_radius

    obs = [quad.state_vector(quad)]
    if env_unwrapped.num_use_neighbor_obs > 0:
        obs = env_unwrapped.add_neighborhood_obs(obs)
    if env_unwrapped.use_obstacles:
        obs = env_unwrapped.obstacles.step(obs=obs, quads_pos=env_unwrapped.pos)

    metadata = {
        "initial_position": pos,
        "initial_goal": goal,
        "initial_velocity": vel,
        "initial_omega": omega,
        "initial_rotation": rotation.copy(),
    }
    return np.asarray(obs, dtype=np.float32), metadata


def _pad_with_fill(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.shape[0] >= target_len:
        return values[:target_len]
    pad_len = target_len - values.shape[0]
    pad_shape = (pad_len,) + values.shape[1:]
    if np.issubdtype(values.dtype, np.bool_):
        pad_value = False
    elif np.issubdtype(values.dtype, np.integer):
        pad_value = 0
    else:
        pad_value = np.nan
    pad = np.full(pad_shape, pad_value, dtype=values.dtype)
    return np.concatenate([values, pad], axis=0)


def _build_eval_cfg(saved_args: Dict) -> "AttrDict":
    spawn_area = saved_args["quads_obst_spawn_area"]
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_use_obstacles=True",
        f"--quads_mode={saved_args['quads_mode']}",
        "--quads_num_agents=1",
        "--quads_neighbor_visible_num=0",
        "--quads_neighbor_obs_type=none",
        "--quads_obstacle_obs_type=octomap",
        f"--quads_obst_density={float(saved_args['quads_obst_density'])}",
        f"--quads_obst_size={float(saved_args['quads_obst_size'])}",
        "--quads_obst_spawn_area",
        str(float(spawn_area[0])),
        str(float(spawn_area[1])),
        "--quads_collision_reward=8.0",
        "--quads_collision_hitbox_radius=2.5",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=False",
    ]
    return parse_swarm_cfg(eval_cli, evaluation=True)


def _run_obstacle_trajectory(
    env,
    solo_actor,
    init_rnn_states: torch.Tensor,
    solo_obs_dim: int,
    solo_action_fn,
    obstacle_radius_margin: float,
    geometry: Dict[str, np.ndarray | float],
    max_steps: int,
    deterministic: bool,
    disable_boundary_collision: bool,
    point_towards_goal: bool,
    action_repeat: int,
):
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)

    _reset_env(env)
    obs_run, initial_state = _apply_authoritative_obstacle_environment(env, geometry, point_towards_goal)
    done = False
    step_num = 0
    run_rnn_states = init_rnn_states.clone()
    held_nominal_action = None
    held_filtered_action = None

    run_logs: Dict[str, List] = {
        "position": [],
        "velocity": [],
        "goal_dist": [],
        "collision_obstacle": [],
        "boundary_dist": [],
        "clearance": [],
        "cbf_clearance": [],
        "model_mismatch_state": [],
        "nominal_acceleration": [],
        "filtered_acceleration": [],
    }

    while not done and step_num < max_steps:
        recompute_action = held_filtered_action is None or (step_num % action_repeat == 0)
        if recompute_action:
            obs_solo_self = obs_run[0, :solo_obs_dim]
            obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
            with torch.no_grad():
                normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
                policy_solo = solo_actor(normalized_solo, run_rnn_states)
            run_rnn_states = policy_solo["new_rnn_states"]
            action_solo = policy_solo["actions"]
            if deterministic:
                action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]
            held_nominal_action = np.asarray(action_solo, dtype=np.float64).copy()
            held_filtered_action = np.asarray(
                solo_action_fn(
                    base_action=action_solo,
                    env_state=env,
                    _unused_swarm_state=None,
                ),
                dtype=np.float64,
            ).copy()

        nominal_action = np.asarray(held_nominal_action, dtype=np.float64).copy()

        dynamics = env.unwrapped.envs[0].dynamics
        dt = float(env.unwrapped.control_dt)
        nominal_norm = np.clip(0.5 * (nominal_action + 1.0), 0.0, 1.0)
        current_vel = np.asarray(dynamics.vel, dtype=np.float64).copy()
        _, nominal_next_vel, _, _ = cbf_dynamics(nominal_norm, dynamics, dt)
        nominal_acc = (np.asarray(nominal_next_vel, dtype=np.float64) - current_vel) / dt

        filtered_action = np.asarray(held_filtered_action, dtype=np.float64).copy()
        filtered_norm = np.clip(0.5 * (filtered_action + 1.0), 0.0, 1.0)
        _, filtered_next_vel, _, _ = cbf_dynamics(filtered_norm, dynamics, dt)
        filtered_acc = (np.asarray(filtered_next_vel, dtype=np.float64) - current_vel) / dt
        mismatch_state = _state_model_mismatch(
            filtered_action,
            dynamics,
            dt,
        )

        actions = filtered_action[None, :]
        obs_run, rewards, dones, infos = _step_env(env, actions)

        pos, vel = extract_positions_velocities(env.unwrapped)
        solo_pos = np.asarray(pos[0], dtype=np.float64)
        solo_vel = np.asarray(vel[0], dtype=np.float64)
        goal = np.asarray(env.unwrapped.envs[0].goal, dtype=np.float64)
        goal_dist = float(np.linalg.norm(solo_pos - goal))

        obstacle_centers = np.asarray(env.unwrapped.obstacles.pos_arr, dtype=np.float64).reshape(-1, 3)
        obstacle_radius = float(env.unwrapped.obstacles.obstacle_radius)
        quad_radius = float(env.unwrapped.obstacles.quad_radius)
        if obstacle_centers.size == 0:
            boundary_dist = float("inf")
            clearance = float("inf")
            cbf_clearance = float("inf")
        else:
            center_dists_xy = np.linalg.norm(obstacle_centers[:, :2] - solo_pos[None, :2], axis=1)
            boundary_dist = float(np.min(center_dists_xy - obstacle_radius))
            clearance = float(np.min(center_dists_xy - (obstacle_radius + quad_radius)))
            cbf_clearance = float(np.min(center_dists_xy - (obstacle_radius + quad_radius + obstacle_radius_margin)))

        rew_obst_raw = infos[0]["rewards"].get("rewraw_quadcol_obstacle", 0.0)
        collision_obstacle = float(rew_obst_raw) < 0.0

        run_logs["position"].append(solo_pos.copy())
        run_logs["velocity"].append(solo_vel.copy())
        run_logs["goal_dist"].append(goal_dist)
        run_logs["collision_obstacle"].append(collision_obstacle)
        run_logs["boundary_dist"].append(boundary_dist)
        run_logs["clearance"].append(clearance)
        run_logs["cbf_clearance"].append(cbf_clearance)
        run_logs["model_mismatch_state"].append(mismatch_state)
        run_logs["nominal_acceleration"].append(np.asarray(nominal_acc, dtype=np.float32))
        run_logs["filtered_acceleration"].append(np.asarray(filtered_acc, dtype=np.float32))

        done = bool(np.all(dones))
        step_num += 1

    if run_logs["position"]:
        position = np.asarray(run_logs["position"], dtype=np.float32)
        velocity = np.asarray(run_logs["velocity"], dtype=np.float32)
    else:
        position = np.empty((0, 3), dtype=np.float32)
        velocity = np.empty((0, 3), dtype=np.float32)

    return {
        "position": position,
        "velocity": velocity,
        "goal_dist": np.asarray(run_logs["goal_dist"], dtype=np.float32),
        "collision_obstacle": np.asarray(run_logs["collision_obstacle"], dtype=np.bool_),
        "boundary_dist": np.asarray(run_logs["boundary_dist"], dtype=np.float32),
        "clearance": np.asarray(run_logs["clearance"], dtype=np.float32),
        "cbf_clearance": np.asarray(run_logs["cbf_clearance"], dtype=np.float32),
        "model_mismatch_state": np.asarray(run_logs["model_mismatch_state"], dtype=np.float32),
        "nominal_acceleration": np.asarray(run_logs["nominal_acceleration"], dtype=np.float32),
        "filtered_acceleration": np.asarray(run_logs["filtered_acceleration"], dtype=np.float32),
        "initial_position": np.asarray(initial_state["initial_position"], dtype=np.float64),
        "initial_goal": np.asarray(initial_state["initial_goal"], dtype=np.float64),
        "initial_velocity": np.asarray(initial_state["initial_velocity"], dtype=np.float64),
        "initial_omega": np.asarray(initial_state["initial_omega"], dtype=np.float64),
        "initial_rotation": np.asarray(initial_state["initial_rotation"], dtype=np.float64),
        "trajectory_length": int(step_num),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.r_mismatch):
        raise ValueError("--r_mismatch must be positive.")
    if args.num_trajectories <= 0:
        raise ValueError("--num_trajectories must be positive.")
    if args.action_repeat <= 0:
        raise ValueError("--action_repeat must be positive.")

    geometry = _load_environment_geometry(args.conformal_obstacles_environment)
    saved_args = _load_json(args.conf_rand_obs_args)
    gamma = float(saved_args["gamma"])
    obstacle_radius_margin = float(saved_args["obstacle_radius_margin"])
    episode_length = int(saved_args["episode_length"])
    deterministic = bool(saved_args.get("deterministic", False))
    disable_boundary_collision = bool(saved_args.get("disable_boundary_collision", False))

    if not (0.0 < gamma <= 1.0):
        raise ValueError("Saved --gamma must satisfy 0 < gamma <= 1.")
    if episode_length <= 0:
        raise ValueError("Saved --episode_length must be positive.")

    torch.set_grad_enabled(False)
    register_swarm_components()
    set_global_seed(args.seed)

    cfg_solo = load_cfg(saved_args["solo_train_dir"], saved_args["solo_experiment"])
    eval_cfg = _build_eval_cfg(saved_args)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)

    solo_ckpt = latest_checkpoint(saved_args["solo_train_dir"], saved_args["solo_experiment"], policy_index=0)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
    filter_fn = make_obstacle_cbf_filter(
        args.r_mismatch,
        gamma,
        obstacle_radius_margin,
        use_4step_sampled=bool(args.use_4step_sampled),
    )

    trajectories = []
    progress_bar = tqdm(range(args.num_trajectories))
    for _ in progress_bar:
        traj = _run_obstacle_trajectory(
            env=env,
            solo_actor=solo_actor,
            init_rnn_states=solo_rnn_states,
            solo_obs_dim=solo_obs_dim,
            solo_action_fn=filter_fn,
            obstacle_radius_margin=obstacle_radius_margin,
            geometry=geometry,
            max_steps=episode_length,
            deterministic=deterministic,
            disable_boundary_collision=disable_boundary_collision,
            point_towards_goal=args.point_towards_goal,
            action_repeat=int(args.action_repeat),
        )
        trajectories.append(traj)
        progress_bar.set_postfix_str(f"last_len={traj['trajectory_length']}")

    max_len = max(traj["trajectory_length"] for traj in trajectories)
    positions = np.stack([_pad_with_fill(traj["position"], max_len) for traj in trajectories], axis=0)
    velocities = np.stack([_pad_with_fill(traj["velocity"], max_len) for traj in trajectories], axis=0)
    goal_dist = np.stack([_pad_with_fill(traj["goal_dist"], max_len) for traj in trajectories], axis=0)
    collision_obstacle = np.stack([_pad_with_fill(traj["collision_obstacle"], max_len) for traj in trajectories], axis=0)
    boundary_dist = np.stack([_pad_with_fill(traj["boundary_dist"], max_len) for traj in trajectories], axis=0)
    clearance = np.stack([_pad_with_fill(traj["clearance"], max_len) for traj in trajectories], axis=0)
    cbf_clearance = np.stack([_pad_with_fill(traj["cbf_clearance"], max_len) for traj in trajectories], axis=0)
    model_mismatch_state = np.stack([_pad_with_fill(traj["model_mismatch_state"], max_len) for traj in trajectories], axis=0)
    nominal_accelerations = np.stack([_pad_with_fill(traj["nominal_acceleration"], max_len) for traj in trajectories], axis=0)
    filtered_accelerations = np.stack([_pad_with_fill(traj["filtered_acceleration"], max_len) for traj in trajectories], axis=0)
    trajectory_lengths = np.asarray([traj["trajectory_length"] for traj in trajectories], dtype=np.int32)
    initial_positions = np.stack([traj["initial_position"] for traj in trajectories], axis=0)
    initial_goals = np.stack([traj["initial_goal"] for traj in trajectories], axis=0)
    initial_velocities = np.stack([traj["initial_velocity"] for traj in trajectories], axis=0)
    initial_omegas = np.stack([traj["initial_omega"] for traj in trajectories], axis=0)
    initial_rotations = np.stack([traj["initial_rotation"] for traj in trajectories], axis=0)

    output_path = os.path.abspath(args.output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    quad_radius = float(env.unwrapped.quad_arm)
    cbf_obstacle_radius = float(geometry["obstacle_radius"]) + quad_radius + obstacle_radius_margin

    np.savez_compressed(
        output_path,
        positions=positions,
        velocities=velocities,
        goal_dist=goal_dist,
        collision_obstacle=collision_obstacle,
        boundary_dist=boundary_dist,
        clearance=clearance,
        cbf_clearance=cbf_clearance,
        model_mismatch_state=model_mismatch_state,
        nominal_accelerations=nominal_accelerations,
        filtered_accelerations=filtered_accelerations,
        trajectory_lengths=trajectory_lengths,
        initial_positions=initial_positions,
        initial_goals=initial_goals,
        initial_velocities=initial_velocities,
        initial_omegas=initial_omegas,
        initial_rotations=initial_rotations,
        start_point=np.asarray(geometry["start_point"], dtype=np.float64),
        goal_point=np.asarray(geometry["goal_point"], dtype=np.float64),
        obstacle_positions=np.asarray(geometry["obstacle_positions"], dtype=np.float64),
        obstacle_radius=float(geometry["obstacle_radius"]),
        quad_radius=quad_radius,
        cbf_obstacle_radius=cbf_obstacle_radius,
        num_trajectories=args.num_trajectories,
        obstacle_count=int(np.asarray(geometry["obstacle_positions"]).shape[0]),
        r_mismatch=float(args.r_mismatch),
        seed=int(args.seed),
        episode_length=episode_length,
        gamma=gamma,
        deterministic=deterministic,
        disable_boundary_collision=disable_boundary_collision,
        obstacle_radius_margin=obstacle_radius_margin,
        point_towards_goal=bool(args.point_towards_goal),
        use_4step_sampled=bool(args.use_4step_sampled),
        action_repeat=int(args.action_repeat),
        conformal_obstacles_environment_path=os.path.abspath(args.conformal_obstacles_environment),
        conf_rand_obs_args_path=os.path.abspath(args.conf_rand_obs_args),
    )

    env.close()

    print(f"[collect_rand_obs] Saved {args.num_trajectories} trajectories to {output_path}")
    print(f"[collect_rand_obs] Max trajectory length: {max_len}")
    print(f"[collect_rand_obs] Obstacle count: {np.asarray(geometry['obstacle_positions']).shape[0]}")
    print(f"[collect_rand_obs] point_towards_goal={bool(args.point_towards_goal)}")
    print(f"[collect_rand_obs] use_4step_sampled={bool(args.use_4step_sampled)}")
    print(f"[collect_rand_obs] action_repeat={int(args.action_repeat)}")


if __name__ == "__main__":
    main()
