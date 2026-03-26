#!/usr/bin/env python3
"""Collect fixed-radius joint-trajectory rollouts for plotting."""

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

from project_utils.joint_cbf_utils import (
    _normalized_to_thrust,
    apply_cbf_filter,
    cbf_dynamics,
    real_dynamics,
)
from project_utils.restart_utils import extract_positions_velocities, set_global_seed
from project_utils.utils import OBS_KEY, load_actor, load_cfg, latest_checkpoint


DEVICE = torch.device("cpu")
COLLISION_FAR_DISTANCE = 10000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fixed-radius joint trajectories.")
    parser.add_argument(
        "--r_mismatch",
        type=float,
        required=True,
        help="Conformal mismatch radius to pass into the joint filter.",
    )
    parser.add_argument(
        "--conformal_joint_args",
        required=True,
        help="Path to conformal_joint_args.json from conf_rand_joint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used once before rollout collection (not reset per trajectory).",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=100,
        help="Number of trajectories to collect.",
    )
    parser.add_argument(
        "--output_path",
        default="rand_joint_trajectories.npz",
        help="Path to save the collected rollout dataset.",
    )
    parser.add_argument(
        "--conformal_joint_environment",
        default=None,
        help="Optional path to the canonical joint environment JSON saved by conf_ball_joint.",
    )
    parser.add_argument(
        "--spawn_ball_radius",
        type=float,
        default=0.0,
        help="Radius of the 3D ball used to resample each quad around its saved canonical joint start.",
    )
    parser.add_argument(
        "--spawn_ball_max_tries",
        type=int,
        default=1000,
        help="Maximum number of pairwise-valid joint spawn-ball attempts before failing.",
    )
    return parser.parse_args()


def _load_conformal_joint_args(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_conformal_joint_environment(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_points = np.asarray(data["start_points"], dtype=np.float64)
    goal_pairs = np.asarray(data["goal_pairs"], dtype=np.float64)
    if start_points.ndim != 2 or start_points.shape[1] != 3:
        raise ValueError(f"start_points must have shape (num_agents, 3), got {start_points.shape}")
    if goal_pairs.ndim != 3 or goal_pairs.shape[1:] != (2, 3):
        raise ValueError(f"goal_pairs must have shape (num_agents, 2, 3), got {goal_pairs.shape}")
    if start_points.shape[0] != goal_pairs.shape[0]:
        raise ValueError("start_points and goal_pairs disagree on agent count")
    return {
        "start_points": start_points,
        "goal_pairs": goal_pairs,
    }


def _capture_canonical_joint_layout(env_unwrapped) -> Dict[str, np.ndarray]:
    scenario = env_unwrapped.scenario
    start_points = np.asarray(scenario.start_points, dtype=np.float64).copy()
    goal_pairs = np.asarray(scenario.goal_pairs, dtype=np.float64).copy()
    if start_points.ndim != 2 or start_points.shape[1] != 3:
        raise ValueError(f"scenario.start_points must have shape (num_agents, 3), got {start_points.shape}")
    if goal_pairs.ndim != 3 or goal_pairs.shape[1:] != (2, 3):
        raise ValueError(f"scenario.goal_pairs must have shape (num_agents, 2, 3), got {goal_pairs.shape}")
    if start_points.shape[0] != goal_pairs.shape[0]:
        raise ValueError("scenario.start_points and scenario.goal_pairs disagree on agent count")
    return {
        "start_points": start_points,
        "goal_pairs": goal_pairs,
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
        raise ValueError("--spawn_ball_radius must be non-negative")
    if max_tries <= 0:
        raise ValueError("--spawn_ball_max_tries must be positive")
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


def _joint_state_model_mismatch(actions: np.ndarray, env_unwrapped) -> float:
    actions = np.asarray(actions, dtype=np.float64)
    dt = float(env_unwrapped.control_dt)
    states_cbf = []
    states_real = []
    for agent_id, quad in enumerate(env_unwrapped.envs):
        dynamics = quad.dynamics
        normalized = np.clip(0.5 * (actions[agent_id] + 1.0), 0.0, 1.0)
        states_cbf.append(_pack_state_tuple(*cbf_dynamics(normalized, dynamics, dt)))
        states_real.append(_pack_state_tuple(*real_dynamics(normalized, dynamics, dt)))
    return float(np.linalg.norm(np.concatenate(states_cbf, axis=0) - np.concatenate(states_real, axis=0)))


def _joint_action_thrusts_and_accelerations(actions: np.ndarray, env_unwrapped):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"Expected actions with shape (num_agents, 4), got {actions.shape}")

    dt = float(env_unwrapped.control_dt)
    thrusts = []
    accelerations = []
    for agent_id, quad in enumerate(env_unwrapped.envs):
        dynamics = quad.dynamics
        normalized = np.clip(0.5 * (actions[agent_id] + 1.0), 0.0, 1.0)
        thrust_vec, _ = _normalized_to_thrust(normalized, dynamics)
        current_vel = np.asarray(dynamics.vel, dtype=np.float64).copy()
        _, next_vel, _, _ = cbf_dynamics(normalized, dynamics, dt)
        accel = (np.asarray(next_vel, dtype=np.float64) - current_vel) / dt
        thrusts.append(np.asarray(thrust_vec, dtype=np.float64))
        accelerations.append(np.asarray(accel, dtype=np.float64))

    return np.stack(thrusts, axis=0), np.stack(accelerations, axis=0)


def make_joint_cbf_filter(
    r_mismatch: float,
    separation_radius: float,
    gamma: float,
    use_repeated_linearization: bool = False,
):
    def _filter(base_action: np.ndarray, env_state):
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_state.unwrapped,
            r=float(r_mismatch),
            separation_radius=float(separation_radius),
            gamma=float(gamma),
            use_repeated_linearization=use_repeated_linearization,
        )

    return _filter


def _pad_with_nan(values: np.ndarray, target_len: int) -> np.ndarray:
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


def _capture_reference_joint_initialization(env_unwrapped) -> Dict[str, np.ndarray]:
    scenario = env_unwrapped.scenario
    positions = []
    velocities = []
    rotations = []
    omegas = []
    goals = []
    for quad in env_unwrapped.envs:
        dynamics = quad.dynamics
        positions.append(np.asarray(dynamics.pos, dtype=np.float64).copy())
        velocities.append(np.asarray(dynamics.vel, dtype=np.float64).copy())
        rotations.append(np.asarray(dynamics.rot, dtype=np.float64).copy())
        omegas.append(np.asarray(dynamics.omega, dtype=np.float64).copy())
        goals.append(np.asarray(quad.goal, dtype=np.float64).copy())

    return {
        "positions": np.stack(positions, axis=0),
        "velocities": np.stack(velocities, axis=0),
        "rotations": np.stack(rotations, axis=0),
        "omegas": np.stack(omegas, axis=0),
        "goals": np.stack(goals, axis=0),
        "goal_pairs": np.asarray(scenario.goal_pairs, dtype=np.float64).copy(),
        "active_goal_index": np.asarray(scenario.active_goal_index, dtype=np.int64).copy(),
    }


def _restore_reference_joint_initialization(env_unwrapped, reference_init: Dict[str, np.ndarray]) -> np.ndarray:
    scenario = env_unwrapped.scenario
    scenario.goal_pairs = np.asarray(reference_init["goal_pairs"], dtype=np.float64).copy()
    scenario.active_goal_index = np.asarray(reference_init["active_goal_index"], dtype=np.int64).copy()

    saved_positions = np.asarray(reference_init["positions"], dtype=np.float64)
    saved_velocities = np.asarray(reference_init["velocities"], dtype=np.float64)
    saved_rotations = np.asarray(reference_init["rotations"], dtype=np.float64)
    saved_omegas = np.asarray(reference_init["omegas"], dtype=np.float64)
    saved_goals = np.asarray(reference_init["goals"], dtype=np.float64)

    for agent_id, quad in enumerate(env_unwrapped.envs):
        quad.dynamics.set_state(
            saved_positions[agent_id],
            saved_velocities[agent_id],
            saved_rotations[agent_id],
            saved_omegas[agent_id],
        )
        quad.dynamics.reset()
        quad.dynamics.on_floor = False
        quad.dynamics.crashed_floor = quad.dynamics.crashed_wall = quad.dynamics.crashed_ceiling = False
        quad.tick = 0
        quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]
        quad.goal = saved_goals[agent_id].copy()
        env_unwrapped.pos[agent_id, :] = quad.dynamics.pos
        env_unwrapped.vel[agent_id, :] = quad.dynamics.vel

    scenario.goals = saved_goals.copy()

    obs = [quad.state_vector(quad) for quad in env_unwrapped.envs]
    if env_unwrapped.num_use_neighbor_obs > 0:
        obs = env_unwrapped.add_neighborhood_obs(obs)
    return np.asarray(obs, dtype=np.float32)


def _apply_authoritative_joint_environment(
    env_unwrapped,
    saved_layout: Dict[str, np.ndarray],
    spawn_ball_radius: float,
    spawn_ball_max_tries: int,
    min_pairwise_distance: float,
) -> np.ndarray:
    scenario = env_unwrapped.scenario
    start_points = np.asarray(saved_layout["start_points"], dtype=np.float64)
    goal_pairs = np.asarray(saved_layout["goal_pairs"], dtype=np.float64)

    if start_points.shape != (len(env_unwrapped.envs), 3):
        raise ValueError(
            f"Saved start_points shape {start_points.shape} does not match expected ({len(env_unwrapped.envs)}, 3)"
        )
    if goal_pairs.shape != (len(env_unwrapped.envs), 2, 3):
        raise ValueError(
            f"Saved goal_pairs shape {goal_pairs.shape} does not match expected ({len(env_unwrapped.envs)}, 2, 3)"
        )

    sampled_positions = _sample_joint_ball_spawn_positions(
        start_points,
        spawn_ball_radius,
        min_pairwise_distance,
        spawn_ball_max_tries,
    )

    scenario.goal_pairs = goal_pairs.copy()
    if hasattr(scenario, "start_points"):
        scenario.start_points = start_points.copy()
    scenario.active_goal_index = np.zeros(len(env_unwrapped.envs), dtype=np.int64)
    if hasattr(scenario, "steps_since_switch"):
        scenario.steps_since_switch = np.zeros(len(env_unwrapped.envs), dtype=np.int64)
    scenario.goals = goal_pairs[:, 0].copy()
    if hasattr(scenario, "spawn_points"):
        scenario.spawn_points = sampled_positions.copy()

    for agent_id, quad in enumerate(env_unwrapped.envs):
        goal = goal_pairs[agent_id, 0].copy()
        pos = sampled_positions[agent_id].copy()
        vel = np.zeros(3, dtype=np.float64)
        omega = np.zeros(3, dtype=np.float64)
        rotation = _make_yaw_towards_goal_rotation(pos, goal)

        quad.goal = goal.copy()
        if hasattr(quad, "spawn_point"):
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


def _run_joint_trajectory(
    env,
    solo_actor,
    init_rnn_states: torch.Tensor,
    solo_obs_dim: int,
    joint_action_fn,
    saved_layout: Dict[str, np.ndarray],
    spawn_ball_radius: float,
    spawn_ball_max_tries: int,
    min_spawn_pairwise_distance: float,
    num_agents: int,
    max_steps: int,
    deterministic: bool,
    disable_boundary_collision: bool,
):
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)

    env.reset()
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)
    obs_run = _apply_authoritative_joint_environment(
        env.unwrapped,
        saved_layout,
        spawn_ball_radius,
        spawn_ball_max_tries,
        min_spawn_pairwise_distance,
    )
    done = False
    step_num = 0
    run_rnn_states = init_rnn_states.clone()

    scenario = env.unwrapped.scenario
    goals = []
    for agent_id in range(num_agents):
        active = scenario.active_goal_index[agent_id]
        target = scenario.goal_pairs[agent_id, active]
        goals.append(np.asarray(target, dtype=np.float64).copy())

    initial_positions, _ = extract_positions_velocities(env.unwrapped)
    initial_goals = np.asarray(goals, dtype=np.float64).copy()

    run_logs: Dict[str, List[np.ndarray]] = {
        "positions": [],
        "velocities": [],
        "goal_dist": [],
        "goal_swap": [],
        "model_mismatch_state": [],
        "nominal_thrust": [],
        "filtered_thrust": [],
        "nominal_acceleration": [],
        "filtered_acceleration": [],
    }

    while not done and step_num < max_steps:
        obs_self = obs_run[:, :solo_obs_dim]
        obs_dict = {OBS_KEY: obs_self}
        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(solo_actor, obs_dict)
            policy_out = solo_actor(normalized_obs, run_rnn_states)
        run_rnn_states = policy_out["new_rnn_states"]
        actions = policy_out["actions"]
        if deterministic:
            actions = argmax_actions(solo_actor.action_distribution())
        nominal_actions = np.asarray(actions.detach().cpu().numpy(), dtype=np.float64)
        if nominal_actions.ndim == 1:
            if nominal_actions.size != 4 * num_agents:
                raise ValueError(
                    f"Flat joint policy action has size {nominal_actions.size}, expected {4 * num_agents}"
                )
            nominal_actions = nominal_actions.reshape(num_agents, 4)
        if nominal_actions.shape != (num_agents, 4):
            raise ValueError(
                f"Joint policy action has shape {nominal_actions.shape}, expected ({num_agents}, 4)"
            )

        filtered_actions = np.asarray(joint_action_fn(base_action=nominal_actions, env_state=env), dtype=np.float64)
        nominal_thrust, nominal_acc = _joint_action_thrusts_and_accelerations(nominal_actions, env.unwrapped)
        filtered_thrust, filtered_acc = _joint_action_thrusts_and_accelerations(filtered_actions, env.unwrapped)
        mismatch = _joint_state_model_mismatch(filtered_actions, env.unwrapped)

        obs_run, rewards, dones, infos = _unpack_step_result(env.step(filtered_actions))
        obs_run = np.array(obs_run, dtype=np.float32)
        pos, vel = extract_positions_velocities(env.unwrapped)

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
        run_logs["model_mismatch_state"].append(float(mismatch))
        run_logs["nominal_thrust"].append(nominal_thrust.astype(np.float32))
        run_logs["filtered_thrust"].append(filtered_thrust.astype(np.float32))
        run_logs["nominal_acceleration"].append(nominal_acc.astype(np.float32))
        run_logs["filtered_acceleration"].append(filtered_acc.astype(np.float32))

        done = np.all(dones)
        step_num += 1

    for key in run_logs:
        run_logs[key] = np.asarray(run_logs[key])

    return {
        "positions": run_logs["positions"],
        "velocities": run_logs["velocities"],
        "goal_dist": run_logs["goal_dist"],
        "goal_swap": run_logs["goal_swap"],
        "model_mismatch_state": run_logs["model_mismatch_state"],
        "nominal_thrust": np.asarray(run_logs["nominal_thrust"], dtype=np.float32),
        "filtered_thrust": np.asarray(run_logs["filtered_thrust"], dtype=np.float32),
        "nominal_acceleration": np.asarray(run_logs["nominal_acceleration"], dtype=np.float32),
        "filtered_acceleration": np.asarray(run_logs["filtered_acceleration"], dtype=np.float32),
        "initial_positions": np.asarray(initial_positions, dtype=np.float64),
        "initial_goals": initial_goals,
        "trajectory_length": int(step_num),
    }


def _build_eval_cfg(num_agents: int, use_downwash: bool) -> "AttrDict":
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={num_agents}",
        "--quads_neighbor_visible_num=0",
        "--quads_neighbor_obs_type=none",
        "--quads_collision_reward=8.0",
        "--quads_collision_hitbox_radius=2.5",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        f"--quads_use_downwash={bool(use_downwash)}",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=False",
    ]
    return parse_swarm_cfg(eval_cli, evaluation=True)


def main() -> None:
    args = parse_args()
    if args.num_trajectories <= 0:
        raise ValueError("--num_trajectories must be positive.")
    if args.spawn_ball_radius < 0.0:
        raise ValueError("--spawn_ball_radius must be non-negative.")
    if args.spawn_ball_max_tries <= 0:
        raise ValueError("--spawn_ball_max_tries must be positive.")

    conf_args = _load_conformal_joint_args(args.conformal_joint_args)

    if "solo_train_dir" not in conf_args:
        raise KeyError(f"{args.conformal_joint_args} is missing 'solo_train_dir'.")
    if "solo_experiment" not in conf_args:
        raise KeyError(f"{args.conformal_joint_args} is missing 'solo_experiment'.")

    num_agents = int(conf_args.get("num_agents", 8))
    episode_length = int(conf_args.get("episode_length", 1500))
    separation_radius = float(conf_args.get("separation_radius", 0.5))
    gamma = float(conf_args.get("gamma", 0.8))
    disable_boundary_collision = bool(conf_args.get("disable_boundary_collision", False))
    deterministic = bool(conf_args.get("deterministic", False))
    use_downwash = bool(conf_args.get("use_downwash", False))
    use_repeated_linearization = bool(conf_args.get("use_repeated_linearization", False))
    if num_agents < 2:
        raise ValueError("--num_agents must be >= 2 for joint collection.")

    torch.set_grad_enabled(False)
    register_swarm_components()
    set_global_seed(args.seed)

    cfg_path = os.path.abspath(args.conformal_joint_args)
    print(f"[collect_rand_joint] Loading conformal config from {cfg_path}")

    eval_cfg = _build_eval_cfg(num_agents=num_agents, use_downwash=use_downwash)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)
    arm_len = float(env.quad_arm)
    min_spawn_pairwise_distance = arm_len * 2.5 + separation_radius

    cfg_solo = load_cfg(conf_args["solo_train_dir"], conf_args["solo_experiment"])
    solo_ckpt = latest_checkpoint(conf_args["solo_train_dir"], conf_args["solo_experiment"], policy_index=0)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    filter_fn = make_joint_cbf_filter(
        args.r_mismatch,
        separation_radius,
        gamma,
        use_repeated_linearization=use_repeated_linearization,
    )
    init_rnn_states = torch.zeros((num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)

    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)
    env.reset()
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)
    if args.conformal_joint_environment is not None:
        saved_layout = _load_conformal_joint_environment(args.conformal_joint_environment)
        if int(saved_layout["start_points"].shape[0]) != num_agents:
            raise ValueError("--conformal_joint_environment agent count does not match num_agents from conformal_joint_args")
    else:
        saved_layout = _capture_canonical_joint_layout(env.unwrapped)

    run_logs = []
    for _ in tqdm(range(args.num_trajectories), desc="Collecting trajectories", unit="traj"):
        run_logs.append(
            _run_joint_trajectory(
                env=env,
                solo_actor=solo_actor,
                init_rnn_states=init_rnn_states,
                solo_obs_dim=solo_obs_dim,
                joint_action_fn=filter_fn,
                saved_layout=saved_layout,
                spawn_ball_radius=args.spawn_ball_radius,
                spawn_ball_max_tries=args.spawn_ball_max_tries,
                min_spawn_pairwise_distance=min_spawn_pairwise_distance,
                num_agents=num_agents,
                max_steps=episode_length,
                deterministic=deterministic,
                disable_boundary_collision=disable_boundary_collision,
            )
        )

    env.close()

    max_len = max(run["trajectory_length"] for run in run_logs) if run_logs else 0
    if max_len <= 0:
        raise RuntimeError("No trajectory steps were collected.")

    positions = np.stack([_pad_with_nan(run["positions"], max_len) for run in run_logs], axis=0)
    velocities = np.stack([_pad_with_nan(run["velocities"], max_len) for run in run_logs], axis=0)
    goal_dist = np.stack([_pad_with_nan(run["goal_dist"], max_len) for run in run_logs], axis=0)
    goal_swap = np.stack([_pad_with_nan(run["goal_swap"], max_len) for run in run_logs], axis=0)
    model_mismatch_state = np.stack([_pad_with_nan(run["model_mismatch_state"], max_len) for run in run_logs], axis=0)
    nominal_thrusts = np.stack([_pad_with_nan(run["nominal_thrust"], max_len) for run in run_logs], axis=0)
    filtered_thrusts = np.stack([_pad_with_nan(run["filtered_thrust"], max_len) for run in run_logs], axis=0)
    nominal_accelerations = np.stack([_pad_with_nan(run["nominal_acceleration"], max_len) for run in run_logs], axis=0)
    filtered_accelerations = np.stack([_pad_with_nan(run["filtered_acceleration"], max_len) for run in run_logs], axis=0)
    initial_positions = np.stack([run["initial_positions"] for run in run_logs], axis=0)
    initial_goals = np.stack([run["initial_goals"] for run in run_logs], axis=0)
    trajectory_lengths = np.asarray([run["trajectory_length"] for run in run_logs], dtype=np.int32)

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.output_path,
        positions=positions,
        velocities=velocities,
        goal_dist=goal_dist,
        goal_swap=goal_swap,
        model_mismatch_state=model_mismatch_state,
        nominal_thrusts=nominal_thrusts,
        filtered_thrusts=filtered_thrusts,
        nominal_accelerations=nominal_accelerations,
        filtered_accelerations=filtered_accelerations,
        trajectory_lengths=trajectory_lengths,
        initial_positions=initial_positions,
        initial_goals=initial_goals,
        canonical_start_points=np.asarray(saved_layout["start_points"], dtype=np.float64),
        canonical_goal_pairs=np.asarray(saved_layout["goal_pairs"], dtype=np.float64),
        r_mismatch=np.array(float(args.r_mismatch), dtype=np.float64),
        seed=np.array(int(args.seed), dtype=np.int64),
        num_agents=np.array(int(num_agents), dtype=np.int64),
        episode_length=np.array(int(episode_length), dtype=np.int64),
        separation_radius=np.array(float(separation_radius), dtype=np.float64),
        gamma=np.array(float(gamma), dtype=np.float64),
        use_downwash=np.array(bool(use_downwash)),
        use_repeated_linearization=np.array(bool(use_repeated_linearization)),
        deterministic=np.array(bool(deterministic)),
        disable_boundary_collision=np.array(bool(disable_boundary_collision)),
        conformal_joint_args_path=np.array(cfg_path),
        conformal_joint_environment_path=np.array(
            os.path.abspath(args.conformal_joint_environment) if args.conformal_joint_environment is not None else ""
        ),
        spawn_ball_radius=np.array(float(args.spawn_ball_radius), dtype=np.float64),
        spawn_ball_max_tries=np.array(int(args.spawn_ball_max_tries), dtype=np.int64),
        num_trajectories=np.array(int(args.num_trajectories), dtype=np.int64),
    )

    print(f"[collect_rand_joint] Saved trajectories to {os.path.abspath(args.output_path)}")
    print(f"[collect_rand_joint] Collected {len(run_logs)} trajectories, max_len={max_len}")


if __name__ == "__main__":
    main()
