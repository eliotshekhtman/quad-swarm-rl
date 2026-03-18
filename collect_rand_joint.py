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

from project_utils.full_cbf_utils import apply_cbf_filter, cbf_dynamics, real_dynamics
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
    return parser.parse_args()


def _load_conformal_joint_args(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _run_joint_trajectory(
    env,
    solo_actor,
    init_rnn_states: torch.Tensor,
    solo_obs_dim: int,
    joint_action_fn,
    num_agents: int,
    max_steps: int,
    deterministic: bool,
    disable_boundary_collision: bool,
):
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)

    reset_out = env.reset()
    obs_run = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
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
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        actions = actions.detach().cpu().numpy()

        actions = joint_action_fn(base_action=actions, env_state=env)
        mismatch = _joint_state_model_mismatch(actions, env.unwrapped)

        obs_run, rewards, dones, infos = _unpack_step_result(env.step(actions))
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
        "initial_positions": np.asarray(initial_positions, dtype=np.float64),
        "initial_goals": initial_goals,
        "trajectory_length": int(step_num),
    }


def _build_eval_cfg(num_agents: int) -> "AttrDict":
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
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=False",
    ]
    return parse_swarm_cfg(eval_cli, evaluation=True)


def main() -> None:
    args = parse_args()
    if args.num_trajectories <= 0:
        raise ValueError("--num_trajectories must be positive.")

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

    if num_agents < 2:
        raise ValueError("--num_agents must be >= 2 for joint collection.")

    torch.set_grad_enabled(False)
    register_swarm_components()
    set_global_seed(args.seed)

    cfg_path = os.path.abspath(args.conformal_joint_args)
    print(f"[collect_rand_joint] Loading conformal config from {cfg_path}")

    eval_cfg = _build_eval_cfg(num_agents=num_agents)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)

    cfg_solo = load_cfg(conf_args["solo_train_dir"], conf_args["solo_experiment"])
    solo_ckpt = latest_checkpoint(conf_args["solo_train_dir"], conf_args["solo_experiment"], policy_index=0)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    filter_fn = make_joint_cbf_filter(args.r_mismatch, separation_radius, gamma)
    init_rnn_states = torch.zeros((num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)

    run_logs = []
    for _ in tqdm(range(args.num_trajectories), desc="Collecting trajectories", unit="traj"):
        run_logs.append(
            _run_joint_trajectory(
                env=env,
                solo_actor=solo_actor,
                init_rnn_states=init_rnn_states,
                solo_obs_dim=solo_obs_dim,
                joint_action_fn=filter_fn,
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
        trajectory_lengths=trajectory_lengths,
        initial_positions=initial_positions,
        initial_goals=initial_goals,
        r_mismatch=np.array(float(args.r_mismatch), dtype=np.float64),
        seed=np.array(int(args.seed), dtype=np.int64),
        num_agents=np.array(int(num_agents), dtype=np.int64),
        episode_length=np.array(int(episode_length), dtype=np.int64),
        separation_radius=np.array(float(separation_radius), dtype=np.float64),
        gamma=np.array(float(gamma), dtype=np.float64),
        deterministic=np.array(bool(deterministic)),
        disable_boundary_collision=np.array(bool(disable_boundary_collision)),
        conformal_joint_args_path=np.array(cfg_path),
        num_trajectories=np.array(int(args.num_trajectories), dtype=np.int64),
    )

    print(f"[collect_rand_joint] Saved trajectories to {os.path.abspath(args.output_path)}")
    print(f"[collect_rand_joint] Collected {len(run_logs)} trajectories, max_len={max_len}")


if __name__ == "__main__":
    main()
