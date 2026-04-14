#!/usr/bin/env python3
"""
Collect fixed-radius conformal rollouts and render a composite replay video.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Sequence

import numpy as np
import torch

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.huggingface.huggingface_utils import generate_replay_video
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.cbf_utils import make_cbf_filter
from project_utils.conformal_utils import fall_down, run_multi_agents
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import (
    OBS_KEY,
    get_swarm_state,
    latest_checkpoint,
    load_actor,
    load_cfg,
)


DEVICE = torch.device("cpu")

GREEN_RGBA = np.array([0.0, 1.0, 0.0, 0.22], dtype=np.float64)
RED_RGBA = np.array([1.0, 0.0, 0.0, 0.22], dtype=np.float64)
ORANGE_RGBA = np.array([1.0, 0.55, 0.0, 0.22], dtype=np.float64)
BLUE_RGBA = np.array([0.0, 0.45, 1.0, 0.24], dtype=np.float64)
TRANSPARENT_RGBA = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fixed-radius conformal patrol videos.")
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", required=True)
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--predictions_dir", default="train_dir")
    parser.add_argument("--init_predictions", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--num_closest", type=int, default=1)
    parser.add_argument("--output_path", default="train_dir/collect_video.npz")
    parser.add_argument("--video_name", default="collect_video.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_view_mode", default="topdown")
    return parser.parse_args()


def _normalize_output_path(path: str) -> str:
    return path if path.endswith(".npz") else f"{path}.npz"


def _resolve_video_output(output_path: str, video_name: str) -> str:
    if os.path.isabs(video_name):
        return video_name
    return os.path.join(os.path.dirname(output_path) or ".", video_name)


def _append_video_frame(env, video_frames: List[np.ndarray]) -> None:
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "collect_video composite rendering requires the normal simulator renderer, but env.render() returned None."
        )
    video_frames.append(np.asarray(frame, dtype=np.uint8).copy())


def _save_video_frames(video_path: str, video_frames: List[np.ndarray], fps: int) -> None:
    if len(video_frames) == 0:
        raise ValueError("No video frames to save.")

    if shutil.which("ffmpeg") is not None:
        video_dir = os.path.dirname(video_path) or "."
        video_file = os.path.basename(video_path)
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, fps, video_cfg)
        return

    import cv2

    first = np.asarray(video_frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for frame in video_frames:
            frame_arr = np.asarray(frame, dtype=np.uint8)
            if frame_arr.shape[:2] != (height, width):
                raise ValueError("All video frames must have the same resolution.")
            writer.write(cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _reset_env(env) -> np.ndarray:
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return np.asarray(reset_result[0], dtype=np.float32)
    return np.asarray(reset_result, dtype=np.float32)


def _build_eval_cfg(cfg_multi, num_agents: int, enable_render: bool, view_mode: str):
    quads_collision_hitbox_radius = 2.5
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={num_agents}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_collision_reward=8.0",
        f"--quads_collision_hitbox_radius={quads_collision_hitbox_radius}",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        f"--quads_render={'True' if enable_render else 'False'}",
        f"--quads_view_mode={view_mode}",
    ]
    return parse_swarm_cfg(eval_cli, evaluation=True)


def _scenario_signature(stored_states, env) -> Dict[str, np.ndarray]:
    positions = np.stack([np.asarray(state.position, dtype=np.float64) for state in stored_states], axis=0)
    goals = np.stack([np.asarray(state.goal, dtype=np.float64) for state in stored_states], axis=0)
    goal_pairs = np.asarray(env.unwrapped.scenario.goal_pairs, dtype=np.float64).copy()
    active_goal_index = np.asarray(env.unwrapped.scenario.active_goal_index, dtype=np.int64).copy()
    return {
        "positions": positions,
        "goals": goals,
        "goal_pairs": goal_pairs,
        "active_goal_index": active_goal_index,
    }


def _load_or_generate_pred_trajectories(
    args: argparse.Namespace,
    env,
    stored_states,
    num_multi_agents: int,
    multi_actor,
    multi_rnn_states: torch.Tensor,
    solo_actor,
    solo_rnn_states: torch.Tensor,
    solo_obs_dim: int,
) -> List[np.ndarray]:
    if args.init_predictions is not None:
        pred_path = os.path.join(args.predictions_dir, args.init_predictions, "pred_trajectories.npz")
        pred_data = np.load(pred_path)
        pred_trajectories = pred_data["pred_trajectories"].astype(np.float32)
        return [pred_trajectories[i] for i in range(pred_trajectories.shape[0])]

    obs = deterministic_reset(env, args.seed, stored_states)[0]
    snapshot = safe_capture_env_snapshot(env)
    temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
    dummy_pred = np.zeros((num_multi_agents, args.episode_length, 6), dtype=np.float32)
    try:
        logs = run_multi_agents(
            temp_env,
            obs,
            num_multi_agents,
            multi_actor,
            multi_rnn_states,
            solo_actor,
            solo_rnn_states,
            solo_obs_dim,
            pred_trajectories=dummy_pred,
            solo_action_fn=fall_down,
            deterministic=True,
            max_steps=args.episode_length,
            num_threads=1,
        )
    finally:
        temp_env.close()

    pred_trajectories = []
    for agent_id in range(num_multi_agents):
        positions = np.asarray(logs[agent_id][0]["position"], dtype=np.float32)
        velocities = np.asarray(logs[agent_id][0]["velocity"], dtype=np.float32)
        pred_trajectories.append(np.concatenate([positions, velocities], axis=1))
    return pred_trajectories


def _extract_agent_state(quad) -> Dict[str, np.ndarray]:
    dynamics = quad.dynamics
    return {
        "position": np.asarray(dynamics.pos, dtype=np.float64).copy(),
        "velocity": np.asarray(dynamics.vel, dtype=np.float64).copy(),
        "rotation": np.asarray(dynamics.rot, dtype=np.float64).copy(),
        "omega": np.asarray(dynamics.omega, dtype=np.float64).copy(),
        "goal": np.asarray(quad.goal, dtype=np.float64).copy(),
    }


def _collect_single_trajectory(
    snapshot,
    obs: np.ndarray,
    trajectory_idx: int,
    num_multi_agents: int,
    multi_actor,
    multi_rnn_states: torch.Tensor,
    solo_actor,
    solo_rnn_states: torch.Tensor,
    solo_obs_dim: int,
    pred_trajectories: Sequence[np.ndarray],
    solo_action_fn,
    max_steps: int,
    deterministic: bool,
) -> Dict[str, np.ndarray]:
    env_run = clone_env_from_snapshot(snapshot, restore_rng=True)
    obs_run = np.asarray(obs, dtype=np.float32).copy()
    num_agents = num_multi_agents + 1

    positions = []
    velocities = []
    rotations = []
    omegas = []
    goals = []

    initial_states = [_extract_agent_state(quad) for quad in env_run.envs]

    run_multi_rnn_states = multi_rnn_states.clone()
    run_solo_rnn_states = solo_rnn_states.clone()

    done = False
    step_num = 0
    force_deterministic = deterministic or trajectory_idx == 0

    try:
        while (not done) and step_num < max_steps:
            obs_multi_dict = {OBS_KEY: obs_run[:num_multi_agents]}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_output = multi_actor(normalized_obs, run_multi_rnn_states)
            actions_multi = policy_output["actions"]
            run_multi_rnn_states = policy_output["new_rnn_states"]
            if force_deterministic:
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
            action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]

            swarm_state = get_swarm_state(env_run)
            for agent_id in range(num_multi_agents):
                swarm_state.positions[agent_id, :] = pred_trajectories[agent_id][step_num][:3]
                swarm_state.velocities[agent_id, :] = pred_trajectories[agent_id][step_num][3:]
            action_solo = solo_action_fn(base_action=action_solo, env_state=env_run, swarm_state=swarm_state)

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs_run, _rewards, dones, _infos = env_run.step(actions)
            obs_run = np.asarray(obs_run, dtype=np.float32)

            frame_positions = []
            frame_velocities = []
            frame_rotations = []
            frame_omegas = []
            frame_goals = []
            for quad in env_run.envs:
                state = _extract_agent_state(quad)
                frame_positions.append(state["position"])
                frame_velocities.append(state["velocity"])
                frame_rotations.append(state["rotation"])
                frame_omegas.append(state["omega"])
                frame_goals.append(state["goal"])

            positions.append(np.stack(frame_positions, axis=0))
            velocities.append(np.stack(frame_velocities, axis=0))
            rotations.append(np.stack(frame_rotations, axis=0))
            omegas.append(np.stack(frame_omegas, axis=0))
            goals.append(np.stack(frame_goals, axis=0))

            done = bool(np.all(dones))
            step_num += 1
    finally:
        env_run.close()

    def _initial_stack(key: str, shape_tail: tuple[int, ...]):
        return np.stack([np.asarray(state[key], dtype=np.float64).reshape(shape_tail) for state in initial_states], axis=0)

    if step_num > 0:
        positions_arr = np.stack(positions, axis=0)
        velocities_arr = np.stack(velocities, axis=0)
        rotations_arr = np.stack(rotations, axis=0)
        omegas_arr = np.stack(omegas, axis=0)
        goals_arr = np.stack(goals, axis=0)
    else:
        positions_arr = np.zeros((0, num_agents, 3), dtype=np.float64)
        velocities_arr = np.zeros((0, num_agents, 3), dtype=np.float64)
        rotations_arr = np.zeros((0, num_agents, 3, 3), dtype=np.float64)
        omegas_arr = np.zeros((0, num_agents, 3), dtype=np.float64)
        goals_arr = np.zeros((0, num_agents, 3), dtype=np.float64)

    return {
        "trajectory_length": np.int32(step_num),
        "initial_positions": _initial_stack("position", (3,)),
        "initial_velocities": _initial_stack("velocity", (3,)),
        "initial_rotations": _initial_stack("rotation", (3, 3)),
        "initial_omegas": _initial_stack("omega", (3,)),
        "initial_goals": _initial_stack("goal", (3,)),
        "positions": positions_arr,
        "velocities": velocities_arr,
        "rotations": rotations_arr,
        "omegas": omegas_arr,
        "goals": goals_arr,
    }


def _state_for_frame(traj: Dict[str, np.ndarray], frame_idx: int):
    traj_len = int(traj["trajectory_length"])
    if frame_idx <= 0 or traj_len <= 0:
        return (
            np.asarray(traj["initial_positions"], dtype=np.float64),
            np.asarray(traj["initial_velocities"], dtype=np.float64),
            np.asarray(traj["initial_rotations"], dtype=np.float64),
            np.asarray(traj["initial_omegas"], dtype=np.float64),
            np.asarray(traj["initial_goals"], dtype=np.float64),
        )

    step_idx = min(frame_idx - 1, traj_len - 1)
    return (
        np.asarray(traj["positions"][step_idx], dtype=np.float64),
        np.asarray(traj["velocities"][step_idx], dtype=np.float64),
        np.asarray(traj["rotations"][step_idx], dtype=np.float64),
        np.asarray(traj["omegas"][step_idx], dtype=np.float64),
        np.asarray(traj["goals"][step_idx], dtype=np.float64),
    )


def _prediction_step_for_frame(frame_idx: int, episode_length: int) -> int:
    if episode_length <= 0:
        return 0
    if frame_idx <= 0:
        return 0
    return min(frame_idx - 1, episode_length - 1)


def _bubble_color(current_h: float, had_prior_violation: bool, left_prediction_ball: bool) -> np.ndarray:
    if left_prediction_ball:
        return BLUE_RGBA.copy()
    if current_h <= 0.0:
        return RED_RGBA.copy()
    if had_prior_violation:
        return ORANGE_RGBA.copy()
    return GREEN_RGBA.copy()


def _set_composite_replay_frame(
    env_unwrapped,
    trajectories: Sequence[Dict[str, np.ndarray]],
    pred_trajectories: np.ndarray,
    frame_idx: int,
    radius: float,
    num_multi_agents: int,
    num_closest: int,
    had_h_violation: np.ndarray,
) -> None:
    total_agents = len(trajectories) * (num_multi_agents + 1)
    if len(env_unwrapped.envs) != total_agents:
        raise ValueError("Replay env agent count does not match composite trajectory count.")

    bubble_positions = np.zeros((total_agents, 3), dtype=np.float64)
    bubble_rgba = np.zeros((total_agents, 4), dtype=np.float64)
    goals_out = []

    pred_step = _prediction_step_for_frame(frame_idx, pred_trajectories.shape[1])
    local_agents = num_multi_agents + 1

    for traj_idx, traj in enumerate(trajectories):
        positions, velocities, rotations, omegas, goals = _state_for_frame(traj, frame_idx)
        actual_ego_pos = positions[num_multi_agents]
        actual_teammate_pos = positions[:num_multi_agents]
        predicted_positions = pred_trajectories[:, pred_step, :3]

        current_h = np.sum((predicted_positions - actual_ego_pos[None, :]) ** 2, axis=1) - float(radius) ** 2
        prior_h = had_h_violation[traj_idx].copy()
        left_prediction_ball = np.linalg.norm(actual_teammate_pos - predicted_positions, axis=1) > float(radius)
        actual_distances = np.linalg.norm(actual_teammate_pos - actual_ego_pos[None, :], axis=1)
        nearest_count = min(max(int(num_closest), 0), num_multi_agents)
        nearest = np.argsort(actual_distances)[:nearest_count]
        had_h_violation[traj_idx] = np.logical_or(had_h_violation[traj_idx], current_h <= 0.0)

        for local_idx in range(local_agents):
            global_idx = traj_idx * local_agents + local_idx
            quad = env_unwrapped.envs[global_idx]
            quad.goal = np.asarray(goals[local_idx], dtype=np.float64).copy()
            quad.spawn_point = np.asarray(traj["initial_positions"][local_idx], dtype=np.float64).copy()
            quad.dynamics.set_state(
                np.asarray(positions[local_idx], dtype=np.float64),
                np.asarray(velocities[local_idx], dtype=np.float64),
                np.asarray(rotations[local_idx], dtype=np.float64),
                np.asarray(omegas[local_idx], dtype=np.float64),
            )
            quad.dynamics.reset()
            quad.dynamics.on_floor = False
            quad.dynamics.crashed_floor = False
            quad.dynamics.crashed_wall = False
            quad.dynamics.crashed_ceiling = False
            quad.tick = int(frame_idx)
            quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]

            env_unwrapped.pos[global_idx, :] = quad.dynamics.pos
            env_unwrapped.vel[global_idx, :] = quad.dynamics.vel
            goals_out.append(np.asarray(quad.goal, dtype=np.float64).copy())

        for teammate_idx in nearest:
            global_idx = traj_idx * local_agents + teammate_idx
            bubble_positions[global_idx] = predicted_positions[teammate_idx]
            bubble_rgba[global_idx] = _bubble_color(
                float(current_h[teammate_idx]),
                bool(prior_h[teammate_idx]),
                bool(left_prediction_ball[teammate_idx]),
            )

    env_unwrapped.render_bubble_radius = float(radius)
    env_unwrapped.render_bubble_positions = bubble_positions
    env_unwrapped.render_bubble_rgba = bubble_rgba
    env_unwrapped.all_collisions = {
        "drone": np.zeros(total_agents, dtype=np.float64),
        "ground": np.zeros(total_agents, dtype=np.float64),
        "obstacle": np.zeros(total_agents, dtype=np.float64),
    }
    if hasattr(env_unwrapped.scenario, "goals"):
        env_unwrapped.scenario.goals = np.asarray(goals_out, dtype=np.float64)


def _record_composite_video(
    cfg_multi,
    trajectories: Sequence[Dict[str, np.ndarray]],
    pred_trajectories: np.ndarray,
    radius: float,
    num_multi_agents: int,
    num_closest: int,
    video_path: str,
    video_fps: int,
    video_view_mode: str,
) -> None:
    if len(trajectories) == 0:
        raise ValueError("Need at least one trajectory to render a composite video.")

    total_agents = len(trajectories) * (num_multi_agents + 1)
    replay_cfg = _build_eval_cfg(cfg_multi, num_agents=total_agents, enable_render=True, view_mode=video_view_mode)
    replay_env = make_quadrotor_env("quadrotor_multi", cfg=replay_cfg, render_mode="rgb_array")
    replay_env.unwrapped.render_bubble_radius = float(radius)

    try:
        _reset_env(replay_env)
        max_len = max(int(traj["trajectory_length"]) for traj in trajectories)
        video_frames: List[np.ndarray] = []
        had_h_violation = np.zeros((len(trajectories), num_multi_agents), dtype=bool)
        for frame_idx in range(max_len + 1):
            _set_composite_replay_frame(
                replay_env.unwrapped,
                trajectories,
                pred_trajectories,
                frame_idx,
                radius=radius,
                num_multi_agents=num_multi_agents,
                num_closest=num_closest,
                had_h_violation=had_h_violation,
            )
            _append_video_frame(replay_env, video_frames)
        _save_video_frames(video_path, video_frames, video_fps)
    finally:
        replay_env.close()


def main() -> None:
    args = parse_args()
    if args.r <= 0.0:
        raise ValueError("--r must be positive.")
    if args.num_trajectories <= 0:
        raise ValueError("--num_trajectories must be positive.")
    if args.num_closest < 0:
        raise ValueError("--num_closest must be non-negative.")

    output_path = _normalize_output_path(args.output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    video_path = _resolve_video_output(output_path, args.video_name)
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    torch.set_grad_enabled(False)
    register_swarm_components()

    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    if args.num_multi_agents < 0:
        args.num_multi_agents = int(cfg_multi.quads_num_agents)

    num_agents = int(args.num_multi_agents) + 1
    eval_cfg = _build_eval_cfg(cfg_multi, num_agents=num_agents, enable_render=False, view_mode=args.video_view_mode)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)

    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    try:
        multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
        multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, DEVICE)
        multi_rnn_size = get_rnn_size(cfg_multi)
        multi_rnn_states = torch.zeros((args.num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE)

        solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
        solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
        solo_obs_dim = int(solo_env.observation_space.shape[0])
        solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
    finally:
        solo_env.close()

    try:
        obs, stored_states = deterministic_reset(env, args.seed, None)
        initial_signature = _scenario_signature(stored_states, env)
        pred_trajectories_list = _load_or_generate_pred_trajectories(
            args,
            env,
            stored_states,
            args.num_multi_agents,
            multi_actor,
            multi_rnn_states,
            solo_actor,
            solo_rnn_states,
            solo_obs_dim,
        )
        if len(pred_trajectories_list) != args.num_multi_agents:
            raise ValueError(
                f"Predicted trajectory count ({len(pred_trajectories_list)}) does not match "
                f"--num_multi_agents ({args.num_multi_agents})."
            )
        pred_trajectories = np.stack(pred_trajectories_list, axis=0).astype(np.float64)

        obs, stored_states = deterministic_reset(env, args.seed, stored_states)
        seeded_signature = _scenario_signature(stored_states, env)
        for key in initial_signature:
            if not np.allclose(initial_signature[key], seeded_signature[key]):
                raise RuntimeError(f"Deterministic seeded reset mismatch for {key}.")

        snapshot = safe_capture_env_snapshot(env)
        radii = np.full(args.num_multi_agents, float(args.r), dtype=np.float64)
        solo_action_fn = make_cbf_filter(radii)

        trajectories = []
        for traj_idx in range(args.num_trajectories):
            trajectories.append(
                _collect_single_trajectory(
                    snapshot,
                    obs,
                    trajectory_idx=traj_idx,
                    num_multi_agents=args.num_multi_agents,
                    multi_actor=multi_actor,
                    multi_rnn_states=multi_rnn_states,
                    solo_actor=solo_actor,
                    solo_rnn_states=solo_rnn_states,
                    solo_obs_dim=solo_obs_dim,
                    pred_trajectories=pred_trajectories_list,
                    solo_action_fn=solo_action_fn,
                    max_steps=args.episode_length,
                    deterministic=args.deterministic,
                )
            )
    finally:
        env.close()

    trajectory_lengths = np.asarray([traj["trajectory_length"] for traj in trajectories], dtype=np.int32)
    positions = np.stack([traj["positions"] for traj in trajectories], axis=0)
    velocities = np.stack([traj["velocities"] for traj in trajectories], axis=0)
    rotations = np.stack([traj["rotations"] for traj in trajectories], axis=0)
    omegas = np.stack([traj["omegas"] for traj in trajectories], axis=0)
    goals = np.stack([traj["goals"] for traj in trajectories], axis=0)
    initial_positions = np.stack([traj["initial_positions"] for traj in trajectories], axis=0)
    initial_velocities = np.stack([traj["initial_velocities"] for traj in trajectories], axis=0)
    initial_rotations = np.stack([traj["initial_rotations"] for traj in trajectories], axis=0)
    initial_omegas = np.stack([traj["initial_omegas"] for traj in trajectories], axis=0)
    initial_goals = np.stack([traj["initial_goals"] for traj in trajectories], axis=0)

    np.savez_compressed(
        output_path,
        positions=positions,
        velocities=velocities,
        rotations=rotations,
        omegas=omegas,
        goals=goals,
        initial_positions=initial_positions,
        initial_velocities=initial_velocities,
        initial_rotations=initial_rotations,
        initial_omegas=initial_omegas,
        initial_goals=initial_goals,
        trajectory_lengths=trajectory_lengths,
        pred_trajectories=pred_trajectories,
        r=np.float64(args.r),
        num_closest=np.int32(args.num_closest),
        num_multi_agents=np.int32(args.num_multi_agents),
        seed=np.int32(args.seed),
        scenario_positions=seeded_signature["positions"],
        scenario_goals=seeded_signature["goals"],
        scenario_goal_pairs=seeded_signature["goal_pairs"],
        scenario_active_goal_index=seeded_signature["active_goal_index"],
    )

    args_path = os.path.splitext(output_path)[0] + "_args.json"
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    _record_composite_video(
        cfg_multi=cfg_multi,
        trajectories=trajectories,
        pred_trajectories=pred_trajectories,
        radius=float(args.r),
        num_multi_agents=args.num_multi_agents,
        num_closest=args.num_closest,
        video_path=video_path,
        video_fps=args.video_fps,
        video_view_mode=args.video_view_mode,
    )


if __name__ == "__main__":
    main()
