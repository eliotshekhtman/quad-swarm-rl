#!/usr/bin/env python3
"""
Collect a fixed-radius conformal rollout and render a replay video.
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

from gym_art.quadrotor_multi.quad_utils import QUAD_COLOR
from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.conformal_utils import fall_down, run_multi_agents
from project_utils.full_cbf_utils import make_cbf_filter
from project_utils.restart_utils import deterministic_reset, reseed_quads, set_global_seed
from project_utils.utils import (
    OBS_KEY,
    get_swarm_state,
    latest_checkpoint,
    load_actor,
    load_cfg,
)


DEVICE = torch.device("cpu")

EGO_QUAD_RGB = np.array([1.0, 0.0, 0.0], dtype=np.float64)
SPHERE_ALPHA = 0.24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fixed-radius conformal patrol videos.")
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", required=True)
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--predictions_dir", default="train_dir")
    parser.add_argument("--init_predictions", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout_seed", type=int, default=None)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--num_closest", type=int, default=1)
    parser.add_argument("--output_path", default="train_dir/collect_video.npz")
    parser.add_argument("--video_name", default="collect_video.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_view_mode", default="topdownfollow")
    parser.add_argument("--video_topdownfollow_zoom", type=float, default=5.0)
    parser.add_argument(
        "--video_global_elev",
        type=float,
        default=None,
        help="Matplotlib-style elevation angle in degrees for --video_view_mode global.",
    )
    parser.add_argument(
        "--video_global_azim",
        type=float,
        default=None,
        help="Matplotlib-style azimuth angle in degrees for --video_view_mode global.",
    )
    parser.add_argument(
        "--video_global_radius",
        type=float,
        default=None,
        help="Camera distance for --video_view_mode global.",
    )
    parser.add_argument("--skip_video", action="store_true")
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
            "collect_video replay rendering requires the normal simulator renderer, but env.render() returned None."
        )
    video_frames.append(np.asarray(frame, dtype=np.uint8).copy())


def _apply_rollout_seed(env, rollout_seed: int) -> None:
    set_global_seed(int(rollout_seed))
    reseed_quads(env.unwrapped, int(rollout_seed))


def _configure_replay_scene(
    env,
    ego_global_idx: int,
    topdownfollow_zoom: float,
    agent_colors: np.ndarray,
    global_elev: float | None = None,
    global_azim: float | None = None,
    global_radius: float | None = None,
) -> None:
    env_unwrapped = env.unwrapped
    if len(env_unwrapped.scenes) == 0:
        env_unwrapped.init_scene_multi()
    for scene in env_unwrapped.scenes:
        scene.camera_drone_index = int(ego_global_idx)
        scene.agent_colors = np.asarray(agent_colors, dtype=np.float64).copy()
        if scene.viewpoint == "topdownfollow":
            scene.chase_cam.view_dist = float(topdownfollow_zoom)
        elif scene.viewpoint == "global":
            if global_elev is not None:
                scene.chase_cam.theta = np.deg2rad(90.0 - float(global_elev))
            if global_azim is not None:
                scene.chase_cam.phi = np.deg2rad(float(global_azim))
            if global_radius is not None:
                scene.chase_cam.radius = float(global_radius)


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
    _apply_rollout_seed(env, args.rollout_seed)
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


def _collect_trajectory(
    snapshot,
    obs: np.ndarray,
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

    try:
        while (not done) and step_num < max_steps:
            obs_multi_dict = {OBS_KEY: obs_run[:num_multi_agents]}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_output = multi_actor(normalized_obs, run_multi_rnn_states)
            actions_multi = policy_output["actions"]
            run_multi_rnn_states = policy_output["new_rnn_states"]
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
            action_solo = policy_solo["actions"]
            if deterministic:
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


def _build_replay_agent_colors(num_multi_agents: int) -> np.ndarray:
    local_agents = int(num_multi_agents) + 1
    non_ego_palette = [
        np.asarray(color, dtype=np.float64)
        for color in QUAD_COLOR
        if not np.allclose(np.asarray(color, dtype=np.float64), EGO_QUAD_RGB)
    ]
    if len(non_ego_palette) == 0:
        raise ValueError("QUAD_COLOR must contain at least one non-red color for non-ego quads.")

    agent_colors = np.zeros((local_agents, 3), dtype=np.float64)
    for agent_idx in range(local_agents):
        if agent_idx == int(num_multi_agents):
            agent_colors[agent_idx] = EGO_QUAD_RGB
        else:
            agent_colors[agent_idx] = non_ego_palette[agent_idx % len(non_ego_palette)]
    return agent_colors


def _set_replay_frame(
    env_unwrapped,
    trajectory: Dict[str, np.ndarray],
    pred_trajectories: np.ndarray,
    agent_colors: np.ndarray,
    frame_idx: int,
    radius: float,
    num_multi_agents: int,
    num_closest: int,
) -> None:
    num_agents = int(num_multi_agents) + 1
    if len(env_unwrapped.envs) != num_agents:
        raise ValueError("Replay env agent count does not match trajectory agent count.")

    bubble_positions = np.zeros((num_agents, 3), dtype=np.float64)
    bubble_rgba = np.zeros((num_agents, 4), dtype=np.float64)
    goals_out = []

    pred_step = _prediction_step_for_frame(frame_idx, pred_trajectories.shape[1])
    positions, velocities, rotations, omegas, goals = _state_for_frame(trajectory, frame_idx)
    actual_ego_pos = positions[num_multi_agents]
    actual_teammate_pos = positions[:num_multi_agents]
    predicted_positions = pred_trajectories[:, pred_step, :3]

    actual_distances = np.linalg.norm(actual_teammate_pos - actual_ego_pos[None, :], axis=1)
    nearest_count = min(max(int(num_closest), 0), num_multi_agents)
    nearest = np.argsort(actual_distances)[:nearest_count]

    for agent_idx in range(num_agents):
        quad = env_unwrapped.envs[agent_idx]
        quad.goal = np.asarray(goals[agent_idx], dtype=np.float64).copy()
        quad.spawn_point = np.asarray(trajectory["initial_positions"][agent_idx], dtype=np.float64).copy()
        quad.dynamics.set_state(
            np.asarray(positions[agent_idx], dtype=np.float64),
            np.asarray(velocities[agent_idx], dtype=np.float64),
            np.asarray(rotations[agent_idx], dtype=np.float64),
            np.asarray(omegas[agent_idx], dtype=np.float64),
        )
        quad.dynamics.reset()
        quad.dynamics.on_floor = False
        quad.dynamics.crashed_floor = False
        quad.dynamics.crashed_wall = False
        quad.dynamics.crashed_ceiling = False
        quad.tick = int(frame_idx)
        quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]

        env_unwrapped.pos[agent_idx, :] = quad.dynamics.pos
        env_unwrapped.vel[agent_idx, :] = quad.dynamics.vel
        goals_out.append(np.asarray(quad.goal, dtype=np.float64).copy())

    for teammate_idx in nearest:
        bubble_positions[teammate_idx] = predicted_positions[teammate_idx]
        bubble_rgba[teammate_idx, :3] = agent_colors[teammate_idx]
        bubble_rgba[teammate_idx, 3] = SPHERE_ALPHA

    env_unwrapped.render_bubble_radius = float(radius)
    env_unwrapped.render_bubble_positions = bubble_positions
    env_unwrapped.render_bubble_rgba = bubble_rgba
    env_unwrapped.all_collisions = {
        "drone": np.zeros(num_agents, dtype=np.float64),
        "ground": np.zeros(num_agents, dtype=np.float64),
        "obstacle": np.zeros(num_agents, dtype=np.float64),
    }
    if hasattr(env_unwrapped.scenario, "goals"):
        env_unwrapped.scenario.goals = np.asarray(goals_out, dtype=np.float64)


def _record_replay_video(
    cfg_multi,
    trajectory: Dict[str, np.ndarray],
    pred_trajectories: np.ndarray,
    radius: float,
    num_multi_agents: int,
    num_closest: int,
    video_path: str,
    video_fps: int,
    video_view_mode: str,
    video_topdownfollow_zoom: float,
    video_global_elev: float | None,
    video_global_azim: float | None,
    video_global_radius: float | None,
) -> None:
    num_agents = int(num_multi_agents) + 1
    replay_cfg = _build_eval_cfg(cfg_multi, num_agents=num_agents, enable_render=True, view_mode=video_view_mode)
    replay_env = make_quadrotor_env("quadrotor_multi", cfg=replay_cfg, render_mode="rgb_array")
    replay_env.unwrapped.render_bubble_radius = float(radius)
    agent_colors = _build_replay_agent_colors(num_multi_agents)

    try:
        _reset_env(replay_env)
        ego_idx = int(num_multi_agents)
        _configure_replay_scene(
            replay_env,
            ego_idx,
            video_topdownfollow_zoom,
            agent_colors,
            global_elev=video_global_elev,
            global_azim=video_global_azim,
            global_radius=video_global_radius,
        )
        max_len = int(trajectory["trajectory_length"])
        video_frames: List[np.ndarray] = []
        for frame_idx in range(max_len + 1):
            _set_replay_frame(
                replay_env.unwrapped,
                trajectory,
                pred_trajectories,
                agent_colors,
                frame_idx,
                radius=radius,
                num_multi_agents=num_multi_agents,
                num_closest=num_closest,
            )
            _append_video_frame(replay_env, video_frames)
        _save_video_frames(video_path, video_frames, video_fps)
    finally:
        replay_env.close()


def main() -> None:
    args = parse_args()
    if args.rollout_seed is None:
        args.rollout_seed = int(args.seed)
    if args.r <= 0.0:
        raise ValueError("--r must be positive.")
    if args.num_closest < 0:
        raise ValueError("--num_closest must be non-negative.")
    if args.video_topdownfollow_zoom <= 0.0:
        raise ValueError("--video_topdownfollow_zoom must be positive.")
    if args.video_global_radius is not None and args.video_global_radius <= 0.0:
        raise ValueError("--video_global_radius must be positive.")

    output_path = _normalize_output_path(args.output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    video_path = None
    if not args.skip_video:
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

        _apply_rollout_seed(env, args.rollout_seed)
        snapshot = safe_capture_env_snapshot(env)
        radii = np.full(args.num_multi_agents, float(args.r), dtype=np.float64)
        solo_action_fn = make_cbf_filter(radii)

        trajectory = _collect_trajectory(
            snapshot,
            obs,
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
    finally:
        env.close()

    np.savez_compressed(
        output_path,
        positions=trajectory["positions"],
        velocities=trajectory["velocities"],
        rotations=trajectory["rotations"],
        omegas=trajectory["omegas"],
        goals=trajectory["goals"],
        initial_positions=trajectory["initial_positions"],
        initial_velocities=trajectory["initial_velocities"],
        initial_rotations=trajectory["initial_rotations"],
        initial_omegas=trajectory["initial_omegas"],
        initial_goals=trajectory["initial_goals"],
        trajectory_length=np.int32(trajectory["trajectory_length"]),
        pred_trajectories=pred_trajectories,
        r=np.float64(args.r),
        num_closest=np.int32(args.num_closest),
        num_multi_agents=np.int32(args.num_multi_agents),
        seed=np.int32(args.seed),
        rollout_seed=np.int32(args.rollout_seed),
        scenario_positions=seeded_signature["positions"],
        scenario_goals=seeded_signature["goals"],
        scenario_goal_pairs=seeded_signature["goal_pairs"],
        scenario_active_goal_index=seeded_signature["active_goal_index"],
    )

    args_path = os.path.splitext(output_path)[0] + "_args.json"
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    if not args.skip_video:
        _record_replay_video(
            cfg_multi=cfg_multi,
            trajectory=trajectory,
            pred_trajectories=pred_trajectories,
            radius=float(args.r),
            num_multi_agents=args.num_multi_agents,
            num_closest=args.num_closest,
            video_path=video_path,
            video_fps=args.video_fps,
            video_view_mode=args.video_view_mode,
            video_topdownfollow_zoom=args.video_topdownfollow_zoom,
            video_global_elev=args.video_global_elev,
            video_global_azim=args.video_global_azim,
            video_global_radius=args.video_global_radius,
        )


if __name__ == "__main__":
    main()
