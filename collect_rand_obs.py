#!/usr/bin/env python3
"""Collect fixed-radius obstacle-avoidance rollouts for plotting."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

from project_utils.cbf_utils import apply_cbf_filter, cbf_dynamics, real_dynamics
from project_utils.restart_utils import extract_positions_velocities, set_global_seed
from project_utils.utils import OBS_KEY, SwarmState, load_actor, load_cfg, latest_checkpoint


DEVICE = torch.device("cpu")
COLLISION_FAR_DISTANCE = 10000.0
DEFAULT_BUBBLE_OK_RGBA = (0.0, 1.0, 0.0, 0.18)
DEFAULT_BUBBLE_VIOLATION_RGBA = (1.0, 0.0, 0.0, 0.22)
DEFAULT_BUBBLE_RECOVERED_RGBA = (1.0, 0.55, 0.0, 0.22)
NAMED_VIDEO_COLORS = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "green": (0.0, 0.5, 0.0),
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


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
        default=None,
        help="Optional path to geometry JSON containing authoritative start/goal/obstacle data.",
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
        "--action_repeat",
        type=int,
        default=1,
        help="Hold each chosen action fixed for this many environment timesteps before recomputing it.",
    )
    parser.add_argument(
        "--use_repeated_linearization",
        action="store_true",
        help="Run a second ECBF QP solve after re-linearizing h^(4) around the first solution.",
    )
    parser.add_argument(
        "--spawn_ball_radius",
        type=float,
        default=0.0,
        help="Radius of the full 3D ball used to resample the initial quad position every trajectory.",
    )
    parser.add_argument(
        "--spawn_ball_max_tries",
        type=int,
        default=1000,
        help="Maximum number of spawn samples to try before failing.",
    )
    parser.add_argument(
        "--video_name",
        default=None,
        help="Optional mp4 output path/name prefix. When provided, record selected collected trajectories as videos.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="Frames per second for the optional trajectory video.",
    )
    parser.add_argument(
        "--video_trajectory_idx",
        type=int,
        default=0,
        help="Trajectory index to record when --video_name is set. Use -1 to record all collected trajectories.",
    )
    parser.add_argument(
        "--video_view_mode",
        default="topdown",
        help="Viewer mode for the optional trajectory video.",
    )
    parser.add_argument(
        "--video_composite_name",
        default=None,
        help="Optional mp4 output path/name for a replay-only composite video of the selected collected trajectories.",
    )
    parser.add_argument(
        "--dynamic_zoom",
        action="store_true",
        help="If set, dynamically adjust the topdown camera radius from the per-frame quad spread.",
    )
    parser.add_argument(
        "--video_zoom_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the topdown camera radius. Values > 1 zoom in and values < 1 zoom out.",
    )
    parser.add_argument(
        "--video_obstacle_color",
        default="green",
        help="Obstacle color for rendered videos. Use a named color like 'gray' or an RGB triplet like '128,128,128'.",
    )
    parser.add_argument(
        "--video_background_color",
        default="white",
        help="Background color for rendered videos. Use a named color like 'white' or an RGB triplet like '255,255,255'.",
    )
    return parser.parse_args()


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_video_color_arg(value: str, arg_name: str) -> tuple[float, float, float]:
    color_text = value.strip().lower()
    if color_text in NAMED_VIDEO_COLORS:
        return NAMED_VIDEO_COLORS[color_text]
    if color_text.startswith("#"):
        hex_value = color_text[1:]
        if len(hex_value) != 6:
            raise ValueError(f"{arg_name} hex colors must be in #RRGGBB format.")
        return tuple(int(hex_value[i:i + 2], 16) / 255.0 for i in range(0, 6, 2))

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"{arg_name} must be a named color, #RRGGBB, or three comma-separated RGB values."
        )

    try:
        channels = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{arg_name} has invalid numeric color channels.") from exc

    max_channel = max(channels)
    if max_channel > 1.0:
        if max_channel > 255.0:
            raise ValueError(f"{arg_name} RGB channels must be in [0, 1] or [0, 255].")
        channels = [channel / 255.0 for channel in channels]

    if any(channel < 0.0 or channel > 1.0 for channel in channels):
        raise ValueError(f"{arg_name} RGB channels must be in [0, 1] after normalization.")
    return tuple(channels)


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


def _capture_runtime_geometry(env) -> Dict[str, np.ndarray | float]:
    env_unwrapped = env.unwrapped
    return {
        "start_point": np.asarray(env_unwrapped.envs[0].dynamics.pos, dtype=np.float64).copy(),
        "goal_point": np.asarray(env_unwrapped.envs[0].goal, dtype=np.float64).copy(),
        "obstacle_positions": np.asarray(env_unwrapped.obstacles.pos_arr, dtype=np.float64).reshape(-1, 3).copy(),
        "obstacle_radius": float(env_unwrapped.obstacles.obstacle_radius),
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


def _spawn_is_valid_cbf_clearance(
    pos: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radius: float,
    quad_radius: float,
    obstacle_radius_margin: float,
) -> bool:
    obstacle_positions = np.asarray(obstacle_positions, dtype=np.float64).reshape(-1, 3)
    if obstacle_positions.size == 0:
        return True
    center_dists_xy = np.linalg.norm(obstacle_positions[:, :2] - np.asarray(pos, dtype=np.float64)[None, :2], axis=1)
    cbf_clearance = np.min(center_dists_xy - (float(obstacle_radius) + float(quad_radius) + float(obstacle_radius_margin)))
    return bool(cbf_clearance > 0.0)


def make_obstacle_cbf_filter(
    r_mismatch: float,
    obstacle_radius_margin: float,
    use_repeated_linearization: bool,
):
    def _filter(base_action: np.ndarray, env_state, _unused_swarm_state=None):
        env_unwrapped = env_state.unwrapped
        if not getattr(env_unwrapped, "use_obstacles", False) or env_unwrapped.obstacles is None:
            return np.asarray(base_action, dtype=np.float32)

        obstacle_centers = np.asarray(env_unwrapped.obstacles.pos_arr, dtype=np.float64).reshape(-1, 3)
        if obstacle_centers.size == 0:
            return np.asarray(base_action, dtype=np.float32)

        cbf_obstacle_radius = (
            float(env_unwrapped.obstacles.obstacle_radius)
            + float(env_unwrapped.quad_arm)
            + float(obstacle_radius_margin)
        )
        radii = np.full(obstacle_centers.shape[0], cbf_obstacle_radius, dtype=np.float64)
        swarm_state = SwarmState(
            positions=obstacle_centers.copy(),
            velocities=np.zeros_like(obstacle_centers, dtype=np.float64),
            rotations=np.tile(np.eye(3, dtype=np.float64), (obstacle_centers.shape[0], 1, 1)),
        )
        return apply_cbf_filter(
            base_action=base_action,
            radii=radii,
            r_mismatch=float(r_mismatch),
            env_state=env_unwrapped,
            swarm_state=swarm_state,
            use_repeated_linearization=use_repeated_linearization,
        )

    return _filter


def _state_model_mismatch(action: np.ndarray, dynamics, dt: float) -> float:
    action = np.asarray(action, dtype=np.float64)
    normalized = np.clip(0.5 * (action + 1.0), 0.0, 1.0)
    state_cbf = _pack_state_tuple(*cbf_dynamics(normalized, dynamics, dt))
    state_real = _pack_state_tuple(*real_dynamics(normalized, dynamics, dt))
    return float(np.linalg.norm(state_cbf - state_real))


def _cbf_predicted_next_state(action: np.ndarray, dynamics, dt: float, steps: int = 2) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    normalized = np.clip(0.5 * (action + 1.0), 0.0, 1.0)
    return _pack_state_tuple(*cbf_dynamics(normalized, dynamics, dt, steps=steps))


def _actual_state_from_env(env_unwrapped) -> np.ndarray:
    dynamics = env_unwrapped.envs[0].dynamics
    return _pack_state_tuple(dynamics.pos, dynamics.vel, dynamics.rot, dynamics.omega)


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


def _apply_authoritative_obstacle_environment(
    env,
    geometry: Dict[str, np.ndarray | float],
    point_towards_goal: bool,
    spawn_ball_radius: float,
    spawn_ball_max_tries: int,
    obstacle_radius_margin: float,
):
    env_unwrapped = env.unwrapped
    if not getattr(env_unwrapped, "use_obstacles", False):
        raise ValueError("collect_rand_obs requires an obstacle-enabled environment.")
    if len(env_unwrapped.envs) != 1:
        raise ValueError("collect_rand_obs currently supports exactly one quadrotor.")

    quad = env_unwrapped.envs[0]
    dynamics = quad.dynamics

    start_center = np.asarray(geometry["start_point"], dtype=np.float64).copy()
    vel = np.zeros(3, dtype=np.float64)
    omega = np.zeros(3, dtype=np.float64)
    goal = np.asarray(geometry["goal_point"], dtype=np.float64).copy()

    obstacle_positions = np.asarray(geometry["obstacle_positions"], dtype=np.float64).reshape(-1, 3)
    obstacle_radius = float(geometry["obstacle_radius"])
    env_unwrapped.num_obstacles = int(obstacle_positions.shape[0])
    env_unwrapped.obst_size = 2.0 * obstacle_radius
    env_unwrapped.obstacles.pos_arr = obstacle_positions.copy()
    env_unwrapped.obstacles.obstacle_radius = obstacle_radius
    env_unwrapped.obstacles.size = 2.0 * obstacle_radius

    quad_radius = float(env_unwrapped.quad_arm)
    pos = None
    for _ in range(int(spawn_ball_max_tries)):
        candidate = _sample_point_in_ball(start_center, float(spawn_ball_radius))
        if _spawn_is_valid_cbf_clearance(candidate, obstacle_positions, obstacle_radius, quad_radius, obstacle_radius_margin):
            pos = candidate
            break
    if pos is None:
        raise RuntimeError("Failed to sample a valid initial position outside the CBF obstacle clearance region.")

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


def _compute_obstacle_distance_metrics(
    pos: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radius: float,
    quad_radius: float,
    obstacle_radius_margin: float,
) -> tuple[float, float, float]:
    obstacle_positions = np.asarray(obstacle_positions, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(pos, dtype=np.float64)
    if obstacle_positions.size == 0:
        return float("inf"), float("inf"), float("inf")

    center_dists_xy = np.linalg.norm(obstacle_positions[:, :2] - pos[None, :2], axis=1)
    boundary_dist = float(np.min(center_dists_xy - float(obstacle_radius)))
    clearance = float(np.min(center_dists_xy - (float(obstacle_radius) + float(quad_radius))))
    cbf_clearance = float(
        np.min(center_dists_xy - (float(obstacle_radius) + float(quad_radius) + float(obstacle_radius_margin)))
    )
    return boundary_dist, clearance, cbf_clearance


def _obstacle_bubble_rgba(
    cbf_clearance: float,
    had_prior_violation: bool = False,
) -> tuple[float, float, float, float]:
    if float(cbf_clearance) <= 0.0:
        return DEFAULT_BUBBLE_VIOLATION_RGBA
    if bool(had_prior_violation):
        return DEFAULT_BUBBLE_RECOVERED_RGBA
    return DEFAULT_BUBBLE_OK_RGBA


def _obstacle_bubble_rgba_array(cbf_clearances: np.ndarray, prior_violations: np.ndarray | None = None) -> np.ndarray:
    cbf_clearances = np.asarray(cbf_clearances, dtype=np.float64).reshape(-1)
    if prior_violations is None:
        prior_violations = np.zeros(cbf_clearances.shape[0], dtype=bool)
    else:
        prior_violations = np.asarray(prior_violations, dtype=bool).reshape(-1)
        if prior_violations.shape[0] != cbf_clearances.shape[0]:
            raise ValueError("prior_violations must match cbf_clearances length.")
    colors = np.zeros((cbf_clearances.shape[0], 4), dtype=np.float64)
    for idx, clearance in enumerate(cbf_clearances):
        colors[idx, :] = np.asarray(
            _obstacle_bubble_rgba(float(clearance), had_prior_violation=bool(prior_violations[idx])),
            dtype=np.float64,
        )
    return colors


def _set_obstacle_overlay_state(
    env_unwrapped,
    obstacle_radius_margin: float,
    had_prior_violation: bool = False,
) -> float:
    quad_radius = float(env_unwrapped.quad_arm)
    bubble_radius = quad_radius + float(obstacle_radius_margin)
    obstacle_positions = np.asarray(env_unwrapped.obstacles.pos_arr, dtype=np.float64).reshape(-1, 3)
    obstacle_radius = float(env_unwrapped.obstacles.obstacle_radius)
    quad_pos = np.asarray(env_unwrapped.envs[0].dynamics.pos, dtype=np.float64)
    _, _, cbf_clearance = _compute_obstacle_distance_metrics(
        pos=quad_pos,
        obstacle_positions=obstacle_positions,
        obstacle_radius=obstacle_radius,
        quad_radius=quad_radius,
        obstacle_radius_margin=obstacle_radius_margin,
    )
    env_unwrapped.render_bubble_radius = bubble_radius
    env_unwrapped.render_bubble_rgba = _obstacle_bubble_rgba(
        cbf_clearance,
        had_prior_violation=had_prior_violation,
    )
    return cbf_clearance


def _build_eval_cfg(saved_args: Dict, enable_render: bool, view_mode: str, num_agents: int = 1) -> "AttrDict":
    spawn_area = saved_args["quads_obst_spawn_area"]
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_use_obstacles=True",
        f"--quads_mode={saved_args['quads_mode']}",
        f"--quads_num_agents={int(num_agents)}",
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
        f"--quads_use_downwash={bool(saved_args.get('use_downwash', False))}",
        f"--quads_use_wind={bool(saved_args.get('use_wind', False))}",
        f"--quads_wind_y_start={float(saved_args.get('wind_y_start', -2.0))}",
        f"--quads_wind_y_full={float(saved_args.get('wind_y_full', 3.0))}",
        f"--quads_wind_accel_x={float(saved_args.get('wind_accel_x', 0.33))}",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        f"--quads_render={bool(enable_render)}",
        f"--quads_view_mode={view_mode}",
    ]
    return parse_swarm_cfg(eval_cli, evaluation=True)


def _resolve_video_output(video_name: str | None, output_path: str) -> tuple[str, str] | None:
    if not video_name:
        return None
    if os.path.isabs(video_name):
        video_dir = os.path.dirname(video_name) or "."
        video_file = os.path.basename(video_name)
    else:
        output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        video_dir = output_dir
        video_file = video_name
    os.makedirs(video_dir, exist_ok=True)
    return video_dir, video_file


def _trajectory_video_path(
    video_target: tuple[str, str],
    traj_idx: int,
    num_trajectories: int,
    record_all_trajectories: bool,
) -> str:
    video_dir, video_file = video_target
    if (not record_all_trajectories) or num_trajectories <= 1:
        return os.path.abspath(os.path.join(video_dir, video_file))

    stem, ext = os.path.splitext(video_file)
    if ext == "":
        ext = ".mp4"
    digits = max(3, len(str(max(0, num_trajectories - 1))))
    suffixed = f"{stem}_traj{traj_idx:0{digits}d}{ext}"
    return os.path.abspath(os.path.join(video_dir, suffixed))


def _append_video_frame(env, video_frames: List[np.ndarray]) -> None:
    try:
        frame = env.render()
    except Exception as exc:
        raise RuntimeError(
            "collect_rand_obs video capture must use the normal simulator renderer, "
            "but env.render() failed. Start an X display (for example XLaunch) and make "
            "sure DISPLAY is set before running video capture."
        ) from exc

    if frame is None:
        raise RuntimeError("collect_rand_obs expected an rgb_array frame, but env.render() returned None.")

    video_frames.append(frame.copy())


def _save_video_frames(video_path: str, video_frames: List[np.ndarray], fps: int) -> None:
    if len(video_frames) == 0:
        raise ValueError("No video frames to save")

    if shutil.which("ffmpeg") is not None:
        video_dir = os.path.dirname(video_path) or "."
        video_file = os.path.basename(video_path)
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, fps, video_cfg)
        return

    import cv2

    first = np.asarray(video_frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"Expected frames with shape (H, W, 3), got {first.shape}")

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
                raise ValueError("All video frames must have the same resolution")
            writer.write(cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _apply_authoritative_obstacle_geometry_multi(env_unwrapped, geometry: Dict[str, np.ndarray | float]) -> None:
    if not getattr(env_unwrapped, "use_obstacles", False):
        raise ValueError("Composite obstacle replay requires an obstacle-enabled environment.")

    obstacle_positions = np.asarray(geometry["obstacle_positions"], dtype=np.float64).reshape(-1, 3)
    obstacle_radius = float(geometry["obstacle_radius"])
    env_unwrapped.num_obstacles = int(obstacle_positions.shape[0])
    env_unwrapped.obst_size = 2.0 * obstacle_radius
    env_unwrapped.obstacles.pos_arr = obstacle_positions.copy()
    env_unwrapped.obstacles.obstacle_radius = obstacle_radius
    env_unwrapped.obstacles.size = 2.0 * obstacle_radius


def _trajectory_state_for_replay_frame(traj: Dict, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    traj_len = int(traj["trajectory_length"])
    if frame_idx <= 0 or traj_len <= 0:
        return (
            np.asarray(traj["initial_position"], dtype=np.float64),
            np.asarray(traj["initial_velocity"], dtype=np.float64),
            np.asarray(traj["initial_rotation"], dtype=np.float64),
            np.asarray(traj["initial_omega"], dtype=np.float64),
            float(traj["initial_cbf_clearance"]),
            False,
        )

    step_idx = min(frame_idx - 1, traj_len - 1)
    return (
        np.asarray(traj["position"][step_idx], dtype=np.float64),
        np.asarray(traj["velocity"][step_idx], dtype=np.float64),
        np.asarray(traj["rotation"][step_idx], dtype=np.float64),
        np.asarray(traj["omega"][step_idx], dtype=np.float64),
        float(traj["cbf_clearance"][step_idx]),
        bool(traj["collision_obstacle"][step_idx]),
    )


def _trajectory_had_prior_obstacle_violation(traj: Dict, frame_idx: int) -> bool:
    if float(traj["initial_cbf_clearance"]) <= 0.0:
        return True

    traj_len = int(traj["trajectory_length"])
    if frame_idx <= 1 or traj_len <= 0:
        return False

    step_idx = min(frame_idx - 1, traj_len - 1)
    if step_idx <= 0:
        return False

    prior_clearances = np.asarray(traj["cbf_clearance"][:step_idx], dtype=np.float64)
    return bool(np.any(prior_clearances <= 0.0))


def _set_composite_replay_frame(
    env_unwrapped,
    trajectories: List[Dict],
    frame_idx: int,
    obstacle_radius_margin: float,
) -> None:
    num_agents = len(trajectories)
    if len(env_unwrapped.envs) != num_agents:
        raise ValueError("Composite replay env agent count does not match replay trajectory count.")

    obstacle_collisions = np.zeros(num_agents, dtype=np.float64)
    bubble_clearances = np.zeros(num_agents, dtype=np.float64)
    prior_violations = np.zeros(num_agents, dtype=bool)
    goals = []
    quad_radius = float(env_unwrapped.quad_arm)

    for agent_id, traj in enumerate(trajectories):
        pos, vel, rot, omega, cbf_clearance, obstacle_collision = _trajectory_state_for_replay_frame(traj, frame_idx)
        quad = env_unwrapped.envs[agent_id]
        quad.goal = np.asarray(traj["initial_goal"], dtype=np.float64).copy()
        quad.spawn_point = np.asarray(traj["initial_position"], dtype=np.float64).copy()
        quad.dynamics.set_state(pos, vel, rot, omega)
        quad.dynamics.on_floor = False
        quad.dynamics.crashed_floor = False
        quad.dynamics.crashed_wall = False
        quad.dynamics.crashed_ceiling = False
        quad.tick = int(frame_idx)
        quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]

        env_unwrapped.pos[agent_id, :] = quad.dynamics.pos
        env_unwrapped.vel[agent_id, :] = quad.dynamics.vel
        goals.append(np.asarray(quad.goal, dtype=np.float64).copy())
        bubble_clearances[agent_id] = cbf_clearance
        prior_violations[agent_id] = _trajectory_had_prior_obstacle_violation(traj, frame_idx)
        obstacle_collisions[agent_id] = 1.0 if obstacle_collision else 0.0

    env_unwrapped.render_bubble_radius = quad_radius + float(obstacle_radius_margin)
    env_unwrapped.render_bubble_rgba = _obstacle_bubble_rgba_array(
        bubble_clearances,
        prior_violations=prior_violations,
    )
    env_unwrapped.all_collisions = {
        "drone": np.zeros(num_agents, dtype=np.float64),
        "ground": np.zeros(num_agents, dtype=np.float64),
        "obstacle": obstacle_collisions,
    }
    if hasattr(env_unwrapped.scenario, "goals"):
        env_unwrapped.scenario.goals = np.asarray(goals, dtype=np.float64)


def _record_composite_obstacle_video(
    saved_args: Dict,
    geometry: Dict[str, np.ndarray | float],
    trajectories: List[Dict],
    obstacle_radius_margin: float,
    disable_boundary_collision: bool,
    video_path: str,
    video_fps: int,
    video_view_mode: str,
    dynamic_zoom: bool,
    zoom_scale: float,
    obstacle_rgb: tuple[float, float, float],
    background_rgb: tuple[float, float, float],
) -> None:
    if len(trajectories) == 0:
        raise ValueError("Composite obstacle replay requires at least one trajectory.")

    replay_cfg = _build_eval_cfg(
        saved_args,
        enable_render=True,
        view_mode=video_view_mode,
        num_agents=len(trajectories),
    )
    replay_env = make_quadrotor_env("quadrotor_multi", cfg=replay_cfg, render_mode="rgb_array")
    replay_env.unwrapped.render_floor_visible = False
    replay_env.unwrapped.render_walls_visible = False
    replay_env.unwrapped.render_dynamic_zoom = bool(dynamic_zoom)
    replay_env.unwrapped.render_zoom_scale = float(zoom_scale)
    replay_env.unwrapped.render_obstacle_rgba = tuple(float(x) for x in obstacle_rgb) + (1.0,)
    replay_env.unwrapped.render_bgcolor = tuple(float(x) for x in background_rgb)

    try:
        if disable_boundary_collision:
            _configure_far_boundary_geometry(replay_env.unwrapped)
        _reset_env(replay_env)
        if disable_boundary_collision:
            _configure_far_boundary_geometry(replay_env.unwrapped)
        _apply_authoritative_obstacle_geometry_multi(replay_env.unwrapped, geometry)

        max_len = max(int(traj["trajectory_length"]) for traj in trajectories)
        composite_frames: List[np.ndarray] = []
        for frame_idx in range(max_len + 1):
            _set_composite_replay_frame(
                replay_env.unwrapped,
                trajectories,
                frame_idx,
                obstacle_radius_margin=obstacle_radius_margin,
            )
            _append_video_frame(replay_env, composite_frames)

        _save_video_frames(video_path, composite_frames, video_fps)
    finally:
        replay_env.close()


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
    spawn_ball_radius: float,
    spawn_ball_max_tries: int,
    record_video: bool = False,
    video_frames: List[np.ndarray] | None = None,
):
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)

    _reset_env(env)
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)
    obs_run, initial_state = _apply_authoritative_obstacle_environment(
        env,
        geometry,
        point_towards_goal,
        spawn_ball_radius,
        spawn_ball_max_tries,
        obstacle_radius_margin,
    )
    done = False
    step_num = 0
    run_rnn_states = init_rnn_states.clone()
    held_nominal_action = None
    held_filtered_action = None
    initial_cbf_clearance = _set_obstacle_overlay_state(
        env.unwrapped,
        obstacle_radius_margin,
        had_prior_violation=False,
    )
    has_seen_bubble_violation = bool(initial_cbf_clearance <= 0.0)

    if record_video:
        if video_frames is None:
            raise ValueError("record_video=True requires video_frames")
        _append_video_frame(env, video_frames)

    run_logs: Dict[str, List] = {
        "position": [],
        "velocity": [],
        "rotation": [],
        "omega": [],
        "goal_dist": [],
        "collision_obstacle": [],
        "boundary_dist": [],
        "clearance": [],
        "cbf_clearance": [],
        "model_mismatch_state": [],
        "nominal_acceleration": [],
        "filtered_acceleration": [],
        "bubble_violation": [],
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
        wind_active = False
        if bool(getattr(env.unwrapped, "use_wind", False)) and hasattr(env.unwrapped, "_wind_accel_from_position"):
            wind_active = bool(np.any(np.asarray(env.unwrapped._wind_accel_from_position(dynamics.pos), dtype=np.float64)))
        use_true_step_mismatch = wind_active
        mismatch_state = None
        if use_true_step_mismatch:
            predicted_next = _cbf_predicted_next_state(
                filtered_action,
                dynamics,
                dt,
                steps=2,
            )
        else:
            mismatch_state = _state_model_mismatch(filtered_action, dynamics, dt)

        actions = filtered_action[None, :]
        obs_run, rewards, dones, infos = _step_env(env, actions)

        if use_true_step_mismatch:
            actual_next = _actual_state_from_env(env.unwrapped)
            proj_posvel = np.array([1.0] * 6 + [0.0] * 12)
            mismatch_state = float(np.linalg.norm(proj_posvel * (predicted_next - actual_next))) / dt

        pos, vel = extract_positions_velocities(env.unwrapped)
        solo_pos = np.asarray(pos[0], dtype=np.float64)
        solo_vel = np.asarray(vel[0], dtype=np.float64)
        goal = np.asarray(env.unwrapped.envs[0].goal, dtype=np.float64)
        goal_dist = float(np.linalg.norm(solo_pos - goal))

        obstacle_centers = np.asarray(env.unwrapped.obstacles.pos_arr, dtype=np.float64).reshape(-1, 3)
        obstacle_radius = float(env.unwrapped.obstacles.obstacle_radius)
        quad_radius = float(env.unwrapped.quad_arm)
        boundary_dist, clearance, cbf_clearance = _compute_obstacle_distance_metrics(
            pos=solo_pos,
            obstacle_positions=obstacle_centers,
            obstacle_radius=obstacle_radius,
            quad_radius=quad_radius,
            obstacle_radius_margin=obstacle_radius_margin,
        )
        env.unwrapped.render_bubble_radius = quad_radius + float(obstacle_radius_margin)
        env.unwrapped.render_bubble_rgba = _obstacle_bubble_rgba(
            cbf_clearance,
            had_prior_violation=has_seen_bubble_violation,
        )
        if cbf_clearance <= 0.0:
            has_seen_bubble_violation = True

        rew_obst_raw = infos[0]["rewards"].get("rewraw_quadcol_obstacle", 0.0)
        collision_obstacle = float(rew_obst_raw) < 0.0

        if record_video:
            _append_video_frame(env, video_frames)

        run_logs["position"].append(solo_pos.copy())
        run_logs["velocity"].append(solo_vel.copy())
        run_logs["rotation"].append(np.asarray(env.unwrapped.envs[0].dynamics.rot, dtype=np.float64).copy())
        run_logs["omega"].append(np.asarray(env.unwrapped.envs[0].dynamics.omega, dtype=np.float64).copy())
        run_logs["goal_dist"].append(goal_dist)
        run_logs["collision_obstacle"].append(collision_obstacle)
        run_logs["boundary_dist"].append(boundary_dist)
        run_logs["clearance"].append(clearance)
        run_logs["cbf_clearance"].append(cbf_clearance)
        run_logs["model_mismatch_state"].append(mismatch_state)
        run_logs["nominal_acceleration"].append(np.asarray(nominal_acc, dtype=np.float32))
        run_logs["filtered_acceleration"].append(np.asarray(filtered_acc, dtype=np.float32))
        run_logs["bubble_violation"].append(bool(cbf_clearance <= 0.0))

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
        "rotation": np.asarray(run_logs["rotation"], dtype=np.float32),
        "omega": np.asarray(run_logs["omega"], dtype=np.float32),
        "goal_dist": np.asarray(run_logs["goal_dist"], dtype=np.float32),
        "collision_obstacle": np.asarray(run_logs["collision_obstacle"], dtype=np.bool_),
        "boundary_dist": np.asarray(run_logs["boundary_dist"], dtype=np.float32),
        "clearance": np.asarray(run_logs["clearance"], dtype=np.float32),
        "cbf_clearance": np.asarray(run_logs["cbf_clearance"], dtype=np.float32),
        "model_mismatch_state": np.asarray(run_logs["model_mismatch_state"], dtype=np.float32),
        "nominal_acceleration": np.asarray(run_logs["nominal_acceleration"], dtype=np.float32),
        "filtered_acceleration": np.asarray(run_logs["filtered_acceleration"], dtype=np.float32),
        "bubble_violation": np.asarray(run_logs["bubble_violation"], dtype=np.bool_),
        "initial_position": np.asarray(initial_state["initial_position"], dtype=np.float64),
        "initial_goal": np.asarray(initial_state["initial_goal"], dtype=np.float64),
        "initial_velocity": np.asarray(initial_state["initial_velocity"], dtype=np.float64),
        "initial_omega": np.asarray(initial_state["initial_omega"], dtype=np.float64),
        "initial_rotation": np.asarray(initial_state["initial_rotation"], dtype=np.float64),
        "initial_cbf_clearance": float(initial_cbf_clearance),
        "trajectory_length": int(step_num),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.r_mismatch):
        raise ValueError("--r_mismatch must be positive or zero.")
    if args.num_trajectories <= 0:
        raise ValueError("--num_trajectories must be positive.")
    if args.action_repeat <= 0:
        raise ValueError("--action_repeat must be positive.")
    if args.spawn_ball_radius < 0.0:
        raise ValueError("--spawn_ball_radius must be non-negative.")
    if args.spawn_ball_max_tries <= 0:
        raise ValueError("--spawn_ball_max_tries must be positive.")
    if args.video_fps <= 0:
        raise ValueError("--video_fps must be positive.")
    if args.video_trajectory_idx < -1:
        raise ValueError("--video_trajectory_idx must be >= -1.")
    if args.video_zoom_scale <= 0.0:
        raise ValueError("--video_zoom_scale must be positive.")

    saved_args = _load_json(args.conf_rand_obs_args)
    obstacle_radius_margin = float(saved_args["obstacle_radius_margin"])
    episode_length = int(saved_args["episode_length"])
    deterministic = bool(saved_args.get("deterministic", False))
    disable_boundary_collision = bool(saved_args.get("disable_boundary_collision", False))
    use_downwash = bool(saved_args.get("use_downwash", False))

    if episode_length <= 0:
        raise ValueError("Saved --episode_length must be positive.")

    torch.set_grad_enabled(False)
    register_swarm_components()
    set_global_seed(args.seed)

    cfg_solo = load_cfg(saved_args["solo_train_dir"], saved_args["solo_experiment"])
    obstacle_rgb = _parse_video_color_arg(args.video_obstacle_color, "--video_obstacle_color")
    background_rgb = _parse_video_color_arg(args.video_background_color, "--video_background_color")
    multi_traj_video = args.num_trajectories > 1
    individual_video_name = args.video_name
    composite_video_name = args.video_composite_name
    auto_promoted_composite = False
    if multi_traj_video and individual_video_name is not None and composite_video_name is None:
        composite_video_name = individual_video_name
        individual_video_name = None
        auto_promoted_composite = True

    record_video = individual_video_name is not None
    video_target = _resolve_video_output(individual_video_name, args.output_path)
    composite_video_target = _resolve_video_output(composite_video_name, args.output_path)
    record_all_trajectories = record_video and args.video_trajectory_idx == -1
    if record_video:
        if record_all_trajectories:
            print("[collect_rand_obs] Recording individual videos for all collected trajectories.")
        else:
            if args.video_trajectory_idx >= args.num_trajectories:
                raise ValueError("--video_trajectory_idx must be smaller than --num_trajectories, or -1 for all.")
            print(f"[collect_rand_obs] Recording individual video only for trajectory {args.video_trajectory_idx}.")
    if composite_video_target is not None and multi_traj_video:
        if auto_promoted_composite and args.video_trajectory_idx == 0:
            print("[collect_rand_obs] Recording a composite replay video covering all collected trajectories.")
        elif args.video_trajectory_idx == -1:
            print("[collect_rand_obs] Recording a composite replay video covering all collected trajectories.")
        else:
            print(
                f"[collect_rand_obs] Recording a composite replay video only for trajectory {args.video_trajectory_idx}."
            )

    eval_cfg = _build_eval_cfg(saved_args, enable_render=record_video, view_mode=args.video_view_mode)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode="rgb_array" if record_video else None)
    if record_video:
        env.unwrapped.render_floor_visible = False
        env.unwrapped.render_walls_visible = False
        env.unwrapped.render_dynamic_zoom = bool(args.dynamic_zoom)
        env.unwrapped.render_zoom_scale = float(args.video_zoom_scale)
        env.unwrapped.render_obstacle_rgba = tuple(float(x) for x in obstacle_rgb) + (1.0,)
        env.unwrapped.render_bgcolor = tuple(float(x) for x in background_rgb)

    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)
    _reset_env(env)
    if disable_boundary_collision:
        _configure_far_boundary_geometry(env.unwrapped)
    if args.conformal_obstacles_environment is not None:
        geometry = _load_environment_geometry(args.conformal_obstacles_environment)
    else:
        geometry = _capture_runtime_geometry(env)

    solo_ckpt = latest_checkpoint(saved_args["solo_train_dir"], saved_args["solo_experiment"], policy_index=0)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)
    filter_fn = make_obstacle_cbf_filter(
        args.r_mismatch,
        obstacle_radius_margin,
        bool(args.use_repeated_linearization),
    )

    trajectories = []
    video_frame_sets: List[tuple[int, List[np.ndarray]]] = []
    progress_bar = tqdm(range(args.num_trajectories))
    for traj_idx in progress_bar:
        should_record_this_trajectory = record_video and (
            record_all_trajectories or traj_idx == args.video_trajectory_idx
        )
        traj_video_frames: List[np.ndarray] | None = [] if should_record_this_trajectory else None
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
            spawn_ball_radius=float(args.spawn_ball_radius),
            spawn_ball_max_tries=int(args.spawn_ball_max_tries),
            record_video=should_record_this_trajectory,
            video_frames=traj_video_frames,
        )
        trajectories.append(traj)
        if traj_video_frames is not None:
            video_frame_sets.append((traj_idx, traj_video_frames))
        progress_bar.set_postfix_str(f"last_len={traj['trajectory_length']}")

    max_len = max(traj["trajectory_length"] for traj in trajectories)
    positions = np.stack([_pad_with_fill(traj["position"], max_len) for traj in trajectories], axis=0)
    velocities = np.stack([_pad_with_fill(traj["velocity"], max_len) for traj in trajectories], axis=0)
    rotations = np.stack([_pad_with_fill(traj["rotation"], max_len) for traj in trajectories], axis=0)
    omegas = np.stack([_pad_with_fill(traj["omega"], max_len) for traj in trajectories], axis=0)
    goal_dist = np.stack([_pad_with_fill(traj["goal_dist"], max_len) for traj in trajectories], axis=0)
    collision_obstacle = np.stack([_pad_with_fill(traj["collision_obstacle"], max_len) for traj in trajectories], axis=0)
    boundary_dist = np.stack([_pad_with_fill(traj["boundary_dist"], max_len) for traj in trajectories], axis=0)
    clearance = np.stack([_pad_with_fill(traj["clearance"], max_len) for traj in trajectories], axis=0)
    cbf_clearance = np.stack([_pad_with_fill(traj["cbf_clearance"], max_len) for traj in trajectories], axis=0)
    model_mismatch_state = np.stack([_pad_with_fill(traj["model_mismatch_state"], max_len) for traj in trajectories], axis=0)
    nominal_accelerations = np.stack([_pad_with_fill(traj["nominal_acceleration"], max_len) for traj in trajectories], axis=0)
    filtered_accelerations = np.stack([_pad_with_fill(traj["filtered_acceleration"], max_len) for traj in trajectories], axis=0)
    bubble_violation = np.stack([_pad_with_fill(traj["bubble_violation"], max_len) for traj in trajectories], axis=0)
    trajectory_lengths = np.asarray([traj["trajectory_length"] for traj in trajectories], dtype=np.int32)
    initial_positions = np.stack([traj["initial_position"] for traj in trajectories], axis=0)
    initial_goals = np.stack([traj["initial_goal"] for traj in trajectories], axis=0)
    initial_velocities = np.stack([traj["initial_velocity"] for traj in trajectories], axis=0)
    initial_omegas = np.stack([traj["initial_omega"] for traj in trajectories], axis=0)
    initial_rotations = np.stack([traj["initial_rotation"] for traj in trajectories], axis=0)
    initial_cbf_clearance = np.asarray([traj["initial_cbf_clearance"] for traj in trajectories], dtype=np.float32)

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
        rotations=rotations,
        omegas=omegas,
        goal_dist=goal_dist,
        collision_obstacle=collision_obstacle,
        boundary_dist=boundary_dist,
        clearance=clearance,
        cbf_clearance=cbf_clearance,
        model_mismatch_state=model_mismatch_state,
        nominal_accelerations=nominal_accelerations,
        filtered_accelerations=filtered_accelerations,
        bubble_violation=bubble_violation,
        trajectory_lengths=trajectory_lengths,
        initial_positions=initial_positions,
        initial_goals=initial_goals,
        initial_velocities=initial_velocities,
        initial_omegas=initial_omegas,
        initial_rotations=initial_rotations,
        initial_cbf_clearance=initial_cbf_clearance,
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
        deterministic=deterministic,
        disable_boundary_collision=disable_boundary_collision,
        use_downwash=use_downwash,
        obstacle_radius_margin=obstacle_radius_margin,
        point_towards_goal=bool(args.point_towards_goal),
        action_repeat=int(args.action_repeat),
        spawn_ball_radius=float(args.spawn_ball_radius),
        spawn_ball_max_tries=int(args.spawn_ball_max_tries),
        conformal_obstacles_environment_path=np.array(
            os.path.abspath(args.conformal_obstacles_environment) if args.conformal_obstacles_environment is not None else ""
        ),
        conf_rand_obs_args_path=os.path.abspath(args.conf_rand_obs_args),
    )

    env.close()

    print(f"[collect_rand_obs] Saved {args.num_trajectories} trajectories to {output_path}")
    print(f"[collect_rand_obs] Max trajectory length: {max_len}")
    print(f"[collect_rand_obs] Obstacle count: {np.asarray(geometry['obstacle_positions']).shape[0]}")
    print(f"[collect_rand_obs] point_towards_goal={bool(args.point_towards_goal)}")
    print(f"[collect_rand_obs] action_repeat={int(args.action_repeat)}")
    if video_target is not None:
        for traj_idx, video_frames in video_frame_sets:
            if len(video_frames) == 0:
                continue
            final_video_path = _trajectory_video_path(
                video_target,
                traj_idx,
                len(video_frame_sets),
                record_all_trajectories=record_all_trajectories,
            )
            _save_video_frames(final_video_path, video_frames, args.video_fps)
            print(f"[collect_rand_obs] Saved trajectory video to {final_video_path}")
    if composite_video_target is not None:
        if auto_promoted_composite and args.video_trajectory_idx == 0:
            composite_indices = list(range(len(trajectories)))
        elif args.video_trajectory_idx == -1:
            composite_indices = list(range(len(trajectories)))
        else:
            if args.video_trajectory_idx >= len(trajectories):
                raise ValueError("--video_trajectory_idx must be smaller than the number of collected trajectories.")
            composite_indices = [args.video_trajectory_idx]
        composite_runs = [trajectories[idx] for idx in composite_indices]
        composite_video_path = _trajectory_video_path(
            composite_video_target,
            traj_idx=0,
            num_trajectories=1,
            record_all_trajectories=False,
        )
        _record_composite_obstacle_video(
            saved_args=saved_args,
            geometry=geometry,
            trajectories=composite_runs,
            obstacle_radius_margin=obstacle_radius_margin,
            disable_boundary_collision=disable_boundary_collision,
            video_path=composite_video_path,
            video_fps=args.video_fps,
            video_view_mode=args.video_view_mode,
            dynamic_zoom=bool(args.dynamic_zoom),
            zoom_scale=float(args.video_zoom_scale),
            obstacle_rgb=obstacle_rgb,
            background_rgb=background_rgb,
        )
        print(f"[collect_rand_obs] Saved composite trajectory video to {composite_video_path}")


if __name__ == "__main__":
    main()
