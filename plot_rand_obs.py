#!/usr/bin/env python3
"""Plot all trajectories from collect_rand_obs output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectories from collect_rand_obs.")
    parser.add_argument(
        "--plot_data",
        required=True,
        help="Path to trajectory dataset produced by collect_rand_obs.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save trajectory plot.",
    )
    parser.add_argument(
        "--output_name",
        default="rand_obs_trajectories.pdf",
        help="Output file name.",
    )
    parser.add_argument(
        "--highlight_distance",
        type=float,
        default=1.0,
        help="Obstacles within this XY distance (meters) of any plotted trajectory are rendered darker.",
    )
    parser.add_argument(
        "--2d",
        dest="plot_2d",
        action="store_true",
        help="Render the old top-down 2D plot instead of the default 3D scene.",
    )
    parser.add_argument(
        "--action_stride",
        type=int,
        default=-1,
        help="Plot a small predicted-acceleration arrow every t timesteps; t < 0 disables arrows.",
    )
    parser.add_argument(
        "--action_source",
        choices=("filtered", "nominal", "both"),
        default="filtered",
        help="Which predicted acceleration arrows to draw when --action_stride >= 1.",
    )
    parser.add_argument(
        "--color_by",
        choices=("trajectory", "quad"),
        default="trajectory",
        help="Color trajectories by trajectory id or by quad id. In obstacle plots there is one quad, so quad coloring uses one shared color.",
    )
    return parser.parse_args()


def draw_vertical_cylinder(
    ax,
    center_xy: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
    color: str = "gray",
    alpha: float = 0.15,
):
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    z = np.linspace(z_min, z_max, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center_xy[0] + radius * np.cos(theta_grid)
    y_grid = center_xy[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)


def infer_room_height_from_collection(obstacle_positions: np.ndarray) -> float:
    if obstacle_positions.size > 0:
        return max(float(np.max(obstacle_positions[:, 2])) * 2.0, 1.0)
    return 10.0


def point_to_segment_distance_2d(point_xy: np.ndarray, seg_start_xy: np.ndarray, seg_end_xy: np.ndarray) -> float:
    seg = seg_end_xy - seg_start_xy
    seg_norm_sq = float(np.dot(seg, seg))
    if seg_norm_sq <= 1e-12:
        return float(np.linalg.norm(point_xy - seg_start_xy))
    t = float(np.dot(point_xy - seg_start_xy, seg) / seg_norm_sq)
    t = np.clip(t, 0.0, 1.0)
    proj = seg_start_xy + t * seg
    return float(np.linalg.norm(point_xy - proj))


def obstacle_near_trajectory_xy(center_xy: np.ndarray, trajectory_xyz: np.ndarray, threshold: float) -> bool:
    traj_xy = np.asarray(trajectory_xyz[:, :2], dtype=np.float64)
    if traj_xy.shape[0] == 0:
        return False
    if traj_xy.shape[0] == 1:
        return float(np.linalg.norm(center_xy - traj_xy[0])) <= threshold

    for idx in range(traj_xy.shape[0] - 1):
        dist = point_to_segment_distance_2d(center_xy, traj_xy[idx], traj_xy[idx + 1])
        if dist <= threshold:
            return True
    return False


def obstacle_near_any_trajectory(
    center_xy: np.ndarray,
    positions: np.ndarray,
    trajectory_lengths: np.ndarray,
    threshold: float,
) -> bool:
    max_steps = positions.shape[1]
    for run_id in range(positions.shape[0]):
        run_len = int(trajectory_lengths[run_id])
        run_len = max(0, min(run_len, max_steps))
        if run_len <= 0:
            continue
        traj = positions[run_id, :run_len, :]
        if obstacle_near_trajectory_xy(center_xy, traj, threshold):
            return True
    return False


def point_near_any_obstacle_xy(
    point_xyz: np.ndarray,
    obstacle_positions: np.ndarray,
    threshold: float,
) -> bool:
    if obstacle_positions.size == 0:
        return False
    point_xy = np.asarray(point_xyz[:2], dtype=np.float64)
    obstacle_xy = np.asarray(obstacle_positions[:, :2], dtype=np.float64)
    dists = np.linalg.norm(obstacle_xy - point_xy[None, :], axis=1)
    return bool(np.any(dists <= threshold))


def _finite_xyz_bounds(
    positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radius: float,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    room_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    xyz_points = []

    flat = positions.reshape(-1, positions.shape[-1])
    finite_mask = np.isfinite(flat).all(axis=1)
    if np.any(finite_mask):
        xyz_points.append(flat[finite_mask])

    xyz_points.append(np.asarray(start_point, dtype=np.float64)[None, :])
    xyz_points.append(np.asarray(goal_point, dtype=np.float64)[None, :])

    if obstacle_positions.size > 0:
        obstacle_bottom = obstacle_positions.copy()
        obstacle_bottom[:, 2] = 0.0
        obstacle_top = obstacle_positions.copy()
        obstacle_top[:, 2] = room_height
        xyz_points.append(obstacle_bottom)
        xyz_points.append(obstacle_top)
        xy_plus = obstacle_positions[:, :2] + obstacle_radius
        xy_minus = obstacle_positions[:, :2] - obstacle_radius
        z_mid = obstacle_positions[:, 2:3]
        xyz_points.append(np.concatenate([xy_plus, z_mid], axis=1))
        xyz_points.append(np.concatenate([xy_minus, z_mid], axis=1))

    all_xyz = np.concatenate(xyz_points, axis=0)
    return np.min(all_xyz, axis=0), np.max(all_xyz, axis=0)


def _finite_xy_bounds(
    positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radius: float,
    start_point: np.ndarray,
    goal_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xy_points = []

    flat = positions.reshape(-1, positions.shape[-1])
    finite_mask = np.isfinite(flat).all(axis=1)
    if np.any(finite_mask):
        xy_points.append(flat[finite_mask][:, :2])

    if obstacle_positions.size > 0:
        obst_xy = obstacle_positions[:, :2]
        xy_points.append(obst_xy + obstacle_radius)
        xy_points.append(obst_xy - obstacle_radius)

    xy_points.append(np.asarray(start_point[:2], dtype=np.float64)[None, :])
    xy_points.append(np.asarray(goal_point[:2], dtype=np.float64)[None, :])

    all_xy = np.concatenate(xy_points, axis=0)
    return np.min(all_xy, axis=0), np.max(all_xy, axis=0)


def _iter_acceleration_arrows(
    base_positions: np.ndarray,
    accelerations: np.ndarray,
    trajectory_lengths: np.ndarray,
    obstacle_positions: np.ndarray,
    highlight_distance: float,
    stride: int,
    arrow_length: float,
):
    if stride < 1:
        return
    max_steps = base_positions.shape[1]
    for run_id in range(base_positions.shape[0]):
        run_len = int(trajectory_lengths[run_id])
        run_len = max(0, min(run_len, max_steps))
        for step in range(0, run_len, stride):
            p = base_positions[run_id, step]
            a = accelerations[run_id, step]
            if not np.isfinite(p).all() or not np.isfinite(a).all():
                continue
            if not point_near_any_obstacle_xy(p, obstacle_positions, highlight_distance):
                continue
            a_norm = float(np.linalg.norm(a))
            if a_norm <= 1e-8:
                continue
            delta = (a / a_norm) * arrow_length
            yield p, delta


def _prediction_bases_from_rollout(positions: np.ndarray, initial_positions: np.ndarray) -> np.ndarray:
    base_positions = np.full_like(positions, np.nan)
    if positions.shape[0] != initial_positions.shape[0]:
        raise ValueError("initial_positions length must match number of trajectories.")
    base_positions[:, 0, :] = initial_positions
    if positions.shape[1] > 1:
        base_positions[:, 1:, :] = positions[:, :-1, :]
    return base_positions


def _make_palette(count: int) -> np.ndarray:
    if count <= 20:
        return plt.cm.tab20(np.linspace(0.0, 1.0, count))
    return plt.cm.plasma(np.linspace(0.0, 1.0, count))


def _color_for_run(run_id: int, color_by: str, colors: np.ndarray) -> np.ndarray:
    if color_by == "trajectory":
        return colors[run_id % len(colors)]
    return colors[0]


def _add_action_arrows_2d(
    ax,
    base_positions: np.ndarray,
    accelerations: np.ndarray,
    trajectory_lengths: np.ndarray,
    obstacle_positions: np.ndarray,
    highlight_distance: float,
    stride: int,
    color: str,
    alpha: float,
    zorder: int,
    arrow_length: float,
) -> None:
    for p, delta in _iter_acceleration_arrows(
        base_positions,
        accelerations,
        trajectory_lengths,
        obstacle_positions,
        highlight_distance,
        stride,
        arrow_length,
    ):
        ax.arrow(
            p[0],
            p[1],
            delta[0],
            delta[1],
            length_includes_head=True,
            head_width=0.05,
            head_length=0.08,
            linewidth=0.4,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )


def _add_action_arrows_3d(
    ax,
    base_positions: np.ndarray,
    accelerations: np.ndarray,
    trajectory_lengths: np.ndarray,
    obstacle_positions: np.ndarray,
    highlight_distance: float,
    stride: int,
    color: str,
    alpha: float,
    arrow_length: float,
) -> None:
    for p, delta in _iter_acceleration_arrows(
        base_positions,
        accelerations,
        trajectory_lengths,
        obstacle_positions,
        highlight_distance,
        stride,
        arrow_length,
    ):
        ax.quiver(
            p[0],
            p[1],
            p[2],
            delta[0],
            delta[1],
            delta[2],
            length=1.0,
            normalize=False,
            arrow_length_ratio=0.25,
            color=color,
            alpha=alpha,
            linewidth=0.4,
        )


def _plot_2d(
    output_path: str,
    positions: np.ndarray,
    trajectory_lengths: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radius: float,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    r_mismatch: float,
    seed: int,
    disable_boundary_collision: bool,
    point_towards_goal: bool,
    highlight_distance: float,
    action_stride: int,
    action_source: str,
    color_by: str,
    initial_positions: np.ndarray,
    nominal_accelerations: np.ndarray | None,
    filtered_accelerations: np.ndarray | None,
) -> None:
    num_runs = positions.shape[0]
    max_steps = positions.shape[1]
    colors = _make_palette(num_runs if color_by == "trajectory" else 1)

    fig, ax = plt.subplots(figsize=(9, 9))

    for obstacle_center in obstacle_positions:
        theta = np.linspace(0.0, 2.0 * np.pi, 100)
        x = obstacle_center[0] + obstacle_radius * np.cos(theta)
        y = obstacle_center[1] + obstacle_radius * np.sin(theta)
        ax.fill(x, y, color="0.75", alpha=0.9, zorder=1)
        ax.plot(x, y, color="0.35", linewidth=1.0, zorder=1)

    for run_id in range(num_runs):
        run_len = int(trajectory_lengths[run_id])
        run_len = max(0, min(run_len, max_steps))
        if run_len <= 0:
            continue
        traj = positions[run_id, :run_len, :]
        color = _color_for_run(run_id, color_by, colors)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.28, color=color, zorder=2)
        ax.scatter(traj[0, 0], traj[0, 1], s=10, alpha=0.45, color=color, zorder=3)

    base_positions = _prediction_bases_from_rollout(positions, initial_positions)
    arrow_length = 0.35
    if action_source in ("nominal", "both") and nominal_accelerations is not None:
        _add_action_arrows_2d(
            ax,
            base_positions,
            nominal_accelerations,
            trajectory_lengths,
            obstacle_positions,
            highlight_distance,
            action_stride,
            color="tab:blue",
            alpha=0.24,
            zorder=3,
            arrow_length=arrow_length,
        )
    if action_source in ("filtered", "both") and filtered_accelerations is not None:
        _add_action_arrows_2d(
            ax,
            base_positions,
            filtered_accelerations,
            trajectory_lengths,
            obstacle_positions,
            highlight_distance,
            action_stride,
            color="black",
            alpha=0.22,
            zorder=4,
            arrow_length=arrow_length,
        )

    ax.scatter(
        start_point[0],
        start_point[1],
        s=90,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=4,
        label="start",
    )
    ax.scatter(
        goal_point[0],
        goal_point[1],
        s=110,
        marker="*",
        facecolor="gold",
        edgecolor="black",
        linewidth=0.9,
        zorder=4,
        label="goal",
    )

    min_xy, max_xy = _finite_xy_bounds(
        positions=positions,
        obstacle_positions=obstacle_positions,
        obstacle_radius=obstacle_radius,
        start_point=start_point,
        goal_point=goal_point,
    )
    if np.all(np.isclose(min_xy, max_xy)):
        margin = 1.0
        min_xy = min_xy - margin
        max_xy = max_xy + margin

    center = 0.5 * (min_xy + max_xy)
    half_range = 0.5 * np.max(max_xy - min_xy)
    half_range = max(half_range, 1.0)
    margin = 0.08 * (2.0 * half_range)
    ax.set_xlim(center[0] - half_range - margin, center[0] + half_range + margin)
    ax.set_ylim(center[1] - half_range - margin, center[1] + half_range + margin)
    ax.set_aspect("equal", adjustable="box")

    boundary_tag = "far boundaries" if disable_boundary_collision else "normal room"
    heading_tag = ", yaw-to-goal" if point_towards_goal else ""
    ax.set_title(
        rf"Obstacle CBF trajectories ($r_{{mismatch}}={r_mismatch:.4g}$, seed={seed}, {boundary_tag}{heading_tag})"
    )
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.legend(loc="best")

    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)


def _plot_3d(
    output_path: str,
    positions: np.ndarray,
    trajectory_lengths: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radius: float,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    r_mismatch: float,
    seed: int,
    disable_boundary_collision: bool,
    point_towards_goal: bool,
    highlight_distance: float,
    action_stride: int,
    action_source: str,
    color_by: str,
    initial_positions: np.ndarray,
    nominal_accelerations: np.ndarray | None,
    filtered_accelerations: np.ndarray | None,
) -> None:
    num_runs = positions.shape[0]
    max_steps = positions.shape[1]
    room_height = infer_room_height_from_collection(obstacle_positions)
    cylinder_z_min, cylinder_z_max = 0.0, room_height
    colors = _make_palette(num_runs if color_by == "trajectory" else 1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for run_id in range(num_runs):
        run_len = int(trajectory_lengths[run_id])
        run_len = max(0, min(run_len, max_steps))
        if run_len <= 0:
            continue
        traj = positions[run_id, :run_len, :]
        color = _color_for_run(run_id, color_by, colors)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.12, linewidth=0.9)

    base_positions = _prediction_bases_from_rollout(positions, initial_positions)
    arrow_length = 0.45
    if action_source in ("nominal", "both") and nominal_accelerations is not None:
        _add_action_arrows_3d(
            ax,
            base_positions,
            nominal_accelerations,
            trajectory_lengths,
            obstacle_positions,
            highlight_distance,
            action_stride,
            color="tab:blue",
            alpha=0.22,
            arrow_length=arrow_length,
        )
    if action_source in ("filtered", "both") and filtered_accelerations is not None:
        _add_action_arrows_3d(
            ax,
            base_positions,
            filtered_accelerations,
            trajectory_lengths,
            obstacle_positions,
            highlight_distance,
            action_stride,
            color="black",
            alpha=0.18,
            arrow_length=arrow_length,
        )

    ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], color="tab:green", marker="o", s=30, label="Start")
    ax.scatter([goal_point[0]], [goal_point[1]], [goal_point[2]], color="tab:blue", marker="*", s=60, label="Goal")

    for obs_idx, center in enumerate(obstacle_positions):
        center_xy = np.asarray(center[:2], dtype=np.float64)
        is_near = obstacle_near_any_trajectory(center_xy, positions, trajectory_lengths, highlight_distance)
        cylinder_color = "dimgray" if is_near else "lightgray"
        cylinder_alpha = 0.45 if is_near else 0.08
        draw_vertical_cylinder(
            ax,
            center_xy=center_xy,
            radius=obstacle_radius,
            z_min=cylinder_z_min,
            z_max=cylinder_z_max,
            color=cylinder_color,
            alpha=cylinder_alpha,
        )
        marker_color = "black" if is_near else "gray"
        marker_size = 10 if is_near else 6
        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            color=marker_color,
            s=marker_size,
            label="Obstacle center" if obs_idx == 0 else None,
        )

    min_xyz, max_xyz = _finite_xyz_bounds(
        positions=positions,
        obstacle_positions=obstacle_positions,
        obstacle_radius=obstacle_radius,
        start_point=start_point,
        goal_point=goal_point,
        room_height=room_height,
    )
    center_xyz = 0.5 * (min_xyz + max_xyz)
    xy_half_range = 0.5 * max(max_xyz[0] - min_xyz[0], max_xyz[1] - min_xyz[1])
    xy_half_range = max(xy_half_range, 1.0)
    xy_margin = 0.08 * (2.0 * xy_half_range)
    ax.set_xlim(center_xyz[0] - xy_half_range - xy_margin, center_xyz[0] + xy_half_range + xy_margin)
    ax.set_ylim(center_xyz[1] - xy_half_range - xy_margin, center_xyz[1] + xy_half_range + xy_margin)
    ax.set_zlim(cylinder_z_min, cylinder_z_max)

    boundary_tag = "far boundaries" if disable_boundary_collision else "normal room"
    heading_tag = ", yaw-to-goal" if point_towards_goal else ""
    ax.set_title(
        rf"3D obstacle trajectories ($r_{{mismatch}}={r_mismatch:.4g}$, seed={seed}, {boundary_tag}{heading_tag})"
    )
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_zlabel(r"$z$ (m)")
    ax.legend()

    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = os.path.abspath(args.plot_data)
    with np.load(data_path) as data:
        positions = data["positions"]
        trajectory_lengths = data["trajectory_lengths"]
        obstacle_positions = data["obstacle_positions"]
        obstacle_radius = float(data["obstacle_radius"])
        start_point = data["start_point"]
        goal_point = data["goal_point"]
        initial_positions = data["initial_positions"]
        r_mismatch = float(data.get("r_mismatch", 0.0))
        seed = int(data.get("seed", 0))
        disable_boundary_collision = bool(data.get("disable_boundary_collision", False))
        point_towards_goal = bool(data.get("point_towards_goal", False))
        nominal_accelerations = data["nominal_accelerations"] if "nominal_accelerations" in data.files else None
        filtered_accelerations = data["filtered_accelerations"] if "filtered_accelerations" in data.files else None

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (num_trajectories, T, 3).")
    if trajectory_lengths.shape[0] != positions.shape[0]:
        raise ValueError("trajectory_lengths length must match number of trajectories.")
    if obstacle_positions.ndim != 2 or obstacle_positions.shape[1] != 3:
        raise ValueError("obstacle_positions must have shape (num_obstacles, 3).")
    if start_point.shape != (3,) or goal_point.shape != (3,):
        raise ValueError("start_point and goal_point must each have shape (3,).")
    if initial_positions.shape != (positions.shape[0], 3):
        raise ValueError("initial_positions must have shape (num_trajectories, 3).")

    output_dir = args.output_dir or os.path.join("./", "plots", Path(data_path).stem)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)

    if args.plot_2d:
        _plot_2d(
            output_path=output_path,
            positions=positions,
            trajectory_lengths=trajectory_lengths,
            obstacle_positions=obstacle_positions,
            obstacle_radius=obstacle_radius,
            start_point=start_point,
            goal_point=goal_point,
            r_mismatch=r_mismatch,
            seed=seed,
            disable_boundary_collision=disable_boundary_collision,
            point_towards_goal=point_towards_goal,
            highlight_distance=args.highlight_distance,
            action_stride=args.action_stride,
            action_source=args.action_source,
            color_by=args.color_by,
            initial_positions=initial_positions,
            nominal_accelerations=nominal_accelerations,
            filtered_accelerations=filtered_accelerations,
        )
    else:
        _plot_3d(
            output_path=output_path,
            positions=positions,
            trajectory_lengths=trajectory_lengths,
            obstacle_positions=obstacle_positions,
            obstacle_radius=obstacle_radius,
            start_point=start_point,
            goal_point=goal_point,
            r_mismatch=r_mismatch,
            seed=seed,
            disable_boundary_collision=disable_boundary_collision,
            point_towards_goal=point_towards_goal,
            highlight_distance=args.highlight_distance,
            action_stride=args.action_stride,
            action_source=args.action_source,
            color_by=args.color_by,
            initial_positions=initial_positions,
            nominal_accelerations=nominal_accelerations,
            filtered_accelerations=filtered_accelerations,
        )

    print(f"[plot_rand_obs] Loaded {positions.shape[0]} trajectories from {data_path}")
    print(f"[plot_rand_obs] Saved trajectory plot to {output_path}")


if __name__ == "__main__":
    main()
