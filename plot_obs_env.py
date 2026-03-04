#!/usr/bin/env python3
"""Plot obstacle environment geometry without trajectory data."""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Plot obstacle environment geometry from JSON.")
    parser.add_argument(
        "--env_geometry",
        required=True,
        help="Path to conformal_obstacles_environment.json.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save the plot.",
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
    theta = np.linspace(0, 2 * np.pi, 36)
    z = np.linspace(z_min, z_max, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center_xy[0] + radius * np.cos(theta_grid)
    y_grid = center_xy[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)


def infer_room_height(env: dict, obstacle_positions: np.ndarray) -> float:
    room_dims = env.get("room_dims")
    if isinstance(room_dims, list) and len(room_dims) >= 3:
        return float(room_dims[2])
    if obstacle_positions.size > 0:
        return max(float(np.max(obstacle_positions[:, 2])) * 2.0, 1.0)
    return 10.0


def main() -> None:
    args = parse_args()
    env_path = os.path.abspath(args.env_geometry)
    out_dir = args.output_dir or os.path.join("./", "plots", Path(env_path).stem)
    os.makedirs(out_dir, exist_ok=True)

    with open(env_path, "r", encoding="utf-8") as f:
        env = json.load(f)

    start_point = np.asarray(env.get("start_point", [0.0, 0.0, 0.0]), dtype=np.float32)
    goal_point = np.asarray(env.get("goal_point", [0.0, 0.0, 0.0]), dtype=np.float32)
    obstacle_positions = np.asarray(env.get("obstacle_positions", []), dtype=np.float32)
    obstacle_radius = float(env.get("obstacle_radius", 0.0))
    room_height = infer_room_height(env, obstacle_positions)
    z_min, z_max = 0.0, room_height

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], color="tab:green", marker="o", s=45, label="Start")
    ax.scatter([goal_point[0]], [goal_point[1]], [goal_point[2]], color="tab:blue", marker="*", s=80, label="Goal")

    for obs_idx, center in enumerate(obstacle_positions):
        draw_vertical_cylinder(
            ax,
            center_xy=np.asarray(center[:2], dtype=np.float32),
            radius=obstacle_radius,
            z_min=z_min,
            z_max=z_max,
        )
        ax.scatter(
            [center[0]], [center[1]], [center[2]],
            color="black", s=10,
            label="Obstacle center" if obs_idx == 0 else None,
        )

    ax.set_title("Obstacle Environment Geometry")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_zlabel(r"$z$ (m)")
    ax.set_zlim(z_min, z_max)
    ax.legend()

    out_path = os.path.join(out_dir, "obstacle_environment.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[plot_obs_env] Loaded geometry from {env_path}")
    print(f"[plot_obs_env] Saved environment plot to {out_path}")


if __name__ == "__main__":
    main()
