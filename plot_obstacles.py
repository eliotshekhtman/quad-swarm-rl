#!/usr/bin/env python3
"""Generate conformal obstacle experiment plots from saved metrics."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot conformal obstacle experiment results.")
    parser.add_argument(
        "--plot_data",
        required=True,
        help="Path to conformal_obstacles_metrics.npz produced by conformal_obstacles.py.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--env_geometry",
        help="Path to conformal_obstacles_environment.json. Defaults to sibling file next to --plot_data.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Alpha for performance error bars; defaults to the value stored in the metrics file.",
    )
    return parser.parse_args()


def draw_sphere(ax, center: np.ndarray, radius: float, color: str = "gray", alpha: float = 0.15):
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def _load_geometry(plot_data_path: str, env_geometry_path: Optional[str]) -> dict:
    if env_geometry_path is None:
        env_geometry_path = os.path.join(os.path.dirname(plot_data_path), "conformal_obstacles_environment.json")
    if not os.path.exists(env_geometry_path):
        raise FileNotFoundError(
            f"Could not find obstacle geometry JSON at {env_geometry_path}. "
            "Pass --env_geometry explicitly."
        )
    with open(env_geometry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    data_path = os.path.abspath(args.plot_data)
    exp_name = Path(data_path).stem
    output_dir = args.output_dir or os.path.join("./", "plots", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    geometry = _load_geometry(data_path, args.env_geometry)
    obstacle_positions = np.asarray(geometry.get("obstacle_positions", []), dtype=np.float32)
    obstacle_radius = float(geometry.get("obstacle_radius", 0.0))
    start_point = np.asarray(geometry.get("start_point", [0.0, 0.0, 0.0]), dtype=np.float32)
    goal_point = np.asarray(geometry.get("goal_point", [0.0, 0.0, 0.0]), dtype=np.float32)

    with np.load(data_path) as data:
        episodes = data["episodes"]
        if "r_mismatch_per_episode" in data.files:
            radius_arr = data["r_mismatch_per_episode"]
        elif "radius_per_episode" in data.files:
            radius_arr = data["radius_per_episode"]
        else:
            raise KeyError("Expected r_mismatch_per_episode (or radius_per_episode) in metrics file.")
        qj_arr = data["qj_per_episode"]
        crashes_arr = data["crashes_per_episode"]
        cumulative_reward_arr = data["cumulative_reward_per_episode"]
        cumulative_reward_runs = data["cumulative_reward_per_run"]
        safety_arr = data["safety_per_episode"]
        mismatch_arr = data["mismatch_per_episode"]
        min_clearance_arr = data["min_clearance_per_episode"]
        alpha = float(data["alpha"])
        solo_positions = data["solo_positions_first_run"]

    num_episodes = len(episodes)
    x_ticks = np.arange(0, num_episodes, 1)
    x_lim = (0, max(num_episodes - 1, 0))
    target_alpha = alpha if alpha is not None else 0.1
    target_line = 1 - target_alpha

    plot_paths = {}

    fig, ax = plt.subplots()
    shifted_radius_arr = np.insert(radius_arr, 0, 2)[:-1]
    ax.plot(episodes, shifted_radius_arr, label=r"$r_j$", marker="s")
    ax.plot(episodes, qj_arr, label=r"$q_j$ ($1 - \bar \alpha$ quantile)", marker="o")
    ax.set_title(r"Radius Across Episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel(r"Radius ($m$)")
    ax.legend()
    path = os.path.join(output_dir, "radius_across_episodes.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["radius"] = path

    fig, ax = plt.subplots()
    alpha_for_error = args.alpha if args.alpha is not None else alpha
    lower = np.quantile(cumulative_reward_runs, alpha_for_error, axis=1)
    upper = np.quantile(cumulative_reward_runs, 1 - alpha_for_error, axis=1)
    yerr = np.vstack([
        np.maximum(cumulative_reward_arr - lower, 0),
        np.maximum(upper - cumulative_reward_arr, 0),
    ])
    ax.errorbar(
        episodes,
        cumulative_reward_arr,
        yerr=yerr,
        label=rf"Cumulative progress ({alpha_for_error:.2g}/{1 - alpha_for_error:.2g} quantiles)",
        marker="s",
        capsize=4,
    )
    ax.set_title(r"Performance Across Episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel(r"Cumulative reward ($m$)")
    ax.legend(loc="center right")
    path = os.path.join(output_dir, "performance_cumulative_reward.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["performance"] = path

    fig, ax = plt.subplots()
    ax.plot(episodes, safety_arr, label=r"Empirical safety", marker="s")
    ax.axhline(target_line, linestyle="--", color="gray", label=r"Target $(1 - \alpha)$")
    ax.set_title(r"Empirical Safety Coverage")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel(r"Coverage (\%)")
    ax.legend()
    path = os.path.join(output_dir, "empirical_safety_coverage.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["safety"] = path

    fig, ax = plt.subplots()
    ax.plot(episodes, crashes_arr, label="Crash rate", marker="s")
    ax.set_title("Crash Rate Across Episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel("Crash rate")
    ax.legend()
    path = os.path.join(output_dir, "crash_rate.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["crash_rate"] = path

    fig, ax = plt.subplots()
    ax.plot(episodes, mismatch_arr, label="Max state mismatch", marker="s")
    ax.set_title("Mismatch Across Episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel("Mismatch")
    ax.legend()
    path = os.path.join(output_dir, "mismatch_across_episodes.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["mismatch"] = path

    fig, ax = plt.subplots()
    ax.plot(episodes, min_clearance_arr, label="Min obstacle clearance", marker="s")
    ax.axhline(0.0, linestyle="--", color="gray", label="Collision boundary")
    ax.set_title("Clearance Across Episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel("Clearance (m)")
    ax.legend()
    path = os.path.join(output_dir, "clearance_across_episodes.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["clearance"] = path

    for episode_idx in range(num_episodes):
        traj = solo_positions[episode_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:red", label=f"Ego (radius {radius_arr[episode_idx]:.3g})")
        ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], color="tab:green", marker="o", s=30, label="Start")
        ax.scatter([goal_point[0]], [goal_point[1]], [goal_point[2]], color="tab:blue", marker="*", s=60, label="Goal")

        for obs_idx, center in enumerate(obstacle_positions):
            draw_sphere(ax, center=center, radius=obstacle_radius)
            ax.scatter([center[0]], [center[1]], [center[2]], color="black", s=8, label="Obstacle center" if obs_idx == 0 else None)

        ax.set_title(rf"3D Solo Trajectory with Obstacles (Episode {episode_idx + 1})")
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$y$ (m)")
        ax.set_zlabel(r"$z$ (m)")
        if episode_idx == 0:
            ax.legend()

        path = os.path.join(output_dir, f"trajectories_episode_{episode_idx + 1}.pdf")
        fig.savefig(path, bbox_inches="tight", format="pdf")
        plt.close(fig)
        plot_paths[f"trajectories_{episode_idx + 1}"] = path

    print(f"[plot_obstacles] Loaded data from {data_path}")
    for plot_name, plot_path in plot_paths.items():
        print(f"[plot_obstacles] Saved {plot_name} plot to {plot_path}")


if __name__ == "__main__":
    main()
