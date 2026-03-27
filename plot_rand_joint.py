#!/usr/bin/env python3
"""Plot all trajectories from collect_rand_joint output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectories from collect_rand_joint.")
    parser.add_argument(
        "--plot_data",
        required=True,
        help="Path to trajectory dataset produced by collect_rand_joint.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save trajectory plot.",
    )
    parser.add_argument(
        "--output_name",
        default="rand_joint_trajectories.pdf",
        help="Output file name.",
    )
    parser.add_argument(
        "--color_by",
        choices=("trajectory", "quad"),
        default="trajectory",
        help="Color lines by trajectory id or by quad id.",
    )
    return parser.parse_args()


def _finite_bounds(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = positions.reshape(-1, positions.shape[-1])
    finite_mask = np.isfinite(flat).all(axis=1)
    if not np.any(finite_mask):
        return np.zeros((3,), dtype=np.float64), np.ones((3,), dtype=np.float64)
    finite = flat[finite_mask]
    return np.min(finite, axis=0), np.max(finite, axis=0)


def _make_palette(count: int) -> np.ndarray:
    if count <= 20:
        return plt.cm.tab20(np.linspace(0.0, 1.0, count))
    return plt.cm.plasma(np.linspace(0.0, 1.0, count))


def main() -> None:
    args = parse_args()
    data_path = os.path.abspath(args.plot_data)
    with np.load(data_path) as data:
        positions = data["positions"]
        trajectory_lengths = data["trajectory_lengths"]
        r_mismatch = float(data.get("r_mismatch", 0.0))
        seed = int(data.get("seed", 0))
        num_agents = int(data["num_agents"])
        disable_boundary_collision = bool(data.get("disable_boundary_collision", False))

    if positions.ndim != 4 or positions.shape[2] != num_agents:
        raise ValueError("positions must have shape (num_trajectories, T, num_agents, 3).")
    if trajectory_lengths.shape[0] != positions.shape[0]:
        raise ValueError("trajectory_lengths length must match number of trajectories.")

    num_runs = positions.shape[0]
    max_steps = positions.shape[1]
    palette_size = num_runs if args.color_by == "trajectory" else num_agents
    colors = _make_palette(palette_size)

    output_dir = args.output_dir or os.path.join("./", "plots", Path(data_path).stem)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for run_id in range(num_runs):
        run_len = int(trajectory_lengths[run_id])
        run_len = max(1, min(run_len, max_steps))
        for agent_id in range(num_agents):
            traj = positions[run_id, :run_len, agent_id, :]
            if traj.shape[0] < 2:
                continue
            color_idx = run_id if args.color_by == "trajectory" else agent_id
            color = colors[color_idx % len(colors)]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                linewidth=0.9,
                alpha=0.25,
                color=color,
            )
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=10, alpha=0.4, color=color)

    min_xyz, max_xyz = _finite_bounds(positions)
    if np.all(np.isclose(min_xyz, max_xyz)):
        margin = 1.0
        center = min_xyz
        min_xyz = center - margin
        max_xyz = center + margin
    center = (min_xyz + max_xyz) / 2.0
    half_range = np.max(max_xyz - min_xyz) / 2.0
    half_range = max(half_range, 1.0)
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)

    boundary_tag = "far boundaries" if disable_boundary_collision else "normal room"
    ax.set_title(
        rf"Joint CBF trajectories ($r_{{mismatch}}={r_mismatch:.4g}$, seed={seed}, {boundary_tag})"
    )
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_zlabel(r"$z$ (m)")

    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"[plot_rand_joint] Loaded {num_runs} trajectories from {data_path}")
    print(f"[plot_rand_joint] Saved trajectory plot to {output_path}")


if __name__ == "__main__":
    main()
