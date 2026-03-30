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
        choices=("trajectory", "quad", "safety", "mismatch"),
        default="trajectory",
        help=(
            "Color lines by trajectory id, by quad id, by minimum pairwise separation margin "
            "(green=safe, red if h<0), or by maximum nominal/true dynamics mismatch."
        ),
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


def _lerp_rgba(start: np.ndarray, end: np.ndarray, weight: float) -> np.ndarray:
    weight = float(np.clip(weight, 0.0, 1.0))
    return (1.0 - weight) * start + weight * end


def _trajectory_scalar_over_prefix(
    values: np.ndarray,
    trajectory_lengths: np.ndarray,
    reducer,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (num_trajectories, T).")
    if arr.shape[0] != trajectory_lengths.shape[0]:
        raise ValueError(f"{name} length must match number of trajectories.")
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for run_id in range(arr.shape[0]):
        run_len = max(0, min(int(trajectory_lengths[run_id]), arr.shape[1]))
        if run_len <= 0:
            continue
        out[run_id] = float(reducer(arr[run_id, :run_len]))
    return out


def _trajectory_min_pairwise_clearance(
    positions: np.ndarray,
    trajectory_lengths: np.ndarray,
    separation_radius: float,
) -> np.ndarray:
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (num_trajectories, T, num_agents, 3).")
    if positions.shape[0] != trajectory_lengths.shape[0]:
        raise ValueError("trajectory_lengths length must match number of trajectories.")

    num_runs, max_steps, num_agents, _ = positions.shape
    out = np.full(num_runs, np.nan, dtype=np.float64)
    if num_agents < 2:
        return out

    for run_id in range(num_runs):
        run_len = max(0, min(int(trajectory_lengths[run_id]), max_steps))
        if run_len <= 0:
            continue
        min_margin = float("inf")
        for step in range(run_len):
            pts = positions[run_id, step]
            if not np.isfinite(pts).all():
                continue
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    margin = float(np.linalg.norm(pts[i] - pts[j]) - separation_radius)
                    if margin < min_margin:
                        min_margin = margin
        out[run_id] = min_margin if np.isfinite(min_margin) else np.nan
    return out


def _safety_run_colors(
    positions: np.ndarray,
    trajectory_lengths: np.ndarray,
    separation_radius: float,
) -> np.ndarray:
    green = np.array([0.172, 0.627, 0.172, 1.0], dtype=np.float64)
    orange = np.array([1.0, 0.549, 0.0, 1.0], dtype=np.float64)
    red = np.array([0.839, 0.153, 0.157, 1.0], dtype=np.float64)
    min_clearance = _trajectory_min_pairwise_clearance(
        positions=positions,
        trajectory_lengths=trajectory_lengths,
        separation_radius=separation_radius,
    )
    colors = np.tile(green, (trajectory_lengths.shape[0], 1))
    for run_id, clearance in enumerate(min_clearance):
        if not np.isfinite(clearance):
            colors[run_id] = green
        elif clearance < 0.0:
            colors[run_id] = red
        elif clearance >= 0.5:
            colors[run_id] = green
        else:
            weight = float(np.clip(clearance / 0.5, 0.0, 1.0))
            colors[run_id] = _lerp_rgba(orange, green, weight)
    return colors


def _mismatch_run_colors(model_mismatch_state: np.ndarray, trajectory_lengths: np.ndarray) -> np.ndarray:
    green = np.array([0.172, 0.627, 0.172, 1.0], dtype=np.float64)
    red = np.array([0.839, 0.153, 0.157, 1.0], dtype=np.float64)
    max_mismatch = _trajectory_scalar_over_prefix(
        model_mismatch_state,
        trajectory_lengths,
        np.max,
        "model_mismatch_state",
    )
    finite_values = max_mismatch[np.isfinite(max_mismatch)]
    if finite_values.size == 0:
        return np.tile(green, (trajectory_lengths.shape[0], 1))
    mismatch_scale = max(float(np.percentile(finite_values, 95.0)), 1e-12)
    colors = np.tile(green, (trajectory_lengths.shape[0], 1))
    for run_id, mismatch in enumerate(max_mismatch):
        if not np.isfinite(mismatch):
            colors[run_id] = green
            continue
        weight = float(np.clip(mismatch / mismatch_scale, 0.0, 1.0))
        colors[run_id] = _lerp_rgba(green, red, weight)
    return colors


def _run_colors(
    color_by: str,
    positions: np.ndarray,
    trajectory_lengths: np.ndarray,
    model_mismatch_state: np.ndarray | None,
    separation_radius: float | None,
) -> np.ndarray | None:
    num_runs = positions.shape[0]
    if color_by == "trajectory":
        return _make_palette(num_runs)
    if color_by == "quad":
        return None
    if color_by == "safety":
        if separation_radius is None:
            raise ValueError("plot data is missing separation_radius required for --color_by safety.")
        return _safety_run_colors(
            positions=positions,
            trajectory_lengths=trajectory_lengths,
            separation_radius=separation_radius,
        )
    if color_by == "mismatch":
        if model_mismatch_state is None:
            raise ValueError("plot data is missing model_mismatch_state required for --color_by mismatch.")
        return _mismatch_run_colors(model_mismatch_state, trajectory_lengths)
    raise ValueError(f"Unsupported color_by mode: {color_by}")


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
        model_mismatch_state = data["model_mismatch_state"] if "model_mismatch_state" in data.files else None
        separation_radius = float(data["separation_radius"]) if "separation_radius" in data.files else None

    if positions.ndim != 4 or positions.shape[2] != num_agents:
        raise ValueError("positions must have shape (num_trajectories, T, num_agents, 3).")
    if trajectory_lengths.shape[0] != positions.shape[0]:
        raise ValueError("trajectory_lengths length must match number of trajectories.")

    num_runs = positions.shape[0]
    max_steps = positions.shape[1]
    quad_colors = _make_palette(num_agents)
    run_colors = _run_colors(
        color_by=args.color_by,
        positions=positions,
        trajectory_lengths=trajectory_lengths,
        model_mismatch_state=model_mismatch_state,
        separation_radius=separation_radius,
    )

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
            if args.color_by == "quad":
                color = quad_colors[agent_id % len(quad_colors)]
            else:
                color = run_colors[run_id]
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
        rf"Joint CBF trajectories ($r={r_mismatch:.4g}$)"
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
