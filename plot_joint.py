#!/usr/bin/env python3
"""Generate conformal joint experiment plots from saved metrics."""

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
    parser = argparse.ArgumentParser(description="Plot conformal joint experiment results.")
    parser.add_argument(
        "--plot_data",
        required=True,
        help="Path to conformal_joint_metrics.npz produced by conformal_joint.py.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Alpha for performance error bars; defaults to the value stored in the metrics file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = os.path.abspath(args.plot_data)
    exp_name = Path(data_path).stem
    output_dir = args.output_dir or os.path.join("./", "plots", exp_name)
    os.makedirs(output_dir, exist_ok=True)

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
        alpha = float(data["alpha"])
        agent_locs = data["agent_locs_first_run"]

    num_episodes = len(episodes)
    x_ticks = np.arange(0, num_episodes, 1)
    x_lim = (0, max(num_episodes - 1, 0))
    target_alpha = alpha if alpha is not None else 0.1
    target_line = 1 - target_alpha

    plot_paths = {}

    fig, ax = plt.subplots()
    shifted_radius_arr = np.insert(radius_arr, 0, 8)[:-1]
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

    for episode_idx in range(num_episodes):
        trajs = agent_locs[episode_idx]
        num_agents = trajs.shape[1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for agent_id in range(num_agents):
            traj = trajs[:, agent_id, :]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Agent {agent_id}")
            start = traj[0]
            ax.scatter([start[0]], [start[1]], [start[2]], s=20)

        ax.set_title(rf"3D Joint Trajectories (Episode {episode_idx + 1})")
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$y$ (m)")
        ax.set_zlabel(r"$z$ (m)")
        if episode_idx == 0:
            ax.legend()

        path = os.path.join(output_dir, f"trajectories_episode_{episode_idx + 1}.pdf")
        fig.savefig(path, bbox_inches="tight", format="pdf")
        plt.close(fig)
        plot_paths[f"trajectories_{episode_idx + 1}"] = path

    print(f"[plot_joint] Loaded data from {data_path}")
    for plot_name, plot_path in plot_paths.items():
        print(f"[plot_joint] Saved {plot_name} plot to {plot_path}")


if __name__ == "__main__":
    main()
