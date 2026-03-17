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
        mismatch_arr = data["mismatch_per_episode"]
        alpha = float(data["alpha"])
        agent_locs = data["agent_locs_first_run"]

    num_episodes = len(episodes)
    x_radius = episodes
    x_other = episodes[1:]
    x_radius_ticks = np.arange(0, num_episodes, 1)
    x_other_ticks = np.arange(1, num_episodes, 1)
    x_radius_lim = (0, max(num_episodes - 1, 0))
    x_other_lim = (1, max(num_episodes - 1, 1))

    plot_paths = {}

    fig, ax = plt.subplots()
    shifted_radius_arr = np.insert(radius_arr, 0, 2)[:-1]
    ax.plot(x_radius, shifted_radius_arr, label=r"$r_j$", marker="s")
    ax.plot(x_radius, qj_arr, label=r"$q_j$ ($1 - \bar \alpha$ quantile)", marker="o")
    ax.set_title(r"Radius Across Episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_radius_lim)
    ax.set_xticks(x_radius_ticks)
    ax.set_ylabel(r"Radius ($m$)")
    ax.legend()
    path = os.path.join(output_dir, "radius_across_episodes.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["radius"] = path

    fig, ax = plt.subplots()
    alpha_for_error = args.alpha if args.alpha is not None else alpha
    lower = np.quantile(cumulative_reward_runs, alpha_for_error, axis=1)[:-1]
    upper = np.quantile(cumulative_reward_runs, 1 - alpha_for_error, axis=1)[:-1]
    plot_reward = cumulative_reward_arr[:-1]
    lower_plot = lower
    upper_plot = upper
    yerr = np.vstack([
        np.maximum(plot_reward - lower_plot, 0),
        np.maximum(upper_plot - plot_reward, 0),
    ])
    ax.errorbar(
        x_other,
        plot_reward,
        yerr=yerr,
        label=rf"Cumulative progress ({alpha_for_error:.2g}/{1 - alpha_for_error:.2g} quantiles)",
        marker="s",
        capsize=4,
    )
    ax.set_title(r"Performance Across Episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel(r"Cumulative reward ($m$)")
    ax.legend(loc="center right")
    path = os.path.join(output_dir, "performance_cumulative_reward.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["performance"] = path

    fig, ax = plt.subplots()
    ax.plot(x_other, crashes_arr[:-1], label="Crash rate", marker="s")
    ax.set_title("Crash Rate Across Episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel("Crash rate")
    ax.legend()
    path = os.path.join(output_dir, "crash_rate.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["crash_rate"] = path

    fig, ax = plt.subplots()
    ax.plot(x_other, mismatch_arr[:-1], label="Max state mismatch", marker="s")
    ax.set_title("Mismatch Across Episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
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
