#!/usr/bin/env python3
"""
Generate conformal experiment plots from saved metrics.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Enable LaTeX rendering for labels and legends if available.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot conformal experiment results from saved metrics.")
    parser.add_argument(
        "--plot_data",
        required=True,
        help="Path to conformal_metrics.npz produced by conformal.py.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save plots (default: <plot_data_dir>/plots).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = os.path.abspath(args.plot_data)
    data_dir = os.path.dirname(data_path)
    output_dir = args.output_dir or os.path.join(data_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    with np.load(data_path) as data:
        episodes = data["episodes"]
        radius_arr = data["radius_per_episode"]
        qj_arr = data["qj_per_episode"]
        tube_coverage_arr = data["tube_coverage_per_episode"]
        crashes_arr = data["crashes_per_episode"]
        bad_crashes_arr = data["bad_crashes_per_episode"]
        cumulative_reward_arr = data["cumulative_reward_per_episode"]
        safety_arr = data["safety_per_episode"]
        alpha = float(data.get("alpha"))
        bar_alpha = float(data.get("bar_alpha"))
        agent_locs_per_episode = data.get("agent_locs_per_episode", None)
        predicted_traj_per_episode = data["predicted_traj_per_episode"]
    print(crashes_arr)
    print([np.max(agent_locs_per_episode[0, agent_id] - agent_locs_per_episode[9, agent_id]) for agent_id in range(5)])

    plot_paths = {}
    if len(episodes) > 0:
        # tick_step = 2 if len(episodes) % 2 == 0 else 1
        x_ticks = np.arange(0, len(episodes), 1)
        x_lim = (0, len(episodes) - 1)

        # Plot A: Radius across episodes (rj and qj)
        fig, ax = plt.subplots()
        ax.plot(episodes, radius_arr, label=r"$r_j$", marker='s')
        ax.plot(episodes, qj_arr, label=r"$q_j$ ($1 - \bar \alpha$ quantile)", marker='o')
        ax.set_title(r"Radius Across Episodes")
        ax.set_xlabel(r"Episode ($j$)")
        ax.set_xlim(*x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(r"Radius ($m$)")
        ax.legend()
        radius_plot_path = os.path.join(output_dir, "radius_across_episodes.png")
        fig.savefig(radius_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["radius"] = radius_plot_path

        # Plot B: Performance across episodes (cumulative reward)
        fig, ax = plt.subplots()
        ax.plot(episodes, cumulative_reward_arr, label=r"Cumulative progress towards goal", marker='s')
        ax.set_title(r"Performance Across Episodes")
        ax.set_xlabel(r"Episode ($j$)")
        ax.set_xlim(*x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(r"Cumulative reward ($m$)")
        ax.legend(loc='center right')
        perf_plot_path = os.path.join(output_dir, "performance_cumulative_reward.png")
        fig.savefig(perf_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["performance"] = perf_plot_path

        # Plot C: Empirical tube coverage
        fig, ax = plt.subplots()
        ax.plot(episodes, tube_coverage_arr, label=r"Tube coverage", marker='s')
        target_line = (1 - alpha)
        ax.axhline(target_line, linestyle="--", color="gray", label=r"Target $(1 - \alpha)$")
        ax.set_title(r"Empirical Tube Coverage")
        ax.set_xlabel(r"Episode ($j$)")
        ax.set_xlim(*x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(r"Coverage (\%)")
        ax.legend()
        tube_plot_path = os.path.join(output_dir, "tube_coverage.png")
        fig.savefig(tube_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["tube_coverage"] = tube_plot_path

        # Plot D: Empirical safety coverage (bad crashes)
        fig, ax = plt.subplots()
        ax.plot(episodes, safety_arr, label=r"Empirical safety", marker='s')
        ax.axhline(target_line, linestyle="--", color="gray", label=r"Target $(1 - \alpha)$")
        ax.set_title(r"Empirical Safety Coverage")
        ax.set_xlabel(r"Episode ($j$)")
        ax.set_xlim(*x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(r"Coverage (\%)")
        ax.legend()
        safety_plot_path = os.path.join(output_dir, "empirical_safety_coverage.png")
        fig.savefig(safety_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["safety"] = safety_plot_path

        # Plot E: 3D trajectories for a chosen episode (if available)
        ep_idx_list = [0, len(episodes) // 2, len(episodes) - 1]
        for episode_idx in ep_idx_list:
            if 0 <= episode_idx < agent_locs_per_episode.shape[0]:
                trajs = agent_locs_per_episode[episode_idx]
                num_agents, num_steps, _ = trajs.shape
                solo_idx = num_agents - 1

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                for agent_id in range(num_agents):
                    traj = trajs[agent_id]
                    color = "tab:red" if agent_id == solo_idx else None
                    label = "Solo" if agent_id == solo_idx else f"Agent {agent_id}"
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=label, color=color)
                # Overlay predicted trajectories for the multi agents (dotted lines)
                pred_all = predicted_traj_per_episode[episode_idx]  # shape: num_multi_agents x steps x 6
                num_pred_agents = pred_all.shape[0]
                for agent_id in range(min(num_agents - 1, num_pred_agents)):
                    pred_traj = pred_all[agent_id]
                    ax.plot(
                        pred_traj[:, 0],
                        pred_traj[:, 1],
                        pred_traj[:, 2],
                        linestyle=":",
                        color="tab:blue",
                        label="Predicted (multi)" if agent_id == 0 else None,
                    )
                ax.set_title(rf"3D Trajectories (Episode {episode_idx})")
                ax.set_xlabel(r"$x$ (m)")
                ax.set_ylabel(r"$y$ (m)")
                ax.set_zlabel(r"$z$ (m)")
                ax.legend()
                traj_plot_path = os.path.join(output_dir, f"trajectories_episode_{episode_idx}.png")
                fig.savefig(traj_plot_path, bbox_inches="tight")
                plt.close(fig)
                plot_paths[f"trajectories_{episode_idx}"] = traj_plot_path
            else:
                print(f"[plot_conformal] episode_idx {episode_idx} out of range for trajectories; skipping 3D plot")

    print(f"[plot_conformal] Loaded data from {data_path}")
    for plot_name, plot_path in plot_paths.items():
        print(f"[plot_conformal] Saved {plot_name} plot to {plot_path}")


if __name__ == "__main__":
    main()
