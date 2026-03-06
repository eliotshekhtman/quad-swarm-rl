#!/usr/bin/env python3
"""Combine joint conformal baselines into non-trajectory across-episode plots."""

import argparse
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

ROBUST_INITIAL_RADIUS = 2.0
NAIVE_INITIAL_RADIUS = 2.0
NONROBUST_INITIAL_RADIUS = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined joint conformal metrics.")
    parser.add_argument("--robust", required=True, help="Path to robust conformal_joint metrics .npz.")
    parser.add_argument("--naive", required=True, help="Path to naive conformal_joint metrics .npz.")
    parser.add_argument("--nonrobust", required=True, help="Path to nonrobust conformal_joint metrics .npz.")
    parser.add_argument("--output_dir", help="Directory to save plots.")
    parser.add_argument("--alpha", type=float, help="Alpha for performance error bars; defaults to robust alpha.")
    return parser.parse_args()


def load_run(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        if "r_mismatch_per_episode" not in data.files:
            raise KeyError(f"{path}: expected r_mismatch_per_episode.")
        radius = data["r_mismatch_per_episode"]

        return {
            "episodes": data["episodes"],
            "radius": radius,
            "qj": data["qj_per_episode"],
            "crashes": data["crashes_per_episode"],
            "safety": data["safety_per_episode"],
            "cumulative_reward": data["cumulative_reward_per_episode"],
            "cumulative_reward_runs": data["cumulative_reward_per_run"],
            "mismatch": data["mismatch_per_episode"],
            "mismatch_runs": data["mismatch_per_run"],
            "h_violation": data["h_violation_per_episode"],
            "h_violation_runs": data["h_violation_per_run"],
            "alpha": float(data["alpha"]) if "alpha" in data.files else None,
        }


def prepare_radius(radius: np.ndarray, initial_radius: float) -> np.ndarray:
    return np.insert(np.asarray(radius), 0, initial_radius)[:-1]


def pad_repeat(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] <= 0:
        raise ValueError("Cannot pad empty array.")
    if arr.shape[0] > target_len:
        raise ValueError(f"Array length {arr.shape[0]} exceeds target length {target_len}.")
    if arr.shape[0] == target_len:
        return arr
    pad = np.repeat(arr[-1:], target_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def ensure_len(name: str, arr: np.ndarray, expected: int) -> None:
    if np.asarray(arr).shape[0] != expected:
        raise ValueError(f"{name} has length {np.asarray(arr).shape[0]}, expected {expected}.")


def validate_episode_grids(robust: Dict[str, np.ndarray], naive: Dict[str, np.ndarray], nonrobust: Dict[str, np.ndarray]) -> None:
    robust_eps = np.asarray(robust["episodes"])
    naive_eps = np.asarray(naive["episodes"])
    nonrobust_eps = np.asarray(nonrobust["episodes"])

    if robust_eps.shape[0] != naive_eps.shape[0]:
        raise ValueError("Naive run episode count does not match robust run.")
    if not np.array_equal(robust_eps, naive_eps):
        raise ValueError("Naive and robust episode grids differ; this is not allowed.")
    if not np.array_equal(robust_eps, np.arange(robust_eps.shape[0])):
        raise ValueError("Robust episode grid must be np.arange(num_episodes).")
    if nonrobust_eps.shape[0] != 1 or not np.array_equal(nonrobust_eps, np.arange(1)):
        raise ValueError("Nonrobust run must be single-episode with episodes=np.arange(1).")


def main() -> None:
    args = parse_args()

    robust = load_run(args.robust)
    naive = load_run(args.naive)
    nonrobust = load_run(args.nonrobust)

    validate_episode_grids(robust, naive, nonrobust)

    base_episodes = np.asarray(robust["episodes"])
    num_episodes = base_episodes.shape[0]
    if num_episodes < 1:
        raise ValueError("Robust run has no episodes.")

    ensure_len("robust radius", robust["radius"], num_episodes)
    ensure_len("robust qj", robust["qj"], num_episodes)
    ensure_len("robust crashes", robust["crashes"], num_episodes)
    ensure_len("robust cumulative_reward", robust["cumulative_reward"], num_episodes)
    ensure_len("robust mismatch", robust["mismatch"], num_episodes)
    ensure_len("robust mismatch_runs", robust["mismatch_runs"], num_episodes)
    ensure_len("robust h_violation", robust["h_violation"], num_episodes)
    ensure_len("robust h_violation_runs", robust["h_violation_runs"], num_episodes)
    ensure_len("robust cumulative_reward_per_run", robust["cumulative_reward_runs"], num_episodes)

    ensure_len("naive radius", naive["radius"], num_episodes)
    ensure_len("naive qj", naive["qj"], num_episodes)
    ensure_len("naive crashes", naive["crashes"], num_episodes)
    ensure_len("naive cumulative_reward", naive["cumulative_reward"], num_episodes)
    ensure_len("naive mismatch", naive["mismatch"], num_episodes)
    ensure_len("naive mismatch_runs", naive["mismatch_runs"], num_episodes)
    ensure_len("naive h_violation", naive["h_violation"], num_episodes)
    ensure_len("naive h_violation_runs", naive["h_violation_runs"], num_episodes)
    ensure_len("naive cumulative_reward_per_run", naive["cumulative_reward_runs"], num_episodes)

    nonrobust_radius = pad_repeat(prepare_radius(nonrobust["radius"], NONROBUST_INITIAL_RADIUS), num_episodes)
    nonrobust_qj = pad_repeat(nonrobust["qj"], num_episodes)
    nonrobust_crashes = pad_repeat(nonrobust["crashes"], num_episodes)
    nonrobust_reward = pad_repeat(nonrobust["cumulative_reward"], num_episodes)
    nonrobust_mismatch = pad_repeat(nonrobust["mismatch"], num_episodes)
    nonrobust_mismatch_runs = pad_repeat(nonrobust["mismatch_runs"], num_episodes)
    nonrobust_h_violation = pad_repeat(nonrobust["h_violation"], num_episodes)
    nonrobust_h_violation_runs = pad_repeat(nonrobust["h_violation_runs"], num_episodes)
    nonrobust_reward_runs = pad_repeat(nonrobust["cumulative_reward_runs"], num_episodes)

    robust_mismatch_q10 = np.quantile(robust["mismatch_runs"], 0.10, axis=1)
    robust_mismatch_q90 = np.quantile(robust["mismatch_runs"], 0.90, axis=1)
    naive_mismatch_q10 = np.quantile(naive["mismatch_runs"], 0.10, axis=1)
    naive_mismatch_q90 = np.quantile(naive["mismatch_runs"], 0.90, axis=1)
    nonrobust_mismatch_q10 = np.quantile(nonrobust_mismatch_runs, 0.10, axis=1)
    nonrobust_mismatch_q90 = np.quantile(nonrobust_mismatch_runs, 0.90, axis=1)

    robust_radius = prepare_radius(robust["radius"], ROBUST_INITIAL_RADIUS)
    naive_radius = prepare_radius(naive["radius"], NAIVE_INITIAL_RADIUS)

    x = base_episodes
    x_ticks = np.arange(0, num_episodes, 1)
    x_lim = (0, max(num_episodes - 1, 0))

    alpha_for_error = args.alpha if args.alpha is not None else robust["alpha"] if robust["alpha"] is not None else 0.1
    target_alpha = robust["alpha"] if robust["alpha"] is not None else 0.1

    robust_stem = Path(args.robust).stem
    output_dir = args.output_dir or os.path.join("./", "plots", f"combined_joint_{robust_stem}")
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # Radius + q_j
    fig, ax = plt.subplots()
    ax.plot(x, robust_radius, label="Robust $r_j$", color="tab:blue", marker="s")
    ax.plot(x, naive_radius, label="Naive $r_j$", color="tab:orange", marker="s")
    ax.plot(x, nonrobust_radius, label="Nonrobust $r_j$", color="tab:red", marker="s")
    ax.plot(x, robust["qj"], label="Robust $q_j$", color="tab:blue", marker="x")
    ax.plot(x, naive["qj"], label="Naive $q_j$", color="tab:orange", marker="x")
    ax.plot(x, nonrobust_qj, label="Nonrobust $q_j$", color="tab:red", marker="x")
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

    # Performance
    fig, ax = plt.subplots()
    perf_series = [
        ("Robust performance", robust["cumulative_reward"], robust["cumulative_reward_runs"], "tab:blue", "s"),
        ("Naive performance", naive["cumulative_reward"], naive["cumulative_reward_runs"], "tab:orange", "o"),
        ("Nonrobust performance", nonrobust_reward, nonrobust_reward_runs, "tab:red", "v"),
    ]
    for label, y, runs, color, marker in perf_series:
        if runs is not None:
            runs = np.asarray(runs)
            lower = np.quantile(runs, alpha_for_error, axis=1)
            upper = np.quantile(runs, 1 - alpha_for_error, axis=1)
            yerr = np.vstack([np.maximum(y - lower, 0), np.maximum(upper - y, 0)])
            ax.errorbar(x, y, yerr=yerr, label=label, color=color, marker=marker, capsize=4)
        else:
            ax.plot(x, y, label=label, color=color, marker=marker)
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

    # H-value violation rate
    fig, ax = plt.subplots()
    ax.axhline(target_alpha, linestyle=":", color="gray", label=r"Target $\alpha$")
    ax.plot(x, robust["h_violation"], label="Robust h-violation", color="tab:blue", marker="s")
    ax.plot(x, naive["h_violation"], label="Naive h-violation", color="tab:orange", marker="o")
    ax.plot(x, nonrobust_h_violation, label="Nonrobust h-violation", color="tab:red", marker="v")
    ax.set_title("Trajectory H-Violation Rate")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel("Frac. trajectories with min(h)<0")
    ax.legend()
    path = os.path.join(output_dir, "h_violation_rate.pdf")
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["h_violation"] = path

    # Crash rate
    fig, ax = plt.subplots()
    ax.plot(x, robust["crashes"], label="Robust crash rate", color="tab:blue", marker="s")
    ax.plot(x, naive["crashes"], label="Naive crash rate", color="tab:orange", marker="o")
    ax.plot(x, nonrobust_crashes, label="Nonrobust crash rate", color="tab:red", marker="v")
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

    # Mismatch
    fig, ax = plt.subplots()
    robust_mismatch_err_low = np.maximum(robust["mismatch"] - robust_mismatch_q10, 0.0)
    robust_mismatch_err_high = np.maximum(robust_mismatch_q90 - robust["mismatch"], 0.0)
    naive_mismatch_err_low = np.maximum(naive["mismatch"] - naive_mismatch_q10, 0.0)
    naive_mismatch_err_high = np.maximum(naive_mismatch_q90 - naive["mismatch"], 0.0)
    nonrobust_mismatch_err_low = np.maximum(nonrobust_mismatch - nonrobust_mismatch_q10, 0.0)
    nonrobust_mismatch_err_high = np.maximum(nonrobust_mismatch_q90 - nonrobust_mismatch, 0.0)

    ax.errorbar(
        x,
        robust["mismatch"],
        yerr=np.vstack([robust_mismatch_err_low, robust_mismatch_err_high]),
        label="Robust mismatch",
        color="tab:blue",
        marker="s",
        capsize=4,
    )
    ax.errorbar(
        x,
        naive["mismatch"],
        yerr=np.vstack([naive_mismatch_err_low, naive_mismatch_err_high]),
        label="Naive mismatch",
        color="tab:orange",
        marker="o",
        capsize=4,
    )
    ax.errorbar(
        x,
        nonrobust_mismatch,
        yerr=np.vstack([nonrobust_mismatch_err_low, nonrobust_mismatch_err_high]),
        label="Nonrobust mismatch",
        color="tab:red",
        marker="v",
        capsize=4,
    )
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

    print(f"[plot_joint_combined] Loaded robust from {os.path.abspath(args.robust)}")
    print(f"[plot_joint_combined] Loaded naive from {os.path.abspath(args.naive)}")
    print(f"[plot_joint_combined] Loaded nonrobust from {os.path.abspath(args.nonrobust)}")
    for name, path in plot_paths.items():
        print(f"[plot_joint_combined] Saved {name} plot to {path}")


if __name__ == "__main__":
    main()
