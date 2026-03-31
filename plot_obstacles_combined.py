#!/usr/bin/env python3
"""Combine obstacle conformal baselines into non-trajectory across-episode plots."""

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

ROBUST_COLOR = "tab:blue"
NAIVE_COLOR = "tab:orange"
CALIBRATE_COLOR = "tab:green"
NONROBUST_COLOR = "tab:red"

ROBUST_INITIAL_RADIUS = 2.0
NAIVE_INITIAL_RADIUS = 2.0
NONROBUST_INITIAL_RADIUS = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined obstacle conformal metrics.")
    parser.add_argument("--robust", required=True, help="Path to robust conformal_obstacles metrics .npz.")
    parser.add_argument("--naive", required=True, help="Path to naive conformal_obstacles metrics .npz.")
    parser.add_argument("--nonrobust", required=True, help="Path to nonrobust conformal_obstacles metrics .npz.")
    parser.add_argument("--output_dir", help="Directory to save plots.")
    parser.add_argument("--alpha", type=float, help="Alpha for performance error bars; defaults to robust alpha.")
    parser.add_argument("--grid", action="store_true", help="Show grid lines on plots.")
    parser.add_argument(
        "--cap_nonrobust",
        action="store_true",
        help="If set, omit the nonrobust q_j curve from the radius plot and omit the nonrobust mismatch curve from the mismatch plot.",
    )
    return parser.parse_args()


def load_run(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        if "r_mismatch_per_episode" not in data.files:
            raise KeyError(f"{path}: expected r_mismatch_per_episode.")
        radius = data["r_mismatch_per_episode"]

        run = {
            "episodes": data["episodes"],
            "radius": radius,
            "qj": data["qj_per_episode"],
            "crashes": data["crashes_per_episode"],
            "safety": data["safety_per_episode"],
            "cumulative_reward": data["cumulative_reward_per_episode"],
            "cumulative_reward_runs": data["cumulative_reward_per_run"],
            "mismatch": data["mismatch_per_episode"],
            "mismatch_runs": data["mismatch_per_run"],
            "clearance": data["min_clearance_per_episode"],
            "h_violation": data["h_violation_per_episode"],
            "h_violation_runs": data["h_violation_per_run"],
            "alpha": float(data["alpha"]) if "alpha" in data.files else None,
        }
    return run


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


def _ordered_legend(ax, entries):
    handles = []
    labels = []
    for handle, label in entries:
        if handle is not None:
            handles.append(handle)
            labels.append(label)
    if handles:
        ax.legend(handles, labels, loc="best")


def _save_pdf(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)


def compute_tube_coverage(mismatch_runs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    mismatch_runs = np.asarray(mismatch_runs)
    thresholds = np.asarray(thresholds)
    if mismatch_runs.ndim != 2:
        raise ValueError(f"Expected mismatch_runs to have shape (num_episodes, num_runs), got {mismatch_runs.shape}.")
    if thresholds.ndim != 1:
        raise ValueError(f"Expected thresholds to have shape (num_episodes,), got {thresholds.shape}.")
    if mismatch_runs.shape[0] != thresholds.shape[0]:
        raise ValueError(
            f"Mismatch run episode count {mismatch_runs.shape[0]} does not match threshold count {thresholds.shape[0]}."
        )
    return np.mean(mismatch_runs <= thresholds[:, None], axis=1)


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

    # Robust/naive must exactly match base horizon.
    ensure_len("robust radius", robust["radius"], num_episodes)
    ensure_len("robust qj", robust["qj"], num_episodes)
    ensure_len("robust crashes", robust["crashes"], num_episodes)
    ensure_len("robust safety", robust["safety"], num_episodes)
    ensure_len("robust cumulative_reward", robust["cumulative_reward"], num_episodes)
    ensure_len("robust mismatch", robust["mismatch"], num_episodes)
    ensure_len("robust mismatch_runs", robust["mismatch_runs"], num_episodes)
    ensure_len("robust h_violation", robust["h_violation"], num_episodes)
    ensure_len("robust h_violation_runs", robust["h_violation_runs"], num_episodes)
    ensure_len("robust clearance", robust["clearance"], num_episodes)
    ensure_len("robust cumulative_reward_per_run", robust["cumulative_reward_runs"], num_episodes)

    ensure_len("naive radius", naive["radius"], num_episodes)
    ensure_len("naive qj", naive["qj"], num_episodes)
    ensure_len("naive crashes", naive["crashes"], num_episodes)
    ensure_len("naive safety", naive["safety"], num_episodes)
    ensure_len("naive cumulative_reward", naive["cumulative_reward"], num_episodes)
    ensure_len("naive mismatch", naive["mismatch"], num_episodes)
    ensure_len("naive mismatch_runs", naive["mismatch_runs"], num_episodes)
    ensure_len("naive h_violation", naive["h_violation"], num_episodes)
    ensure_len("naive h_violation_runs", naive["h_violation_runs"], num_episodes)
    ensure_len("naive clearance", naive["clearance"], num_episodes)
    ensure_len("naive cumulative_reward_per_run", naive["cumulative_reward_runs"], num_episodes)

    # Nonrobust is single-episode and gets padded across robust horizon.
    nonrobust_radius = pad_repeat(prepare_radius(nonrobust["radius"], NONROBUST_INITIAL_RADIUS), num_episodes)
    nonrobust_qj = pad_repeat(nonrobust["qj"], num_episodes)
    nonrobust_crashes = pad_repeat(nonrobust["crashes"], num_episodes)
    nonrobust_reward = pad_repeat(nonrobust["cumulative_reward"], num_episodes)
    nonrobust_mismatch = pad_repeat(nonrobust["mismatch"], num_episodes)
    nonrobust_mismatch_runs = pad_repeat(nonrobust["mismatch_runs"], num_episodes)
    nonrobust_clearance = pad_repeat(nonrobust["clearance"], num_episodes)
    nonrobust_h_violation = pad_repeat(nonrobust["h_violation"], num_episodes)
    nonrobust_h_violation_runs = pad_repeat(nonrobust["h_violation_runs"], num_episodes)
    nonrobust_reward_runs = pad_repeat(nonrobust["cumulative_reward_runs"], num_episodes)
    robust_coverage = compute_tube_coverage(robust["mismatch_runs"], robust["radius"])
    naive_coverage = compute_tube_coverage(naive["mismatch_runs"], naive["radius"])
    nonrobust_coverage = pad_repeat(compute_tube_coverage(nonrobust["mismatch_runs"], nonrobust["radius"]), num_episodes)

    robust_radius = prepare_radius(robust["radius"], ROBUST_INITIAL_RADIUS)
    naive_radius = prepare_radius(naive["radius"], NAIVE_INITIAL_RADIUS)
    cal_once_radius = np.empty_like(naive_radius)
    cal_once_radius[0] = naive_radius[0]
    cal_once_qj = np.empty_like(naive["qj"])
    cal_once_qj[0] = naive["qj"][0]
    if num_episodes > 1:
        cal_once_radius[1:] = np.repeat(naive_radius[1], num_episodes - 1)
        cal_once_qj[1:] = np.repeat(naive["qj"][1], num_episodes - 1)

    robust_mismatch_q10 = np.quantile(robust["mismatch_runs"], 0.10, axis=1)
    robust_mismatch_q90 = np.quantile(robust["mismatch_runs"], 0.90, axis=1)
    naive_mismatch_q10 = np.quantile(naive["mismatch_runs"], 0.10, axis=1)
    naive_mismatch_q90 = np.quantile(naive["mismatch_runs"], 0.90, axis=1)
    nonrobust_mismatch_q10 = np.quantile(nonrobust_mismatch_runs, 0.10, axis=1)
    nonrobust_mismatch_q90 = np.quantile(nonrobust_mismatch_runs, 0.90, axis=1)

    x_radius = base_episodes
    x_other = base_episodes[1:]
    cal_once_performance = np.repeat(naive["cumulative_reward"][0], num_episodes)
    cal_once_performance_runs = (
        np.repeat(naive["cumulative_reward_runs"][:1], num_episodes, axis=0)
        if naive["cumulative_reward_runs"] is not None
        else None
    )
    cal_once_h_violation = np.repeat(naive["h_violation"][0], x_other.shape[0]) if x_other.shape[0] > 0 else np.array([])
    cal_once_crashes = np.repeat(naive["crashes"][0], x_other.shape[0]) if x_other.shape[0] > 0 else np.array([])
    cal_once_clearance = np.repeat(naive["clearance"][0], x_other.shape[0]) if x_other.shape[0] > 0 else np.array([])
    cal_once_mismatch = np.repeat(naive["mismatch"][0], x_other.shape[0]) if x_other.shape[0] > 0 else np.array([])
    cal_once_mismatch_err_low = np.repeat(np.maximum(naive["mismatch"][0] - naive_mismatch_q10[0], 0.0), x_other.shape[0]) if x_other.shape[0] > 0 else np.array([])
    cal_once_mismatch_err_high = np.repeat(np.maximum(naive_mismatch_q90[0] - naive["mismatch"][0], 0.0), x_other.shape[0]) if x_other.shape[0] > 0 else np.array([])
    cal_once_coverage_value = float(np.mean(np.asarray(naive["mismatch_runs"])[0] <= float(np.asarray(naive["radius"])[0])))
    cal_once_coverage = np.repeat(cal_once_coverage_value, x_other.shape[0]) if x_other.shape[0] > 0 else np.array([])
    x_radius_ticks = np.arange(0, num_episodes, 1)
    x_other_ticks = np.arange(1, num_episodes, 1)
    x_radius_lim = (0, max(num_episodes - 1, 0))
    x_other_lim = (1, max(num_episodes - 1, 1))

    alpha_for_error = args.alpha if args.alpha is not None else robust["alpha"] if robust["alpha"] is not None else 0.1
    target_alpha = robust["alpha"] if robust["alpha"] is not None else 0.1

    robust_stem = Path(args.robust).stem
    output_dir = args.output_dir or os.path.join("./", "plots", f"combined_obstacles_{robust_stem}")
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # Radius + q_j
    fig, ax = plt.subplots()
    cal_r_handle = ax.plot(x_radius, cal_once_radius, color=CALIBRATE_COLOR, marker="s", zorder=1)[0]
    nonrobust_r_handle = ax.plot(x_radius, nonrobust_radius, color=NONROBUST_COLOR, marker="s", zorder=2)[0]
    naive_r_handle = ax.plot(x_radius, naive_radius, color=NAIVE_COLOR, marker="s", zorder=3)[0]
    robust_r_handle = ax.plot(x_radius, robust_radius, color=ROBUST_COLOR, marker="s", zorder=4)[0]
    robust_q_handle = ax.plot(x_radius, robust["qj"], color=ROBUST_COLOR, marker="x", zorder=4)[0]
    ax.plot(x_radius, naive["qj"], color=NAIVE_COLOR, marker="x", zorder=3)
    ax.plot(x_radius, cal_once_qj, color=CALIBRATE_COLOR, marker="x", zorder=1)
    if not args.cap_nonrobust:
        ax.plot(x_radius, nonrobust_qj, color=NONROBUST_COLOR, marker="x", zorder=2)
    ax.set_title(r"$r_j$ and $q_j$ across episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_radius_lim)
    ax.set_xticks(x_radius_ticks)
    ax.set_ylabel("")
    if args.grid:
        ax.grid(True, alpha=0.3)
    _ordered_legend(
        ax,
        [
            (robust_r_handle, r"Robust $r_j$"),
            (naive_r_handle, r"Naive $r_j$"),
            (cal_r_handle, r"Calibrate-once $r_j$"),
            (nonrobust_r_handle, r"Nonrobust $r_j$"),
            (robust_q_handle, r"Robust $q_j$"),
        ],
    )
    path = os.path.join(output_dir, "rj_qj_across_episodes.pdf")
    _save_pdf(fig, path)
    plot_paths["radius"] = path

    # Performance
    fig, ax = plt.subplots()
    perf_series = [
        ("Calibrate-once", cal_once_performance, cal_once_performance_runs, CALIBRATE_COLOR, "^"),
        ("Nonrobust", nonrobust_reward, nonrobust_reward_runs, NONROBUST_COLOR, "v"),
        ("Naive", naive["cumulative_reward"], naive["cumulative_reward_runs"], NAIVE_COLOR, "o"),
        ("Robust", robust["cumulative_reward"], robust["cumulative_reward_runs"], ROBUST_COLOR, "s"),
    ]
    perf_handles = {}
    for label, y, runs, color, marker in perf_series:
        if runs is not None:
            runs = np.asarray(runs)
            lower = np.quantile(runs, alpha_for_error, axis=1)[:-1]
            upper = np.quantile(runs, 1 - alpha_for_error, axis=1)[:-1]
            y_plot = y[:-1]
            lower_plot = lower
            upper_plot = upper
            yerr = np.vstack([np.maximum(y_plot - lower_plot, 0), np.maximum(upper_plot - y_plot, 0)])
            handle = ax.errorbar(x_other, y_plot, yerr=yerr, color=color, marker=marker, capsize=4)
            perf_handles[label] = handle.lines[0]
        else:
            y_plot = y[:-1]
            perf_handles[label] = ax.plot(x_other, y_plot, color=color, marker=marker)[0]
    ax.set_title("Cumulative reward across episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel("")
    if args.grid:
        ax.grid(True, alpha=0.3)
    _ordered_legend(
        ax,
        [
            (perf_handles.get("Robust"), "Robust"),
            (perf_handles.get("Naive"), "Naive"),
            (perf_handles.get("Calibrate-once"), "Calibrate-once"),
            (perf_handles.get("Nonrobust"), "Nonrobust"),
        ],
    )
    path = os.path.join(output_dir, "performance_cumulative_reward.pdf")
    _save_pdf(fig, path)
    plot_paths["performance"] = path

    # Safety coverage rate
    fig, ax = plt.subplots()
    reference_handle = ax.axhline(1 - target_alpha, linestyle=":", color="gray")
    cal_handle = ax.plot(x_other, 1 - cal_once_h_violation, color=CALIBRATE_COLOR, marker="^", zorder=1)[0]
    nonrobust_handle = ax.plot(x_other, 1 - nonrobust_h_violation[:-1], color=NONROBUST_COLOR, marker="v", zorder=2)[0]
    naive_handle = ax.plot(x_other, 1 - naive["h_violation"][:-1], color=NAIVE_COLOR, marker="o", zorder=3)[0]
    robust_handle = ax.plot(x_other, 1 - robust["h_violation"][:-1], color=ROBUST_COLOR, marker="s", zorder=4)[0]
    ax.set_title(r"Empirical safety coverage across episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel("")
    ax.set_ylim(0.0, 1.0)
    if args.grid:
        ax.grid(True, alpha=0.3)
    _ordered_legend(
        ax,
        [
            (robust_handle, "Robust"),
            (naive_handle, "Naive"),
            (cal_handle, "Calibrate-once"),
            (nonrobust_handle, "Nonrobust"),
            (reference_handle, r"$1 - \alpha$"),
        ],
    )
    path = os.path.join(output_dir, "safety_coverage_across_episodes.pdf")
    _save_pdf(fig, path)
    plot_paths["safety_coverage"] = path

    # Crash rate
    fig, ax = plt.subplots()
    cal_handle = ax.plot(x_other, cal_once_crashes, color=CALIBRATE_COLOR, marker="^", zorder=1)[0]
    nonrobust_handle = ax.plot(x_other, nonrobust_crashes[:-1], color=NONROBUST_COLOR, marker="v", zorder=2)[0]
    naive_handle = ax.plot(x_other, naive["crashes"][:-1], color=NAIVE_COLOR, marker="o", zorder=3)[0]
    robust_handle = ax.plot(x_other, robust["crashes"][:-1], color=ROBUST_COLOR, marker="s", zorder=4)[0]
    ax.set_title("Crash rate across episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel("")
    if args.grid:
        ax.grid(True, alpha=0.3)
    _ordered_legend(
        ax,
        [
            (robust_handle, "Robust"),
            (naive_handle, "Naive"),
            (cal_handle, "Calibrate-once"),
            (nonrobust_handle, "Nonrobust"),
        ],
    )
    path = os.path.join(output_dir, "crash_rate.pdf")
    _save_pdf(fig, path)
    plot_paths["crash_rate"] = path

    # Mismatch
    fig, ax = plt.subplots()
    robust_mismatch_err_low = np.maximum(robust["mismatch"] - robust_mismatch_q10, 0.0)
    robust_mismatch_err_high = np.maximum(robust_mismatch_q90 - robust["mismatch"], 0.0)
    naive_mismatch_err_low = np.maximum(naive["mismatch"] - naive_mismatch_q10, 0.0)
    naive_mismatch_err_high = np.maximum(naive_mismatch_q90 - naive["mismatch"], 0.0)
    nonrobust_mismatch_err_low = np.maximum(nonrobust_mismatch - nonrobust_mismatch_q10, 0.0)
    nonrobust_mismatch_err_high = np.maximum(nonrobust_mismatch_q90 - nonrobust_mismatch, 0.0)

    cal_handle = ax.errorbar(
        x_other,
        cal_once_mismatch,
        yerr=np.vstack([cal_once_mismatch_err_low, cal_once_mismatch_err_high]),
        color=CALIBRATE_COLOR,
        marker="^",
        capsize=4,
        zorder=1,
    )
    nonrobust_handle = None
    if not args.cap_nonrobust:
        nonrobust_handle = ax.errorbar(
            x_other,
            nonrobust_mismatch[:-1],
            yerr=np.vstack([nonrobust_mismatch_err_low[:-1], nonrobust_mismatch_err_high[:-1]]),
            color=NONROBUST_COLOR,
            marker="v",
            capsize=4,
            zorder=2,
        )
    naive_handle = ax.errorbar(
        x_other,
        naive["mismatch"][:-1],
        yerr=np.vstack([naive_mismatch_err_low[:-1], naive_mismatch_err_high[:-1]]),
        color=NAIVE_COLOR,
        marker="o",
        capsize=4,
        zorder=3,
    )
    robust_handle = ax.errorbar(
        x_other,
        robust["mismatch"][:-1],
        yerr=np.vstack([robust_mismatch_err_low[:-1], robust_mismatch_err_high[:-1]]),
        color=ROBUST_COLOR,
        marker="s",
        capsize=4,
        zorder=4,
    )
    ax.set_title(r"$\max_t \|(\hat x,\hat v)_{t+1} - (x,v)_{t+1}\| / \Delta t$")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel("")
    if args.grid:
        ax.grid(True, alpha=0.3)
    _ordered_legend(
        ax,
        [
            (robust_handle.lines[0], "Robust"),
            (naive_handle.lines[0], "Naive"),
            (cal_handle.lines[0], "Calibrate-once"),
            (None if nonrobust_handle is None else nonrobust_handle.lines[0], "Nonrobust"),
        ],
    )
    path = os.path.join(output_dir, "state_prediction_error_across_episodes.pdf")
    _save_pdf(fig, path)
    plot_paths["mismatch"] = path

    # Score coverage
    fig, ax = plt.subplots()
    reference_handle = ax.axhline(1.0 - target_alpha, linestyle=":", color="gray")
    cal_handle = ax.plot(x_other, cal_once_coverage, color=CALIBRATE_COLOR, marker="^", zorder=1)[0]
    nonrobust_handle = ax.plot(x_other, nonrobust_coverage[:-1], color=NONROBUST_COLOR, marker="v", zorder=2)[0]
    naive_handle = ax.plot(x_other, naive_coverage[:-1], color=NAIVE_COLOR, marker="o", zorder=3)[0]
    robust_handle = ax.plot(x_other, robust_coverage[:-1], color=ROBUST_COLOR, marker="s", zorder=4)[0]
    ax.set_title("Empirical score coverage across episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel("")
    ax.set_ylim(0.0, 1.0)
    if args.grid:
        ax.grid(True, alpha=0.3)
    _ordered_legend(
        ax,
        [
            (robust_handle, "Robust"),
            (naive_handle, "Naive"),
            (cal_handle, "Calibrate-once"),
            (nonrobust_handle, "Nonrobust"),
            (reference_handle, r"$1-\alpha$"),
        ],
    )
    path = os.path.join(output_dir, "score_coverage_across_episodes.pdf")
    _save_pdf(fig, path)
    plot_paths["score_coverage"] = path

    # Clearance
    fig, ax = plt.subplots()
    reference_handle = ax.axhline(0.0, linestyle="--", color="gray")
    cal_handle = ax.plot(x_other, cal_once_clearance, color=CALIBRATE_COLOR, marker="^", zorder=1)[0]
    nonrobust_handle = ax.plot(x_other, nonrobust_clearance[:-1], color=NONROBUST_COLOR, marker="v", zorder=2)[0]
    naive_handle = ax.plot(x_other, naive["clearance"][:-1], color=NAIVE_COLOR, marker="o", zorder=3)[0]
    robust_handle = ax.plot(x_other, robust["clearance"][:-1], color=ROBUST_COLOR, marker="s", zorder=4)[0]
    ax.set_title("Clearance across episodes")
    ax.set_xlabel("Episode (j)")
    ax.set_xlim(*x_other_lim)
    ax.set_xticks(x_other_ticks)
    ax.set_ylabel("")
    if args.grid:
        ax.grid(True, alpha=0.3)
    _ordered_legend(
        ax,
        [
            (robust_handle, "Robust"),
            (naive_handle, "Naive"),
            (cal_handle, "Calibrate-once"),
            (nonrobust_handle, "Nonrobust"),
            (reference_handle, "Collision boundary"),
        ],
    )
    path = os.path.join(output_dir, "clearance_across_episodes.pdf")
    _save_pdf(fig, path)
    plot_paths["clearance"] = path

    print(f"[plot_obstacles_combined] Loaded robust from {os.path.abspath(args.robust)}")
    print(f"[plot_obstacles_combined] Loaded naive from {os.path.abspath(args.naive)}")
    print(f"[plot_obstacles_combined] Loaded nonrobust from {os.path.abspath(args.nonrobust)}")
    for name, path in plot_paths.items():
        print(f"[plot_obstacles_combined] Saved {name} plot to {path}")


if __name__ == "__main__":
    main()
