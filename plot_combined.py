#!/usr/bin/env python3
"""
Combine conformal and non-interaction runs into unified across-episode plots.

Generates four plots (radius, performance, empirical tube coverage, empirical
safety coverage) with multiple lines:
  - Robust conformal run (conformal.py)
  - Naive conformal run  (conformal.py)
  - Calibrate-once run   (conformal_no_interaction.py), mapped so its second
    recorded episode is treated as episode 1 and repeated for all later
    episodes
  - Non-interactive run (single-episode .npz); provides constant lines on all
    four plots.

Trajectory/tube visualizations are intentionally omitted.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Use LaTeX if available for consistency with prior plots.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


MAX_RADIUS = 8.0  # Used to seed r0 for conformal runs (matches plot_conformal.py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined conformal/non-interaction metrics.")
    parser.add_argument("--robust", required=True, help="Path to Robust conformal .npz (conformal.py run).")
    parser.add_argument("--naive", required=True, help="Path to Naive conformal .npz (conformal.py run).")
    parser.add_argument("--calibrate_once", required=True, help="Path to calibrate-once .npz (conformal_no_interaction.py run).")
    parser.add_argument("--non_interactive", required=True, help="Path to single-episode non-interactive .npz.")
    parser.add_argument("--output_dir", help="Directory to save plots (default: ./plots/combined_<robust_stem>).")
    parser.add_argument("--alpha", type=float, help="Alpha for performance error bars; defaults to Robust run's alpha.")
    return parser.parse_args()


def load_run(path: str) -> dict:
    """Load a conformal metrics file into a simple dict."""
    with np.load(path) as data:
        return {
            "episodes": data["episodes"],
            "radius": data["radius_per_episode"],
            "qj": data["qj_per_episode"],
            "tube": data["tube_coverage_per_episode"],
            "safety": data["safety_per_episode"],
            "cumulative_reward": data["cumulative_reward_per_episode"],
            "cumulative_reward_runs": data.get("cumulative_reward_per_run", None),
            "alpha": float(data["alpha"]) if "alpha" in data else None,
        }


def prepare_conformal_radius(radius: np.ndarray) -> np.ndarray:
    """Insert MAX_RADIUS at episode 0 to mimic plot_conformal alignment."""
    return np.insert(radius, 0, MAX_RADIUS)[:-1]


def pad_repeat(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Copy arr to length target_len, repeating the last element as needed."""
    arr = np.asarray(arr)
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    pad = np.repeat(arr[-1:], target_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def main() -> None:
    args = parse_args()

    robust = load_run(args.robust)
    naive = load_run(args.naive)
    calibrate_once = load_run(args.calibrate_once)
    non_int_single_run = load_run(args.non_interactive)

    # Base episode grid comes from the Robust conformal run.
    base_episodes = robust["episodes"]
    num_episodes = len(base_episodes) + 1  # as used in plot_conformal.py

    # Sanity checks to avoid silent misalignment.
    if len(naive["episodes"]) != len(base_episodes):
        raise ValueError("Naive run episode count does not match Robust run.")

    # Calibrate-once: pad/repeat to required lengths.
    cal_once_radius = pad_repeat(calibrate_once["radius"], num_episodes - 1)
    cal_once_qj = pad_repeat(calibrate_once["qj"], num_episodes - 1)
    cal_once_tube = pad_repeat(calibrate_once["tube"], num_episodes)
    cal_once_safety = pad_repeat(calibrate_once["safety"], num_episodes)
    cal_once_perf = pad_repeat(calibrate_once["cumulative_reward"], num_episodes)
    cal_once_runs = pad_repeat(calibrate_once["cumulative_reward_runs"], num_episodes) if calibrate_once["cumulative_reward_runs"] is not None else None

    # Non-interactive: pad/repeat to required lengths.
    single_radius = pad_repeat(non_int_single_run["radius"], num_episodes - 1)
    single_qj = pad_repeat(non_int_single_run["qj"], num_episodes - 1)
    single_tube = pad_repeat(non_int_single_run["tube"], num_episodes)
    single_safety = pad_repeat(non_int_single_run["safety"], num_episodes)
    single_perf = pad_repeat(non_int_single_run["cumulative_reward"], num_episodes)
    single_runs = pad_repeat(non_int_single_run["cumulative_reward_runs"], num_episodes) if non_int_single_run["cumulative_reward_runs"] is not None else None
    non_int_single = float(single_perf[0])

    # Prepare x-axes: conformal runs start at episode 1; calibrate/non-interactive start at 0.
    x_radius = base_episodes  # 0..(N-1)
    x_conformal = np.arange(1, num_episodes)          # 1..N-1
    x_calib = np.arange(0, num_episodes)              # 0..N-1
    x_lim_radius = (0, num_episodes - 2)
    x_lim_perf = (0, num_episodes - 1)

    colors = {
        "robust": "tab:blue",
        "naive": "tab:orange",
        "cal_once": "tab:green",
        "non_int": "tab:red",
    }

    # Prepare radius and q_j series.
    series_radius = [
        {"label": "Robust $r_j$", "y": prepare_conformal_radius(robust["radius"]), "style": {"color": colors["robust"], "marker": "s"}},
        {"label": "Naive $r_j$", "y": prepare_conformal_radius(naive["radius"]), "style": {"color": colors["naive"], "marker": "s"}},
        {"label": "Calibrate-once $r_j$", "y": cal_once_radius, "style": {"color": colors["cal_once"], "marker": "s"}},
        {"label": "Non-interactive $r_j$", "y": single_radius, "style": {"color": colors["non_int"], "marker": "s"}},
    ]
    series_q = [
        {"label": "Robust $q_j$", "y": robust["qj"], "style": {"color": colors["robust"], "marker": "x"}},
        {"label": "Naive $q_j$", "y": naive["qj"], "style": {"color": colors["naive"], "marker": "x"}},
        {"label": "Calibrate-once $q_j$", "y": cal_once_qj, "style": {"color": colors["cal_once"], "marker": "x"}},
        {"label": "Non-interactive $q_j$", "y": single_qj, "style": {"color": colors["non_int"], "marker": "x"}},
    ]

    # Performance series; include optional error bars if available.
    alpha_for_error = args.alpha if args.alpha is not None else robust["alpha"] if robust["alpha"] is not None else 0.1
    perf_series = [
        {
            "label": "Robust performance",
            "y": robust["cumulative_reward"],
            "runs": robust["cumulative_reward_runs"],
            "x": x_conformal,
            "style": {"color": "tab:blue", "marker": "s"},
        },
        {
            "label": "Naive performance",
            "y": naive["cumulative_reward"],
            "runs": naive["cumulative_reward_runs"],
            "x": x_conformal,
            "style": {"color": "tab:orange", "marker": "o"},
        },
        {
            "label": "Calibrate-once performance",
            "y": cal_once_perf,
            "runs": cal_once_runs,
            "x": x_calib,
            "style": {"color": "tab:green", "marker": "^"},
        },
        {
            "label": "Non-interactive performance",
            "y": single_perf,
            "runs": single_runs,
            "x": x_calib,
            "style": {"color": "tab:red", "marker": "v"},
        },
    ]

    # Tube and safety series.
    tube_series = [
        {"label": "Robust tube", "y": robust["tube"], "x": x_conformal, "style": {"color": "tab:blue", "marker": "s"}},
        {"label": "Naive tube", "y": naive["tube"], "x": x_conformal, "style": {"color": "tab:orange", "marker": "o"}},
        {"label": "Calibrate-once tube", "y": cal_once_tube, "x": x_calib, "style": {"color": "tab:green", "marker": "^"}},
        {"label": "Non-interactive tube", "y": single_tube, "x": x_calib, "style": {"color": "tab:red", "marker": "v"}},
    ]
    safety_series = [
        {"label": "Robust safety", "y": robust["safety"], "x": x_conformal, "style": {"color": "tab:blue", "marker": "s"}},
        {"label": "Naive safety", "y": naive["safety"], "x": x_conformal, "style": {"color": "tab:orange", "marker": "o"}},
        {"label": "Calibrate-once safety", "y": cal_once_safety, "x": x_calib, "style": {"color": "tab:green", "marker": "^"}},
        {"label": "Non-interactive safety", "y": single_safety, "x": x_calib, "style": {"color": "tab:red", "marker": "v"}},
    ]

    # Output directory.
    robust_stem = Path(args.robust).stem
    output_dir = args.output_dir or os.path.join("./", "plots", f"combined_{robust_stem}")
    os.makedirs(output_dir, exist_ok=True)

    plot_paths = {}

    # Plot A: Radius across episodes
    fig, ax = plt.subplots()
    for s in series_radius:
        ax.plot(x_radius, s["y"], label=s["label"], **s["style"])
    for s in series_q:
        ax.plot(x_radius, s["y"], label=s["label"], **s["style"])
    ax.set_title(r"Radius Across Episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_lim_radius)
    ax.set_xticks(x_radius)
    ax.set_ylabel(r"Radius ($m$)")
    ax.legend()
    radius_path = os.path.join(output_dir, "radius_across_episodes.pdf")
    fig.savefig(radius_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["radius"] = radius_path

    # Plot B: Performance across episodes
    fig, ax = plt.subplots()
    for s in perf_series:
        y = s["y"]
        runs = np.asarray(s["runs"]) if s["runs"] is not None else None
        if runs is not None:
            lower = np.quantile(runs, alpha_for_error, axis=1)
            upper = np.quantile(runs, 1 - alpha_for_error, axis=1)
            yerr = np.vstack([
                np.maximum(y - lower, 0),
                np.maximum(upper - y, 0),
            ])
            ax.errorbar(s["x"], y, yerr=yerr, label=s["label"], capsize=4, **s["style"])
        else:
            ax.plot(s["x"], y, label=s["label"], **s["style"])
    ax.set_title(r"Performance Across Episodes")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_lim_perf)
    ax.set_xticks(np.arange(0, num_episodes, 1))
    ax.set_ylabel(r"Cumulative reward ($m$)")
    ax.legend(loc="center right")
    perf_path = os.path.join(output_dir, "performance_cumulative_reward.pdf")
    fig.savefig(perf_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["performance"] = perf_path

    # Plot C: Empirical tube coverage
    fig, ax = plt.subplots()
    target_line = 1 - (robust["alpha"] if robust["alpha"] is not None else 0.1)
    ax.axhline(target_line, linestyle="--", color="gray", label=r"Target $(1 - \alpha)$")
    for s in tube_series:
        ax.plot(s["x"], s["y"], label=s["label"], **s["style"])
    ax.set_title(r"Empirical Tube Coverage")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_lim_perf)
    ax.set_xticks(np.arange(0, num_episodes, 1))
    ax.set_ylabel(r"Coverage (\%)")
    ax.legend()
    tube_path = os.path.join(output_dir, "tube_coverage.pdf")
    fig.savefig(tube_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["tube_coverage"] = tube_path

    # Plot D: Empirical safety coverage
    fig, ax = plt.subplots()
    ax.axhline(target_line, linestyle="--", color="gray", label=r"Target $(1 - \alpha)$")
    for s in safety_series:
        ax.plot(s["x"], s["y"], label=s["label"], **s["style"])
    ax.set_title(r"Empirical Safety Coverage")
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_xlim(*x_lim_perf)
    ax.set_xticks(np.arange(0, num_episodes, 1))
    ax.set_ylabel(r"Coverage (\%)")
    ax.legend()
    safety_path = os.path.join(output_dir, "empirical_safety_coverage.pdf")
    fig.savefig(safety_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    plot_paths["safety"] = safety_path

    # Log saved paths for quick reference.
    print(f"[plot_combined] Loaded Robust from {os.path.abspath(args.robust)}")
    print(f"[plot_combined] Loaded Naive from {os.path.abspath(args.naive)}")
    print(f"[plot_combined] Loaded Calibrate-once from {os.path.abspath(args.calibrate_once)}")
    print(f"[plot_combined] Loaded Non-interactive performance from {os.path.abspath(args.non_interactive)}")
    for name, path in plot_paths.items():
        print(f"[plot_combined] Saved {name} plot to {path}")


if __name__ == "__main__":
    main()
