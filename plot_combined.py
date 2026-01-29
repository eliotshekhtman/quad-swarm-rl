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

    # Map calibrate-once arrays: first value -> episode 0, second -> episode 1..end.
    def map_calibrate_once(arr: np.ndarray) -> np.ndarray:
        return np.concatenate(([arr[0]], np.full(num_episodes - 2, arr[1])))

    cal_once_radius = map_calibrate_once(calibrate_once["radius"])
    cal_once_qj = map_calibrate_once(calibrate_once["qj"])
    cal_once_tube = map_calibrate_once(calibrate_once["tube"])
    cal_once_safety = map_calibrate_once(calibrate_once["safety"])
    cal_once_perf = map_calibrate_once(calibrate_once["cumulative_reward"])

    # Broadcast single-episode non-interactive data across episodes.
    def broadcast(arr: np.ndarray) -> np.ndarray:
        return np.full(num_episodes - 1, float(arr[0]))

    single_radius = broadcast(non_int_single_run["radius"])
    single_qj = broadcast(non_int_single_run["qj"])
    single_tube = broadcast(non_int_single_run["tube"])
    single_safety = broadcast(non_int_single_run["safety"])
    single_perf = broadcast(non_int_single_run["cumulative_reward"])
    non_int_single = float(single_perf[0])

    # Prepare x-axes matching plot_conformal conventions.
    x_radius = base_episodes  # 0..(N-1)
    x_perf = np.arange(1, num_episodes)  # 1..N
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
        {"label": "Robust $r_j$", "y": prepare_conformal_radius(robust["radius"]), "style": {"color": colors["robust"], "marker": "P"}},
        {"label": "Naive $r_j$", "y": prepare_conformal_radius(naive["radius"]), "style": {"color": colors["naive"], "marker": "P"}},
        {"label": "Calibrate-once $r_j$", "y": cal_once_radius, "style": {"color": colors["cal_once"], "marker": "P"}},
        {"label": "Non-interactive $r_j$", "y": single_radius, "style": {"color": colors["non_int"], "marker": "P"}},
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
            "style": {"color": "tab:blue", "marker": "s"},
        },
        {
            "label": "Naive performance",
            "y": naive["cumulative_reward"],
            "runs": naive["cumulative_reward_runs"],
            "style": {"color": "tab:orange", "marker": "o"},
        },
        {
            "label": "Calibrate-once performance",
            "y": cal_once_perf,
            "runs": None,
            "style": {"color": "tab:green", "marker": "^"},
        },
        {
            "label": "Non-interactive performance",
            "y": single_perf,
            "runs": None,
            "style": {"color": "tab:red", "marker": "v"},
        },
    ]

    # Tube and safety series.
    tube_series = [
        {"label": "Robust tube", "y": robust["tube"], "style": {"color": "tab:blue", "marker": "s"}},
        {"label": "Naive tube", "y": naive["tube"], "style": {"color": "tab:orange", "marker": "o"}},
        {"label": "Calibrate-once tube", "y": cal_once_tube, "style": {"color": "tab:green", "marker": "^"}},
        {"label": "Non-interactive tube", "y": single_tube, "style": {"color": "tab:red", "marker": "v"}},
    ]
    safety_series = [
        {"label": "Robust safety", "y": robust["safety"], "style": {"color": "tab:blue", "marker": "s"}},
        {"label": "Naive safety", "y": naive["safety"], "style": {"color": "tab:orange", "marker": "o"}},
        {"label": "Calibrate-once safety", "y": cal_once_safety, "style": {"color": "tab:green", "marker": "^"}},
        {"label": "Non-interactive safety", "y": single_safety, "style": {"color": "tab:red", "marker": "v"}},
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
            ax.errorbar(x_perf, y, yerr=yerr, label=s["label"], capsize=4, **s["style"])
        else:
            ax.plot(x_perf, y, label=s["label"], **s["style"])
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
        ax.plot(x_perf, s["y"], label=s["label"], **s["style"])
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
        ax.plot(x_perf, s["y"], label=s["label"], **s["style"])
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
