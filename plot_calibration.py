#!/usr/bin/env python3
"""Plot calibration budgets and within-episode quantile confidence bounds."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run_args(path):
    args_path = Path(path).with_name("conformal_args.json")
    return json.loads(args_path.read_text()) if args_path.exists() else {}


def load_initial_radius(path, data):
    if "anytime_current_radius_per_episode" in data:
        return float(data["anytime_current_radius_per_episode"][0])
    # Runs predating r_init used a fixed initial radius of 8.
    return float(load_run_args(path).get("r_init", 8.0))


def load_noninteractive_radius(path):
    with np.load(path) as data:
        if str(data["update_procedure"]) != "non-interactive":
            raise ValueError("The reference must be a non-interactive run.")
        return float(data["radius_per_episode"][-1])


def calibration_budget(path):
    with np.load(path) as data:
        episodes = data["episodes"].copy()
        procedure = str(data["update_procedure"])
        if procedure in ("anytime", "anytime-nonparametric"):
            counts = data["anytime_trajectory_count_per_episode"].copy()
            initial = int(data["anytime_initial_trajectory_count"])
        else:
            args = load_run_args(path)
            if "num_trajectories" not in args:
                raise ValueError(f"Missing num_trajectories in metadata for {path}")
            counts = np.full(len(episodes), int(args["num_trajectories"]), dtype=int)
            if procedure in ("non-interactive", "calibrate-once", "calibrate-once-paper"):
                # Later batches assess the policy; only q0 determines its updates.
                counts[1:] = 0
            initial = 0
        if counts.shape != episodes.shape or np.any(counts < 0):
            raise ValueError(f"Invalid calibration counts for {path}")
        return episodes, np.cumsum(counts), initial


def _save_figure(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_cumulative_trajectories(runs, output_dir):
    """Plot update calibration, separately displaying initialization overhead."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for label, path in runs.items():
        episodes, cumulative, initial = calibration_budget(path)
        line, = ax.plot(
            episodes, np.where(cumulative > 0, cumulative, np.nan),
            marker="o", label=f"{label}: update calibration",
        )
        if initial:
            ax.plot(
                episodes, cumulative + initial, linestyle="--", marker="s",
                color=line.get_color(),
                label=f"{label}: including {initial:,} initialization",
            )
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel(r"Episode ($j$)")
    ax.set_ylabel("Cumulative calibration trajectories")
    ax.set_title("Calibration sample budget")
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.legend(fontsize=9)
    return _save_figure(fig, output_dir, "cumulative_calibration_trajectories")


def quantile_bound_panels(path, episodes=None):
    with np.load(path) as data:
        ids = data["episodes"]
        reference_quantiles = data.get("anytime_reference_quantile_per_episode")
        if reference_quantiles is not None and reference_quantiles.shape != ids.shape:
            raise ValueError(f"Independent quantile references must match episodes in {path}")
        selected = (
            list(dict.fromkeys([int(ids[0]), int(ids[len(ids) // 2]), int(ids[-1])]))
            if episodes is None else episodes
        )
        panels = []
        for episode in selected:
            matches = np.flatnonzero(ids == episode)
            if len(matches) != 1:
                raise ValueError(f"Episode {episode} not found in {path}")
            index = int(matches[0])
            count = int(data["anytime_trajectory_count_per_episode"][index])
            panels.append({
                "episode": episode,
                "count": count,
                "radius": float(data["anytime_current_radius_per_episode"][index]),
                "reference_quantile": (
                    None if reference_quantiles is None else float(reference_quantiles[index])
                ),
                "lower": data["anytime_quantile_lower_history"][index, :count].copy(),
                "upper": data["anytime_quantile_upper_history"][index, :count].copy(),
                "reason": str(data["anytime_termination_reason_per_episode"][index]),
            })
        return panels


def plot_quantile_bounds(path, output_dir, episodes=None):
    panels = quantile_bound_panels(path, episodes)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), 3.8),
        squeeze=False, layout="constrained",
    )
    for ax, panel in zip(axes[0], panels):
        n = np.arange(1, panel["count"] + 1)
        lower, upper = panel["lower"], panel["upper"]
        finite = np.isfinite(upper)
        ax.plot(n, lower, color="tab:blue", label=r"Lower bound $q^{\mathrm{lo}}_{j,n}$")
        ax.plot(
            n, np.where(finite, upper, np.nan), color="tab:orange",
            label=r"Upper bound $q^{\mathrm{up}}_{j,n}$",
        )
        ax.fill_between(n, lower, np.where(finite, upper, np.nan), color="tab:blue", alpha=0.1)
        ax.axhline(panel["radius"], color="0.4", linestyle="-.", label=r"Deployed radius $r_j$")
        ax.axvline(panel["count"], color="tab:red", linestyle="--", label=r"Stopping time $\tau_j$")
        reference = panel["reference_quantile"]
        if reference is not None and np.isfinite(reference):
            ax.axhline(
                reference, color="black", linestyle=":",
                label=r"Independent estimate of $q_j$",
            )
        ax.set_title(f"Episode {panel['episode']}: {panel['count']:,} samples\n{panel['reason']}")
        ax.set_xlabel(r"Calibration trajectories ($n$)")
        ax.set_ylabel("Score quantile (m)")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, max(1, panel["count"] * 1.05))
    legend = {}
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(
        legend.values(), legend.keys(), loc="outside lower center",
        ncol=3 if len(legend) > 4 else 4, fontsize=9,
    )
    return _save_figure(fig, output_dir, "anytime_quantile_bounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bounds_episodes", type=int, nargs="+")
    args = parser.parse_args()
    with np.load(args.plot_data) as data:
        procedure = str(data["update_procedure"])
    labels = {"anytime": "Anytime", "anytime-nonparametric": "Nonparametric anytime"}
    paths = plot_cumulative_trajectories(
        {labels.get(procedure, procedure): args.plot_data}, args.output_dir,
    )
    if procedure in labels:
        paths += plot_quantile_bounds(args.plot_data, args.output_dir, args.bounds_episodes)
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
