#!/usr/bin/env python3
"""Paper-style comparison with optional informed, hybrid, and directional runs.

Episode alignment and terminal-value repetition match plot_combined.py. Select
independently assessed quantiles or reward in the second column. All safety
measurements concern the ego agent only.
"""

import argparse
from pathlib import Path
import shutil
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

from plot_calibration import (
    calibration_budget,
    load_initial_radius,
    quantile_bound_panels,
)


METHODS = {
    "robust": ("Robust (explicit)", "#595959", "s", "robust"),
    "non_interactive": ("Split (non-interactive)", "#D55E00", "v", "non-interactive"),
    "calibrate_once": ("One-time", "#999999", "D", "calibrate-once-paper"),
    "naive": ("Naive", "#E69F00", "^", "naive"),
    "anytime": ("Jeffreys", "#56B4E9", "s", "anytime"),
    "anytime_informed": ("Informed", "#CC79A7", "v", "anytime"),
    "anytime_jeffreys_informed": ("Jeffreys-informed", "#009E73", "o", "anytime"),
    "anytime_directional": ("Directional", "#F0E442", "D", "anytime"),
    "anytime_nonparametric": ("Normal", "#0072B2", "^", "anytime-nonparametric"),
}
OPTIONAL_METHODS = ("anytime_informed", "anytime_jeffreys_informed", "anytime_directional")
ANYTIME_FAMILIES = {
    "anytime": "jeffreys", "anytime_informed": "informed",
    "anytime_jeffreys_informed": "jeffreys-informed", "anytime_directional": "directional",
    "anytime_nonparametric": "normal",
}
GROUPS = (tuple(METHODS)[:4], tuple(METHODS)[4:])
PAPER_STYLE = {
    "text.usetex": shutil.which("latex") is not None,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in METHODS:
        parser.add_argument(f"--{key}", required=key not in OPTIONAL_METHODS, help=f"{key} metrics .npz")
    parser.add_argument("--output_dir", help="Default: plots/compact_<robust directory>")
    parser.add_argument(
        "--second-panel", choices=("qj", "reward"), default="reward",
        help="qj: independent quantile column; reward: reward column and dotted qj on radius",
    )
    parser.add_argument("--alpha", type=float, help="Reward quantile tails only; default: saved alpha")
    parser.add_argument("--bounds_episodes", type=int, nargs="+", help="Default: first, middle, last")
    parser.add_argument("--tube_agent", type=int, default=0, help="Non-ego agent index for tube panels")
    parser.add_argument("--elev", type=float, default=20, help="3D camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=45, help="3D camera azimuth in degrees")
    return parser.parse_args(argv)


def load_run(path):
    with np.load(path) as data:
        run = {
            "path": Path(path),
            "initial_radius": load_initial_radius(path, data),
            "episodes": data["episodes"],
            "radius": data["radius_per_episode"],
            "tube": data["tube_coverage_per_episode"],
            "safety": data["safety_per_episode"],
            "cumulative_reward": data["cumulative_reward_per_episode"],
            "cumulative_reward_runs": data.get("cumulative_reward_per_run"),
            "alpha": float(data["alpha"]) if "alpha" in data else 0.1,
            "update_procedure": str(data["update_procedure"]),
            "anytime_family": str(data["anytime_family"]) if "anytime_family" in data else None,
            "anytime_current_radius": data.get("anytime_current_radius_per_episode"),
            "anytime_next_radius": data.get("anytime_next_radius_per_episode"),
            "locations": data.get("agent_locs_per_episode"),
            "predictions": data.get("predicted_traj_per_episode"),
            "tube_radius": data.get("tube_radius_per_episode"),
            "assessment_radii": data.get("assessment_radii"),
            "assessment_quantiles": data.get("assessment_quantiles"),
        }
    n = len(run["episodes"])
    if not n or run["episodes"].ndim != 1:
        raise ValueError(f"Expected a nonempty episode vector in {path}")
    for field in ("radius", "tube", "safety", "cumulative_reward"):
        if run[field].shape != (n,):
            raise ValueError(f"{field} must match episodes in {path}")
    rewards = run["cumulative_reward_runs"]
    if rewards is not None and (rewards.ndim != 2 or len(rewards) != n):
        raise ValueError(f"Per-run rewards must have one row per episode in {path}")
    radii, quantiles = run["assessment_radii"], run["assessment_quantiles"]
    if radii is not None or quantiles is not None:
        iterative = run["update_procedure"] in ("robust", "naive", "anytime", "anytime-nonparametric")
        expected = np.r_[run["initial_radius"], run["radius"]] if iterative else run["radius"]
        if (radii is None or quantiles is None or radii.shape != expected.shape
                or quantiles.shape != expected.shape or not np.all(np.isfinite(quantiles))
                or not np.allclose(radii, expected, rtol=1e-6, atol=1e-7)):
            raise ValueError(f"Independent assessment quantiles must align with the policy radii in {path}")
    return run


def prepare_conformal_radius(radius, initial_radius=8.0):
    return np.insert(radius, 0, initial_radius)[:-1]


def pad_repeat(arr, target_len):
    arr = np.asarray(arr)
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.repeat(arr[-1:], target_len - len(arr), axis=0)])


def prepare_anytime_run(run, target_len, expected_procedure):
    if run["update_procedure"] != expected_procedure:
        raise ValueError(f"Expected {expected_procedure!r}, got {run['update_procedure']!r}")
    current, following = run["anytime_current_radius"], run["anytime_next_radius"]
    if current is None or following is None:
        raise ValueError(f"{expected_procedure} is missing anytime radius fields")
    n = len(run["episodes"])
    if not n:
        raise ValueError(f"{expected_procedure} metrics contain no episodes")
    fields = ("radius", "cumulative_reward", "tube", "safety", "cumulative_reward_runs")
    for values in [current, following] + [run[key] for key in fields]:
        if values is not None and len(values) != n:
            raise ValueError(f"{expected_procedure} fields must match episodes")
    if not np.allclose(following, run["radius"], rtol=1e-6, atol=1e-7):
        raise ValueError("Anytime next-radius and deployed-radius data disagree")
    aligned = {key: pad_repeat(run[key], target_len) if run[key] is not None else None for key in fields}
    aligned["radius"] = pad_repeat(np.concatenate([current[:1], following]), target_len)
    return aligned


def prepare_suite(paths):
    runs = {key: load_run(paths[key]) for key in METHODS if paths.get(key) is not None}
    base = runs["robust"]["episodes"]
    if not np.array_equal(runs["naive"]["episodes"], base):
        raise ValueError("Naive episode grid must match Robust")
    n = len(base)
    series = {}
    for key, run in runs.items():
        label, color, marker, procedure = METHODS[key]
        if key == "calibrate_once" and run["update_procedure"] == "calibrate-once":
            label, procedure = "Calibrate-once (legacy)", "calibrate-once"
        if key in ANYTIME_FAMILIES:
            family = run["anytime_family"]
            if family is None:
                family = {"anytime": "jeffreys", "anytime-nonparametric": "normal"}.get(run["update_procedure"])
            if family != ANYTIME_FAMILIES[key]:
                raise ValueError(f"{key}: expected family {ANYTIME_FAMILIES[key]}, got {family}")
            if key == "anytime_nonparametric" and run["update_procedure"] == "anytime":
                procedure = "anytime"
        if run["update_procedure"] != procedure:
            raise ValueError(f"{key}: expected {procedure}, got {run['update_procedure']}")
        if not np.isclose(run["alpha"], runs["robust"]["alpha"]):
            raise ValueError("All runs must have the same coverage target alpha")
        if key in GROUPS[1]:
            values = prepare_anytime_run(run, n, procedure)
            metric_x = np.arange(1, n + 1)
        else:
            count = n if key in ("robust", "naive") else n + 1
            values = {
                field: pad_repeat(run[field], count) if run[field] is not None else None
                for field in ("tube", "safety", "cumulative_reward", "cumulative_reward_runs")
            }
            values["radius"] = (
                prepare_conformal_radius(run["radius"], run["initial_radius"])
                if key in ("robust", "naive") else pad_repeat(run["radius"], n)
            )
            metric_x = np.arange(1, n + 1) if count == n else np.arange(n + 1)
        budget_x, budget, initial = calibration_budget(run["path"])
        values["qj"] = (
            pad_repeat(run["assessment_quantiles"], n)
            if run["assessment_quantiles"] is not None else None
        )
        series[key] = dict(
            values, label=label, radius_x=base, metric_x=metric_x,
            style=dict(color=color, marker=marker, markeredgecolor="0.25", markeredgewidth=0.35),
            budget_x=budget_x, budget=budget, initial=initial,
        )
    return runs, series


def reward_errors(values, samples, alpha):
    if samples is None:
        return None
    lower, upper = np.quantile(samples, [alpha, 1 - alpha], axis=1)
    return np.vstack([np.maximum(values - lower, 0), np.maximum(upper - values, 0)])


def _row_legend(fig, series, keys, y, prefix):
    handles = [Line2D([], [], color="none", label=prefix)]
    handles += [Line2D([], [], label=series[key]["label"], **series[key]["style"]) for key in keys if key in series]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, y), ncol=len(handles))


def _episode_axis(ax, end):
    ax.set_xlim(0, max(1, end))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))


def _target_line(ax, target):
    line = ax.axhline(target, color="0.25", linestyle=":", linewidth=1.5)
    ax.legend([line], [r"$1-\alpha$"], loc="best", fontsize=8, handlelength=1.2)


def plot_progression(runs, series, alpha, second_panel="reward"):
    if second_panel not in ("qj", "reward"):
        raise ValueError("second_panel must be qj or reward")
    missing = [key for key, s in series.items() if s["qj"] is None]
    if missing:
        message = "Independent assessment q_j missing for: " + ", ".join(missing)
        if second_panel == "qj":
            raise ValueError(message + "; use new metrics or --second-panel reward")
        warnings.warn(message + "; dotted q_j omitted for these old files")
    overlay = second_panel == "reward" and len(missing) < len(series)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), sharey="col")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.88, wspace=0.6, hspace=0.48)
    n = len(runs["robust"]["episodes"])
    target = 100 * (1 - runs["robust"]["alpha"])
    for row, keys in enumerate(GROUPS):
        for col, ax in enumerate(axes[row]):
            ax.set_facecolor("0.96" if row == 0 else "white")
            _episode_axis(ax, n - 1 if col == 0 or (col == 1 and second_panel == "qj") else n)
            ax.tick_params(labelbottom=row == 1, labelleft=True)
        for key in keys:
            if key not in series:
                continue
            s = series[key]
            axes[row, 0].plot(s["radius_x"], s["radius"], **s["style"])
            if second_panel == "qj":
                axes[row, 1].plot(s["radius_x"], s["qj"], **s["style"])
            else:
                if s["qj"] is not None:
                    axes[row, 0].plot(s["radius_x"], s["qj"], linestyle=":",
                                      color=s["style"]["color"])
                axes[row, 1].errorbar(
                    s["metric_x"], s["cumulative_reward"],
                    yerr=reward_errors(s["cumulative_reward"], s["cumulative_reward_runs"], alpha),
                    capsize=1.5, elinewidth=0.65, **s["style"],
                )
            axes[row, 2].plot(s["metric_x"], 100 * s["tube"], **s["style"])
            budget = np.where(s["budget"] > 0, s["budget"], np.nan)
            axes[row, 3].plot(s["budget_x"], budget, **s["style"])
            if s["initial"]:
                axes[row, 3].plot(s["budget_x"], s["budget"] + s["initial"],
                                  color=s["style"]["color"], linestyle="--", linewidth=0.9)
        _target_line(axes[row, 2], target)
        # Calibration counts keep their native episode grid, including late stops.
        axes[row, 3].set_xlim(0, max(n, max(s["budget_x"][-1] for s in series.values())))
        axes[row, 3].set_yscale("log")
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        axes[row, 3].yaxis.set_major_formatter(formatter)
        labels = (r"$r_j,\ q_j$ (m)" if overlay else r"$r_j$ (m)",
                  r"$q_j$ (m)" if second_panel == "qj" else "Reward (m)",
                  r"$1-p_j$ ($\%$)", "Trajectories")
        for ax, label in zip(axes[row], labels):
            ax.set_ylabel(label, labelpad=2)
    axes[0, 2].set_ylim(max(0, min(target, min(np.min(s["tube"]) * 100 for s in series.values())) - 4), 102)
    _row_legend(fig, series, GROUPS[0], 0.955, "CP:")
    _row_legend(fig, series, GROUPS[1], 0.51, "Mixtures:")
    fig.supxlabel(r"Number of episodes $j$", y=0.06, fontsize=10)
    footer = "Trajectories: solid = update calibration; dashed = including initialization."
    if overlay:
        footer += r"  Radius: solid $r_j$; dotted $q_j$."
    fig.text(0.5, 0.012, footer,
             ha="center", va="bottom", fontsize=7)
    return fig


def plot_safety(runs, series):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.76, wspace=0.2)
    for ax, keys in zip(axes, GROUPS):
        for key in keys:
            if key not in series:
                continue
            s = series[key]
            ax.plot(s["metric_x"], 100 * s["safety"], label=s["label"], **s["style"])
        _episode_axis(ax, len(runs["robust"]["episodes"]))
        _target_line(ax, 100 * (1 - runs["robust"]["alpha"]))
        ax.set_xlabel(r"Number of episodes $j$")
    axes[0].set_facecolor("0.96")
    axes[0].set_ylabel(r"Ego safety ($\%$)")
    axes[0].set_ylim(max(0, min(100 * (1 - runs["robust"]["alpha"]),
                              min(np.min(s["safety"]) * 100 for s in series.values())) - 2), 102)
    _row_legend(fig, series, GROUPS[0], 0.94, "CP:")
    _row_legend(fig, series, GROUPS[1], 0.84, "Mixtures:")
    return fig


def plot_bounds(path, episodes=None):
    with np.load(path) as data:
        required = ("anytime_quantile_lower_history", "anytime_quantile_upper_history",
                    "anytime_termination_reason_per_episode")
        if any(key not in data for key in required):
            warnings.warn(f"Skipping quantile bounds: histories missing in {path}")
            return None
    panels = quantile_bound_panels(path, episodes)
    if any(p["reference_quantile"] is None or not np.isfinite(p["reference_quantile"]) for p in panels):
        warnings.warn("Independent q_j reference unavailable for some panels; dotted reference omitted there.")
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.55), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.74, wspace=0.32)
    color = "#009E73"
    bounds_label = (r"$\underline{q}_{j,n},\ \overline{q}_{j,n}$" if plt.rcParams["text.usetex"]
                    else r"$q^{\mathrm{lo}}_{j,n},\ q^{\mathrm{up}}_{j,n}$")
    for ax, panel in zip(axes[0], panels):
        count = panel["count"]
        x = np.arange(1, count + 1)
        lower = np.where(np.isfinite(panel["lower"]), panel["lower"], np.nan)
        upper = np.where(np.isfinite(panel["upper"]), panel["upper"], np.nan)
        ax.step(x, lower, where="post", color=color, label=bounds_label)
        ax.step(x, upper, where="post", color=color)
        ax.fill_between(x, lower, upper, step="post", color=color, alpha=0.2)
        reference = panel["reference_quantile"]
        if reference is not None and np.isfinite(reference):
            ax.axhline(reference, color="0.25", linestyle=":", label=r"$q_j$ (independent estimate)")
        for bound in (lower, upper):
            if np.isfinite(bound[-1]):
                ax.plot(count, bound[-1], "o", color=color, markeredgecolor="black",
                        markeredgewidth=0.6, label=r"$n=\tau_j$")
        ax.set_title(rf"$j={panel['episode']}$", pad=3)
        ax.set_xlim(0, max(1, count * 1.04))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0, 0].set_ylabel(bounds_label + " (m)")
    legend = {}
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(legend.values(), legend.keys(), loc="center", bbox_to_anchor=(0.5, 0.94), ncol=3)
    fig.supxlabel(r"Calibration trajectories $n$", y=0.06, fontsize=10)
    return fig


def validate_trajectories(run):
    locations, predictions = run["locations"], run["predictions"]
    if locations is None or predictions is None:
        warnings.warn(f"Skipping trajectories: positions or predictions missing in {run['path']}")
        return False
    n = len(run["episodes"])
    if (locations.ndim != 4 or locations.shape[0] != n or locations.shape[-1] != 3
            or locations.shape[1] < 2 or predictions.ndim != 4
            or predictions.shape[:3] != (n, locations.shape[1] - 1, locations.shape[2])
            or predictions.shape[-1] < 3):
        raise ValueError(f"Invalid trajectory shapes in {run['path']}")
    return True


def sphere_mesh(center, radius):
    u, v = np.linspace(0, 2 * np.pi, 17), np.linspace(0, np.pi, 13)
    return (center[0] + radius * np.outer(np.cos(u), np.sin(v)),
            center[1] + radius * np.outer(np.sin(u), np.sin(v)),
            center[2] + radius * np.outer(np.ones_like(u), np.cos(v)))


def _spatial_limits(points):
    points = np.concatenate([np.asarray(p).reshape(-1, 3) for p in points])
    low, high = points.min(axis=0), points.max(axis=0)
    margin = np.maximum(0.05 * (high - low), 0.1)
    return low - margin, high + margin


def _spatial_axis(ax, limits, elev, azim):
    low, high = limits
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$x$ (m)", labelpad=-3)
    ax.set_ylabel(r"$y$ (m)", labelpad=-3)
    ax.set_zlabel(r"$z$ (m)", labelpad=-3)
    ax.tick_params(pad=0, labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))
        axis.set_pane_color((1, 1, 1, 1))


def _plot_ego(ax, locations):
    ax.plot(*locations.T, color="#D55E00", linewidth=1.5)
    ax.plot(*locations[0], "o", color="#D55E00", markersize=3)
    ax.plot(*locations[-1], "x", color="#D55E00", markersize=4)


def plot_final_trajectories(runs, series, elev, azim):
    available = [validate_trajectories(run) for run in runs.values()]
    if not all(available):
        return None
    limits = _spatial_limits([p for run in runs.values() for p in
                              (run["locations"][-1], run["predictions"][-1, :, :, :3])])
    keys = [key for key in METHODS if key in runs]
    rows = (len(keys) + 2) // 3
    fig = plt.figure(figsize=(7.2, 2.45 * rows))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.85, wspace=0.25, hspace=0.15)
    for index, key in enumerate(keys):
        run = runs[key]
        ax = fig.add_subplot(rows, 3, index + 1, projection="3d", computed_zorder=False)
        for locations in run["locations"][-1, :-1]:
            ax.plot(*locations.T, color="#668699", linewidth=0.8, alpha=0.75)
        for prediction in run["predictions"][-1]:
            ax.plot(*prediction[:, :3].T, color="0.6", linestyle=":", linewidth=0.7)
        _plot_ego(ax, run["locations"][-1, -1])
        _spatial_axis(ax, limits, elev, azim)
        ax.set_title(series[key]["label"], pad=0)
    fig.legend(handles=[
        Line2D([], [], color="#D55E00", label="Ego"),
        Line2D([], [], color="#668699", label="Other agents"),
        Line2D([], [], color="0.6", linestyle=":", label="Predictions"),
        Line2D([], [], color="#D55E00", linestyle="none", marker="o", label="Ego start"),
        Line2D([], [], color="#D55E00", linestyle="none", marker="x", label="Ego end"),
    ], loc="center", bbox_to_anchor=(0.5, 0.96), ncol=5)
    return fig


def plot_tube_progression(run, agent, elev, azim):
    if not validate_trajectories(run):
        return None
    if not 0 <= agent < run["predictions"].shape[1]:
        raise ValueError("tube_agent must index a non-ego agent")
    if run["tube_radius"] is None:
        warnings.warn("Skipping tube progression: saved tube radii unavailable")
        return None
    if run["tube_radius"].shape != run["episodes"].shape or np.any(run["tube_radius"] < 0):
        raise ValueError("Tube radii must be nonnegative and match saved episodes")
    indices = list(dict.fromkeys([0, min(1, len(run["episodes"]) - 1), len(run["episodes"]) - 1]))
    points = []
    for index in indices:
        prediction = run["predictions"][index, agent, :, :3]
        radius = run["tube_radius"][index]
        points.extend([run["locations"][index, [agent, -1]], prediction - radius, prediction + radius])
    limits = _spatial_limits(points)
    fig = plt.figure(figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.79, wspace=0.25)
    for column, index in enumerate(indices):
        ax = fig.add_subplot(1, len(indices), column + 1, projection="3d", computed_zorder=False)
        prediction = run["predictions"][index, agent, :, :3]
        radius = run["tube_radius"][index]
        # Sparse true spheres show tube cross-sections without inventing a 2D geometry.
        for t in np.unique(np.linspace(0, len(prediction) - 1, 12, dtype=int)):
            if radius > 0:
                ax.plot_surface(*sphere_mesh(prediction[t], radius), color="#56B4E9",
                                alpha=0.07, linewidth=0, shade=False, rasterized=True)
        ax.plot(*prediction.T, color="#0072B2", linestyle=":", linewidth=1)
        ax.plot(*run["locations"][index, agent].T, color="#0072B2", linewidth=1)
        _plot_ego(ax, run["locations"][index, -1])
        _spatial_axis(ax, limits, elev, azim)
        # Saved anytime rollouts assess the post-update policy, not current_radius.
        ax.set_title(rf"$j={run['episodes'][index]},\ r_{{j+1}}={run['radius'][index]:.3f}$ m", pad=0)
    fig.legend(handles=[
        Line2D([], [], color="#D55E00", label="Ego"),
        Line2D([], [], color="#0072B2", label=f"Agent {agent}"),
        Line2D([], [], color="#0072B2", linestyle=":", label="Prediction"),
        Line2D([], [], color="#56B4E9", linewidth=5, alpha=0.4, label="Tube"),
    ], loc="center", bbox_to_anchor=(0.5, 0.96), ncol=4)
    return fig


def main(argv=None):
    args = parse_args(argv)
    runs, series = prepare_suite({key: getattr(args, key) for key in METHODS if getattr(args, key)})
    alpha = args.alpha if args.alpha is not None else runs["robust"]["alpha"]
    if not 0 < alpha < 0.5:
        raise ValueError("Reward quantile alpha must be between 0 and 0.5")
    output_dir = Path(args.output_dir or Path("plots") / f"compact_{Path(args.robust).parent.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_STYLE):
        figures = {
            "episode_progression": plot_progression(runs, series, alpha, args.second_panel),
            "ego_safety": plot_safety(runs, series),
            "anytime_quantile_bounds": plot_bounds(args.anytime, args.bounds_episodes),
            "final_trajectories": plot_final_trajectories(runs, series, args.elev, args.azim),
            "anytime_tube_progression": plot_tube_progression(runs["anytime"], args.tube_agent, args.elev, args.azim),
        }
        for stem, fig in figures.items():
            if fig is not None:
                path = output_dir / f"{stem}.pdf"
                fig.savefig(path, dpi=220)
                plt.close(fig)
                print(path.resolve())


if __name__ == "__main__":
    main()
