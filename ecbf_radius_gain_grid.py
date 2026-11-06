#!/usr/bin/env python3
"""Run matched fixed-radius diagnostics over ECBF radius and pole gain."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter

import cvxpy as cp
import numpy as np
from scipy.stats import binom, sem, t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ECBF radius and CBF_K1 using identical seeded disturbances "
            "at every grid point."
        )
    )
    parser.add_argument("--radius-start", type=float, default=1.2)
    parser.add_argument("--radius-stop", type=float, default=1.9)
    parser.add_argument("--radius-step", type=float, default=0.05)
    parser.add_argument(
        "--cbf-k1",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    )
    parser.add_argument("--quantile", type=float, default=0.94)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--num-multi-agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=439)
    parser.add_argument("--multi-experiment", default="multi_ddg_6")
    parser.add_argument("--solo-experiment", default="solo_ddg")
    parser.add_argument(
        "--init-predictions",
        default="homog_gtg_pred_traj_data_seed439",
    )
    parser.add_argument("--multi-train-dir", default="train_dir")
    parser.add_argument("--solo-train-dir", default="train_dir")
    parser.add_argument("--predictions-dir", default="train_dir")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--output-dir",
        default="train_dir/ecbf_radius_gain_grid_rigorous_seed439",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--prepare-nominal-only",
        action="store_true",
        help="Create or validate the shared nominal reference, then exit.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Validate all expected point files and write the combined CSV.",
    )
    parser.add_argument(
        "--include-nominal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a matched no-CBF baseline and compute paired effects.",
    )
    parser.add_argument(
        "--interaction-distances",
        type=float,
        nargs="+",
        default=[1.5, 2.0],
        help="Distances used to count simultaneous close ego/other interactions.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-nominal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-radius", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-k1", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--nominal-summary", help=argparse.SUPPRESS)
    parser.add_argument("--nominal-trajectories", help=argparse.SUPPRESS)
    return parser.parse_args()


def radius_grid(start: float, stop: float, step: float) -> list[float]:
    if start <= 0.0 or stop < start or step <= 0.0:
        raise ValueError("Require 0 < radius-start <= radius-stop and radius-step > 0.")
    count = int(np.floor((stop - start) / step + 1.0e-9)) + 1
    values = [start + index * step for index in range(count)]
    if values[-1] < stop - 1.0e-9:
        values.append(stop)
    return [float(value) for value in values]


def _tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _stats(values) -> dict[str, float]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if array.size == 0:
        return {key: float("nan") for key in ("min", "p50", "p90", "p95", "p99", "max", "mean")}
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _mean_confidence_interval(values, confidence: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2 or np.all(array == array[0]):
        value = float(array[0]) if array.size else float("nan")
        return value, value
    low, high = t.interval(
        confidence,
        array.size - 1,
        loc=float(np.mean(array)),
        scale=float(sem(array)),
    )
    return float(low), float(high)


def _quantile_confidence_interval(
    values,
    quantile: float,
    confidence: float,
) -> tuple[float, float]:
    """Distribution-free interval formed from binomial order statistics."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return float("nan"), float("nan")
    tail = 1.0 - confidence
    lower_rank = int(binom.ppf(tail / 2.0, ordered.size, quantile))
    upper_rank = int(binom.ppf(1.0 - tail / 2.0, ordered.size, quantile)) + 1
    lower_index = max(lower_rank - 1, 0)
    upper_index = min(upper_rank - 1, ordered.size - 1)
    return float(ordered[lower_index]), float(ordered[upper_index])


def _flatten_summary(summary: dict) -> dict:
    row = {
        "controller": summary["controller"],
        "radius": summary["radius"],
        "cbf_k1": summary["cbf_k1"],
        "score_quantile_level": summary["score_quantile_level"],
        "score_quantile": summary["score_quantile"],
        "score_quantile_ci_low": summary["score_quantile_ci"][0],
        "score_quantile_ci_high": summary["score_quantile_ci"][1],
        "num_trajectories": summary["num_trajectories"],
        "qp_calls": summary["qp_calls"],
        "solver_ecos": summary["solver_counts"].get("ECOS", 0),
        "solver_clarabel": summary["solver_counts"].get("CLARABEL", 0),
        "solver_scs": summary["solver_counts"].get("SCS", 0),
        "solver_fallback": summary["solver_counts"].get("fallback", 0),
        "action_intervention_gt_0p1": summary["action_intervention_gt_0p1"],
        "slack_gt_1": summary["slack_gt_1"],
        "slack_gt_100": summary["slack_gt_100"],
        "dominant_error_agent": summary["dominant_error_agent"],
    }
    for name in (
        "joint_error",
        "score",
        "closest_ego_other",
        "action_intervention",
        "action_jump",
        "slack",
        "min_predicted_rho",
        "max_gradient_norm",
        "goal_progress",
        "net_goal_progress",
        "ego_path_length",
        "final_goal_distance",
        "path_efficiency",
        "agents_within_1p5",
        "agents_within_2p0",
        "simultaneous_agents_within_1p5",
        "simultaneous_agents_within_2p0",
        "position_deviation_vs_nominal",
        "heading_deviation_vs_nominal_deg",
        "separation_delta_vs_nominal",
        "goal_progress_delta_vs_nominal",
        "net_goal_progress_delta_vs_nominal",
        "path_efficiency_delta_vs_nominal",
    ):
        for statistic, value in summary[name].items():
            row[f"{name}_{statistic}"] = value
        confidence_interval = summary.get(f"{name}_mean_ci")
        if confidence_interval is not None:
            row[f"{name}_mean_ci_low"] = confidence_interval[0]
            row[f"{name}_mean_ci_high"] = confidence_interval[1]
    for agent, value in enumerate(summary["per_agent_error_quantile"]):
        row[f"agent_{agent}_error_quantile"] = value
    return row


def _write_csv(summaries: list[dict], path: str) -> None:
    rows = [_flatten_summary(summary) for summary in summaries]
    rows.sort(
        key=lambda row: (
            row["controller"] != "nominal",
            row["cbf_k1"] if row["cbf_k1"] is not None else -1.0,
            row["radius"] if row["radius"] is not None else -1.0,
        )
    )
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _point_stem(radius: float, gain: float) -> str:
    return f"r{_tag(radius)}_k1_{_tag(gain)}"


def run_grid(args: argparse.Namespace) -> None:
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must lie strictly between zero and one.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one.")
    if args.num_trajectories < 5:
        raise ValueError("--num-trajectories must be at least 5.")
    if args.episode_length <= 0 or args.num_multi_agents <= 0:
        raise ValueError("Episode length and number of agents must be positive.")
    if any(gain <= 0.0 for gain in args.cbf_k1):
        raise ValueError("All --cbf-k1 values must be positive.")
    if any(distance <= 0.0 for distance in args.interaction_distances):
        raise ValueError("All --interaction-distances must be positive.")
    if args.interaction_distances != [1.5, 2.0]:
        raise ValueError("This version requires --interaction-distances 1.5 2.0.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")
    if args.prepare_nominal_only and args.merge_only:
        raise ValueError("--prepare-nominal-only and --merge-only are mutually exclusive.")
    if args.prepare_nominal_only and not args.include_nominal:
        raise ValueError("--prepare-nominal-only requires --include-nominal.")

    radii = radius_grid(args.radius_start, args.radius_stop, args.radius_step)
    output_dir = os.path.abspath(args.output_dir)
    points_dir = os.path.join(output_dir, "points")
    logs_dir = os.path.join(output_dir, "logs")
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(points_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    config_name = (
        f"grid_config_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
        if args.num_shards > 1 and not args.merge_only
        else "grid_config.json"
    )
    with open(os.path.join(output_dir, config_name), "w", encoding="utf-8") as handle:
        json.dump(
            {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("worker") and not key.startswith("nominal_")
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    path_args = {
        "multi_train_dir": os.path.abspath(args.multi_train_dir),
        "solo_train_dir": os.path.abspath(args.solo_train_dir),
        "predictions_dir": os.path.abspath(args.predictions_dir),
    }
    all_points = [(radius, gain) for gain in args.cbf_k1 for radius in radii]
    points = all_points[args.shard_index::args.num_shards]
    summaries = []
    started = time.monotonic()
    nominal_summary_path = os.path.join(points_dir, "nominal.json")
    nominal_trajectories_path = os.path.join(points_dir, "nominal_trajectories.npz")

    common_command = [
        "--quantile", str(args.quantile),
        "--confidence", str(args.confidence),
        "--num-trajectories", str(args.num_trajectories),
        "--episode-length", str(args.episode_length),
        "--num-multi-agents", str(args.num_multi_agents),
        "--seed", str(args.seed),
        "--multi-experiment", args.multi_experiment,
        "--solo-experiment", args.solo_experiment,
        "--init-predictions", args.init_predictions,
        "--multi-train-dir", path_args["multi_train_dir"],
        "--solo-train-dir", path_args["solo_train_dir"],
        "--predictions-dir", path_args["predictions_dir"],
        "--device", args.device,
        "--output-dir", output_dir,
        "--interaction-distances", *[str(value) for value in args.interaction_distances],
    ]

    if args.merge_only:
        missing = []
        mismatched = []
        if args.include_nominal:
            if not os.path.exists(nominal_summary_path) or not os.path.exists(nominal_trajectories_path):
                missing.append("nominal baseline")
            else:
                with open(nominal_summary_path, "r", encoding="utf-8") as handle:
                    nominal_summary = json.load(handle)
                if _nominal_summary_matches(nominal_summary, args):
                    summaries.append(nominal_summary)
                else:
                    mismatched.append("nominal baseline")
        for radius, gain in all_points:
            stem = _point_stem(radius, gain)
            point_path = os.path.join(points_dir, f"{stem}.json")
            if not os.path.exists(point_path):
                missing.append(stem)
                continue
            with open(point_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if _point_summary_matches(summary, args, radius, gain):
                summaries.append(summary)
            else:
                mismatched.append(stem)
        if missing or mismatched:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if mismatched:
                details.append(f"configuration mismatch: {', '.join(mismatched)}")
            raise RuntimeError("Cannot merge incomplete grid; " + "; ".join(details))
        csv_path = os.path.join(output_dir, "ecbf_radius_gain_grid.csv")
        _write_csv(summaries, csv_path)
        print(f"Validated {len(all_points)} grid points and saved {csv_path}")
        return

    if args.include_nominal:
        reuse_nominal = False
        if not args.overwrite and os.path.exists(nominal_summary_path) and os.path.exists(nominal_trajectories_path):
            with open(nominal_summary_path, "r", encoding="utf-8") as handle:
                nominal_summary = json.load(handle)
            reuse_nominal = _nominal_summary_matches(nominal_summary, args)
        if reuse_nominal:
            print("Reusing matched nominal baseline", flush=True)
        else:
            if args.num_shards > 1 and not args.prepare_nominal_only:
                raise RuntimeError(
                    "Shared nominal baseline is missing or incompatible. Run once with "
                    "--prepare-nominal-only before launching parallel shards."
                )
            print("Running matched nominal baseline", flush=True)
            nominal_command = [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--worker-nominal",
                "--worker-radius", str(radii[0]),
                "--worker-k1", "1.0",
                "--worker-output", nominal_summary_path,
                "--nominal-trajectories", nominal_trajectories_path,
                *common_command,
            ]
            with open(os.path.join(logs_dir, "nominal.log"), "w", encoding="utf-8") as log:
                subprocess.run(nominal_command, stdout=log, stderr=subprocess.STDOUT, check=True)
            with open(nominal_summary_path, "r", encoding="utf-8") as handle:
                nominal_summary = json.load(handle)
        summaries.append(nominal_summary)
    if args.prepare_nominal_only:
        print(f"Nominal baseline is ready at {nominal_summary_path}")
        return

    for point_index, (radius, gain) in enumerate(points, start=1):
        stem = _point_stem(radius, gain)
        point_path = os.path.join(points_dir, f"{stem}.json")
        log_path = os.path.join(logs_dir, f"{stem}.log")
        reuse_point = False
        if not args.overwrite and os.path.exists(point_path):
            with open(point_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            reuse_point = _point_summary_matches(summary, args, radius, gain)
        if reuse_point:
            summaries.append(summary)
            print(f"[{point_index}/{len(points)}] Reusing {stem}", flush=True)
            continue

        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--worker-radius", str(radius),
            "--worker-k1", str(gain),
            "--worker-output", point_path,
            *common_command,
        ]
        if args.include_nominal:
            command.extend(
                [
                    "--nominal-summary", nominal_summary_path,
                    "--nominal-trajectories", nominal_trajectories_path,
                ]
            )
        print(f"[{point_index}/{len(points)}] Running {stem}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        with open(point_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append(summary)
        elapsed = time.monotonic() - started
        remaining = elapsed / point_index * (len(points) - point_index)
        print(
            f"  score_p{100 * args.quantile:g}={summary['score_quantile']:.4f} "
            f"fallbacks={summary['solver_counts'].get('fallback', 0)} "
            f"estimated_remaining={remaining / 60.0:.1f} min",
            flush=True,
        )

    csv_name = (
        f"ecbf_radius_gain_grid_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
        if args.num_shards > 1
        else "ecbf_radius_gain_grid.csv"
    )
    csv_path = os.path.join(output_dir, csv_name)
    _write_csv(summaries, csv_path)
    print(f"Saved grid summary to {csv_path}")


def _summary_matches(summary: dict, args: argparse.Namespace) -> bool:
    return (
        summary.get("schema_version") == 2
        and summary.get("num_trajectories") == args.num_trajectories
        and summary.get("episode_length") == args.episode_length
        and summary.get("seed") == args.seed
        and np.isclose(summary.get("score_quantile_level", -1.0), args.quantile)
        and np.isclose(summary.get("confidence", -1.0), args.confidence)
    )


def _nominal_summary_matches(summary: dict, args: argparse.Namespace) -> bool:
    return summary.get("controller") == "nominal" and _summary_matches(summary, args)


def _point_summary_matches(
    summary: dict,
    args: argparse.Namespace,
    radius: float,
    gain: float,
) -> bool:
    return (
        summary.get("controller") == "ecbf"
        and _summary_matches(summary, args)
        and np.isclose(summary.get("radius", -1.0), radius)
        and np.isclose(summary.get("cbf_k1", -1.0), gain)
    )


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_radius is None or args.worker_k1 is None or not args.worker_output:
        raise ValueError("Worker radius, gain, and output are required.")

    import conformal
    import project_utils.full_cbf_utils as cbf
    from project_utils.conformal_utils import joint_rollout_error

    requested_radius = float(args.worker_radius)
    cbf.CBF_K1 = float(args.worker_k1)
    qp_records = []
    action_records = []
    calibration_runs = []
    ego_trajectories = []
    pending_geometry = []
    context = {"radius": None}
    active_attempts = {"value": None}

    original_coefficients = cbf._ecbf_coefficients
    original_qp_solve = cbf._solve_parameterized_qp
    original_problem_solve = cp.Problem.solve
    original_make_filter = conformal.make_cbf_filter
    original_joint_radius = conformal.joint_conformal_radius

    def coefficients(*call_args, **call_kwargs):
        phi_const, phi_grad, debug = original_coefficients(*call_args, **call_kwargs)
        pending_geometry.append(
            (float(debug["rho"]), float(np.linalg.norm(phi_grad)))
        )
        return phi_const, phi_grad, debug

    def problem_solve(problem, *call_args, **call_kwargs):
        attempts = active_attempts["value"]
        if attempts is None:
            return original_problem_solve(problem, *call_args, **call_kwargs)
        solver = str(call_kwargs.get("solver", "default"))
        try:
            result = original_problem_solve(problem, *call_args, **call_kwargs)
        except Exception as error:
            attempts.append((solver, type(error).__name__, None))
            raise
        attempts.append((solver, None, str(problem.status)))
        return result

    def qp_solve(qp, *, u_min, u_max, fallback_thrust):
        count = len(qp.phi_exprs)
        geometry = pending_geometry[-count:]
        del pending_geometry[-count:]
        u_ref = np.asarray(qp.u_ref.value, dtype=np.float64).copy()
        attempts = []
        active_attempts["value"] = attempts
        result = original_qp_solve(
            qp,
            u_min=u_min,
            u_max=u_max,
            fallback_thrust=fallback_thrust,
        )
        active_attempts["value"] = None
        valid = cbf._has_valid_optimal_solution(qp, u_min, u_max)
        qp_records.append(
            {
                "radius": context["radius"],
                "solver": attempts[-1][0] if valid else "fallback",
                "slack": float(qp.slack.value) if valid else None,
                "intervention": float(np.linalg.norm(result - u_ref)),
                "min_rho": min(value[0] for value in geometry),
                "max_gradient": max(value[1] for value in geometry),
            }
        )
        return result

    def make_filter(radii, *call_args, **call_kwargs):
        action_filter = (
            None
            if args.worker_nominal
            else original_make_filter(radii, *call_args, **call_kwargs)
        )
        filter_radius = float(np.asarray(radii).reshape(-1)[0])
        previous_action = None
        previous_tick = None

        def wrapped(base_action, env_state, swarm_state, debug=False):
            nonlocal previous_action, previous_tick
            tick = int(env_state.envs[-1].tick)
            if previous_tick is not None and tick < previous_tick:
                previous_action = None
            context["radius"] = filter_radius
            if action_filter is None:
                output = np.asarray(base_action, dtype=np.float32).copy()
            else:
                output = action_filter(base_action, env_state, swarm_state, debug=debug)
            safe_action = np.asarray(output[0] if debug else output, dtype=np.float64)
            nominal_action = np.asarray(base_action, dtype=np.float64)
            action_records.append(
                {
                    "radius": filter_radius,
                    "intervention": float(np.linalg.norm(safe_action - nominal_action)),
                    "jump": None if previous_action is None else float(np.linalg.norm(safe_action - previous_action)),
                }
            )
            previous_action = safe_action.copy()
            previous_tick = tick
            context["radius"] = None
            return output

        return wrapped

    def joint_radius(logs, num_agents, predictions, alpha, episode_length, min_radius):
        for run_id in range(len(logs[0])):
            joint_error = joint_rollout_error(
                logs, predictions, num_agents, episode_length, run_id
            )
            ego = np.asarray(logs[num_agents][run_id]["position"])[:episode_length]
            ego_trajectories.append(ego.copy())
            agent_errors = []
            agent_distances = []
            distances_over_time = []
            for agent in range(num_agents):
                actual = np.asarray(logs[agent][run_id]["position"])[:episode_length]
                predicted = np.asarray(predictions[agent])[:episode_length, :3]
                agent_errors.append(float(np.max(np.linalg.norm(actual - predicted, axis=1))))
                distances = np.linalg.norm(ego - actual, axis=1)
                distances_over_time.append(distances)
                agent_distances.append(float(np.min(distances)))
            distances_over_time = np.stack(distances_over_time, axis=1)
            calibration_runs.append(
                {
                    "joint_error": joint_error,
                    "score": float(min_radius + joint_error),
                    "agent_errors": agent_errors,
                    "agent_distances": agent_distances,
                    "closest_ego_other": min(agent_distances),
                    "agents_within_1p5": int(np.sum(np.asarray(agent_distances) <= 1.5)),
                    "agents_within_2p0": int(np.sum(np.asarray(agent_distances) <= 2.0)),
                    "simultaneous_agents_within_1p5": int(np.max(np.sum(distances_over_time <= 1.5, axis=1))),
                    "simultaneous_agents_within_2p0": int(np.max(np.sum(distances_over_time <= 2.0, axis=1))),
                    **_goal_metrics(logs[num_agents][run_id], episode_length),
                }
            )
        return original_joint_radius(
            logs, num_agents, predictions, alpha, episode_length, min_radius
        )

    cbf._ecbf_coefficients = coefficients
    cbf._solve_parameterized_qp = qp_solve
    cp.Problem.solve = problem_solve
    conformal.make_cbf_filter = make_filter
    conformal.joint_conformal_radius = joint_radius

    experiment = f"worker_r{_tag(requested_radius)}_k1_{_tag(args.worker_k1)}"
    sys.argv = [
        "conformal.py",
        "--multi_experiment", args.multi_experiment,
        "--solo_experiment", args.solo_experiment,
        "--experiment_name", experiment,
        "--update-procedure", "robust",
        "--r_init", str(requested_radius),
        "--num_multi_agents", str(args.num_multi_agents),
        "--num_trajectories", str(args.num_trajectories),
        "--num_eval_trajs", "1",
        "--num_episodes", "1",
        "--episode_length", str(args.episode_length),
        "--alpha", "0.9",
        "--delta", "0.05",
        "--init_predictions", args.init_predictions,
        "--deterministic",
        "--seed", str(args.seed),
        "--device", args.device,
        "--train_dir", os.path.join(os.path.abspath(args.output_dir), "runs"),
        "--multi_train_dir", os.path.abspath(args.multi_train_dir),
        "--solo_train_dir", os.path.abspath(args.solo_train_dir),
        "--predictions_dir", os.path.abspath(args.predictions_dir),
    ]
    conformal.main()

    point_qps = [row for row in qp_records if np.isclose(row["radius"], requested_radius)]
    point_actions = [row for row in action_records if np.isclose(row["radius"], requested_radius)]
    scores = [run["score"] for run in calibration_runs]
    agent_quantiles = [
        float(np.quantile([run["agent_errors"][agent] for run in calibration_runs], args.quantile))
        for agent in range(args.num_multi_agents)
    ]
    slacks = [row["slack"] for row in point_qps if row["slack"] is not None]
    comparison_samples = {
        "position_deviation_vs_nominal": [],
        "heading_deviation_vs_nominal_deg": [],
        "separation_delta_vs_nominal": [],
        "goal_progress_delta_vs_nominal": [],
        "net_goal_progress_delta_vs_nominal": [],
        "path_efficiency_delta_vs_nominal": [],
    }
    if args.worker_nominal:
        if not args.nominal_trajectories:
            raise ValueError("Nominal worker requires --nominal-trajectories.")
        np.savez_compressed(
            args.nominal_trajectories,
            ego_positions=np.asarray(ego_trajectories, dtype=np.float32),
        )
    elif args.nominal_summary and args.nominal_trajectories:
        with open(args.nominal_summary, "r", encoding="utf-8") as handle:
            nominal_summary = json.load(handle)
        nominal_positions = np.load(args.nominal_trajectories)["ego_positions"]
        nominal_samples = nominal_summary["samples"]
        if len(nominal_positions) != len(ego_trajectories):
            raise ValueError("Nominal and filtered trajectory counts do not match.")
        for run_id, trajectory in enumerate(ego_trajectories):
            position_delta, heading_delta = _trajectory_deviation(
                trajectory,
                nominal_positions[run_id],
            )
            comparison_samples["position_deviation_vs_nominal"].append(position_delta)
            comparison_samples["heading_deviation_vs_nominal_deg"].append(heading_delta)
            comparison_samples["separation_delta_vs_nominal"].append(
                calibration_runs[run_id]["closest_ego_other"]
                - nominal_samples["closest_ego_other"][run_id]
            )
            for metric in (
                "goal_progress",
                "net_goal_progress",
                "path_efficiency",
            ):
                comparison_samples[f"{metric}_delta_vs_nominal"].append(
                    calibration_runs[run_id][metric]
                    - nominal_samples[metric][run_id]
                )

    metric_samples = {
        "joint_error": [run["joint_error"] for run in calibration_runs],
        "score": scores,
        "closest_ego_other": [run["closest_ego_other"] for run in calibration_runs],
        "action_intervention": [row["intervention"] for row in point_actions],
        "action_jump": [row["jump"] for row in point_actions if row["jump"] is not None],
        "slack": slacks,
        "min_predicted_rho": [row["min_rho"] for row in point_qps],
        "max_gradient_norm": [row["max_gradient"] for row in point_qps],
        "goal_progress": [run["goal_progress"] for run in calibration_runs],
        "net_goal_progress": [run["net_goal_progress"] for run in calibration_runs],
        "ego_path_length": [run["ego_path_length"] for run in calibration_runs],
        "final_goal_distance": [run["final_goal_distance"] for run in calibration_runs],
        "path_efficiency": [run["path_efficiency"] for run in calibration_runs],
        "agents_within_1p5": [run["agents_within_1p5"] for run in calibration_runs],
        "agents_within_2p0": [run["agents_within_2p0"] for run in calibration_runs],
        "simultaneous_agents_within_1p5": [run["simultaneous_agents_within_1p5"] for run in calibration_runs],
        "simultaneous_agents_within_2p0": [run["simultaneous_agents_within_2p0"] for run in calibration_runs],
        **comparison_samples,
    }
    score_interval = _quantile_confidence_interval(
        scores,
        args.quantile,
        args.confidence,
    )
    summary = {
        "schema_version": 2,
        "controller": "nominal" if args.worker_nominal else "ecbf",
        "radius": None if args.worker_nominal else requested_radius,
        "cbf_k1": None if args.worker_nominal else float(args.worker_k1),
        "seed": args.seed,
        "episode_length": args.episode_length,
        "confidence": args.confidence,
        "score_quantile_level": args.quantile,
        "score_quantile": float(np.quantile(scores, args.quantile)),
        "score_quantile_ci": score_interval,
        "num_trajectories": len(calibration_runs),
        "qp_calls": len(point_qps),
        "solver_counts": dict(Counter(row["solver"] for row in point_qps)),
        **{name: _stats(values) for name, values in metric_samples.items()},
        **{
            f"{name}_mean_ci": _mean_confidence_interval(values, args.confidence)
            for name, values in metric_samples.items()
            if values
        },
        "action_intervention_gt_0p1": (
            float(np.mean(np.asarray(metric_samples["action_intervention"]) > 0.1))
            if metric_samples["action_intervention"] else 0.0
        ),
        "slack_gt_1": float(np.mean(np.asarray(slacks) > 1.0)) if slacks else 0.0,
        "slack_gt_100": float(np.mean(np.asarray(slacks) > 100.0)) if slacks else 0.0,
        "per_agent_error_quantile": agent_quantiles,
        "dominant_error_agent": int(np.argmax(agent_quantiles)),
        "samples": {
            name: [float(value) for value in values]
            for name, values in metric_samples.items()
            if len(values) == len(calibration_runs)
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.worker_output)), exist_ok=True)
    with open(args.worker_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _goal_metrics(ego_log: dict, episode_length: int) -> dict[str, float]:
    goal_distance = np.asarray(ego_log["goal_dist"], dtype=np.float64)[:episode_length]
    goal_swap = np.asarray(ego_log["goal_swap"], dtype=bool)[:episode_length]
    positions = np.asarray(ego_log["position"], dtype=np.float64)[:episode_length]
    valid_steps = ~goal_swap[1:]
    positive_progress = np.maximum(goal_distance[:-1] - goal_distance[1:], 0.0)
    if not np.any(valid_steps):
        raise ValueError("Cannot measure goal progress when every step changes goals.")
    goal_progress = float(
        np.sum(positive_progress[valid_steps])
        / np.sum(valid_steps)
        * (episode_length - 1)
    )
    net_goal_progress = float(goal_distance[0] - goal_distance[-1])
    ego_path_length = float(
        np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    )
    return {
        "goal_progress": goal_progress,
        "net_goal_progress": net_goal_progress,
        "ego_path_length": ego_path_length,
        "final_goal_distance": float(goal_distance[-1]),
        "path_efficiency": (
            net_goal_progress / ego_path_length if ego_path_length > 0.0 else 0.0
        ),
    }


def _trajectory_deviation(
    trajectory: np.ndarray,
    nominal_trajectory: np.ndarray,
) -> tuple[float, float]:
    position_deviation = float(
        np.max(np.linalg.norm(trajectory - nominal_trajectory, axis=1))
    )
    velocity = np.diff(trajectory, axis=0)
    nominal_velocity = np.diff(nominal_trajectory, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    nominal_speed = np.linalg.norm(nominal_velocity, axis=1)
    valid = (speed > 1.0e-6) & (nominal_speed > 1.0e-6)
    if not np.any(valid):
        return position_deviation, 0.0
    cosine = np.sum(velocity[valid] * nominal_velocity[valid], axis=1) / (
        speed[valid] * nominal_speed[valid]
    )
    heading_deviation = float(
        np.max(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    )
    return position_deviation, heading_deviation


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
