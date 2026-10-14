#!/usr/bin/env python3
"""Run robust, naive, or calibrate-once conformal CBF experiments."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from sample_factory.model.model_utils import get_rnn_size

from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.conformal_utils import (
    calibrate_once_final_radius,
    conformal_radii,
    fall_down,
    get_alpha_bar,
    next_radius,
    run_multi_agents,
    summarize_rollouts,
)
from project_utils.full_cbf_utils import make_cbf_filter
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import latest_checkpoint, load_actor, load_cfg


DEVICE = torch.device("cpu")
MAX_RADIUS = 8.0
KAPPA = 0.6
COLLISION_HITBOX_RADIUS = 2.5
UPDATE_PROCEDURES = ("robust", "naive", "calibrate-once")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a conformal prediction experiment with the full ECBF."
    )
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", required=True)
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", default="train_dir")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--num_eval_trajs", type=int, default=100)
    parser.add_argument("--num_multi_agents", type=int, default=-1)
    parser.add_argument("--predictions_dir", default="train_dir")
    parser.add_argument("--init_predictions", default=None)
    parser.add_argument(
        "--update_predictions",
        action="store_true",
        help="Replace predictions with the first evaluation run after episode 0.",
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--update_procedure",
        "--update-procedure",
        choices=UPDATE_PROCEDURES,
        default="robust",
        help="Radius update rule. Defaults to the legacy conformal.py behavior.",
    )
    args = parser.parse_args(argv)

    if args.episode_length < 2:
        parser.error("--episode_length must be at least 2")
    if args.num_trajectories <= 0:
        parser.error("--num_trajectories must be positive")
    if args.num_eval_trajs <= 0:
        parser.error("--num_eval_trajs must be positive")
    if args.num_episodes < 0:
        parser.error("--num_episodes must be non-negative")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie in (0, 1)")
    if not 0.0 < args.delta < 1.0:
        parser.error("--delta must lie in (0, 1)")
    if args.update_procedure != "calibrate-once" and args.num_episodes == 0:
        parser.error("--num_episodes must be positive for iterative updates")
    if args.update_procedure == "calibrate-once" and args.update_predictions:
        parser.error("--update_predictions is not supported by calibrate-once")

    return args


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def _empty_metrics() -> dict:
    return {
        "qj_per_episode": [],
        "radius_per_episode": [],
        "tube_coverage_per_episode": [],
        "crashes_per_episode": [],
        "bad_crashes_per_episode": [],
        "safety_per_episode": [],
        "cumulative_reward_per_episode": [],
        "cumulative_reward_runs_per_episode": [],
        "agent_locs_per_episode": [],
        "predicted_traj_per_episode": [],
    }


def _record_episode(
    metrics: dict,
    qj: float,
    radius: float,
    summary: dict,
    pred_trajectories,
) -> None:
    metrics["qj_per_episode"].append(qj)
    metrics["radius_per_episode"].append(radius)
    metrics["tube_coverage_per_episode"].append(summary["tube_coverage"])
    metrics["crashes_per_episode"].append(summary["crashes"])
    metrics["bad_crashes_per_episode"].append(summary["bad_crashes"])
    metrics["safety_per_episode"].append(summary["safety"])
    metrics["cumulative_reward_per_episode"].append(
        summary["cumulative_reward"]
    )
    metrics["cumulative_reward_runs_per_episode"].append(
        summary["cumulative_reward_per_run"]
    )
    metrics["agent_locs_per_episode"].append(summary["agent_locs"])
    metrics["predicted_traj_per_episode"].append(
        np.asarray(pred_trajectories, dtype=np.float32).copy()
    )


def _save_metrics(
    experiment_dir: str,
    args: argparse.Namespace,
    episode_ids,
    metrics: dict,
    alpha_bar: float,
) -> str:
    metrics_path = os.path.join(experiment_dir, "conformal_metrics.npz")
    np.savez(
        metrics_path,
        episodes=np.asarray(episode_ids, dtype=np.int64),
        qj_per_episode=np.asarray(
            metrics["qj_per_episode"],
            dtype=np.float32,
        ),
        radius_per_episode=np.asarray(
            metrics["radius_per_episode"],
            dtype=np.float32,
        ),
        tube_coverage_per_episode=np.asarray(
            metrics["tube_coverage_per_episode"],
            dtype=np.float32,
        ),
        crashes_per_episode=np.asarray(
            metrics["crashes_per_episode"],
            dtype=np.float32,
        ),
        bad_crashes_per_episode=np.asarray(
            metrics["bad_crashes_per_episode"],
            dtype=np.float32,
        ),
        safety_per_episode=np.asarray(
            metrics["safety_per_episode"],
            dtype=np.float32,
        ),
        cumulative_reward_per_episode=np.asarray(
            metrics["cumulative_reward_per_episode"],
            dtype=np.float32,
        ),
        cumulative_reward_per_run=np.asarray(
            metrics["cumulative_reward_runs_per_episode"],
            dtype=np.float32,
        ),
        agent_locs_per_episode=np.asarray(
            metrics["agent_locs_per_episode"],
            dtype=np.float32,
        ),
        predicted_traj_per_episode=np.asarray(
            metrics["predicted_traj_per_episode"],
            dtype=np.float32,
        ),
        alpha=args.alpha,
        delta=args.delta,
        bar_alpha=alpha_bar,
        update_procedure=np.asarray(args.update_procedure),
    )
    return metrics_path


def _validate_predictions(
    pred_trajectories,
    num_multi_agents: int,
    episode_length: int,
):
    predictions = np.asarray(pred_trajectories)
    if predictions.ndim != 3 or predictions.shape[0] != num_multi_agents:
        raise ValueError(
            "Predictions must have shape (num_multi_agents, steps, features)."
        )
    if predictions.shape[1] < episode_length:
        raise ValueError(
            "Predicted trajectories are shorter than --episode_length."
        )
    if predictions.shape[2] < 6:
        raise ValueError(
            "Predicted trajectories must contain position and velocity."
        )
    return [predictions[i] for i in range(num_multi_agents)]


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    register_swarm_components()

    experiment_dir = ensure_experiment_dir(
        args.train_dir,
        args.experiment_name,
    )
    args_path = os.path.join(experiment_dir, "conformal_args.json")
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    if args.num_multi_agents < 0:
        args.num_multi_agents = int(cfg_multi.quads_num_agents)
    if args.num_multi_agents <= 0:
        raise ValueError("--num_multi_agents must be positive.")

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={args.num_multi_agents + 1}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_collision_reward=8.0",
        f"--quads_collision_hitbox_radius={COLLISION_HITBOX_RADIUS}",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=True",
        "--quads_view_mode=topdown",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    env = make_quadrotor_env(
        "quadrotor_multi",
        cfg=eval_cfg,
        render_mode="rgb_array",
    )

    try:
        multi_ckpt = latest_checkpoint(
            args.multi_train_dir,
            args.multi_experiment,
            policy_index=0,
        )
        multi_actor = load_actor(
            cfg_multi,
            env.observation_space,
            env.action_space,
            multi_ckpt,
            DEVICE,
        )
        multi_rnn_size = get_rnn_size(cfg_multi)

        cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
        solo_env = make_quadrotor_env(
            "quadrotor_multi",
            cfg=cfg_solo,
            render_mode=None,
        )
        try:
            solo_ckpt = latest_checkpoint(
                args.solo_train_dir,
                args.solo_experiment,
                policy_index=0,
            )
            solo_actor = load_actor(
                cfg_solo,
                solo_env.observation_space,
                solo_env.action_space,
                solo_ckpt,
                DEVICE,
            )
            solo_obs_dim = int(solo_env.observation_space.shape[0])
            solo_rnn_size = get_rnn_size(cfg_solo)
        finally:
            solo_env.close()

        _obs, stored_states = deterministic_reset(env, args.seed, None)

        def collect_rollouts(
            pred_trajectories,
            solo_action_fn,
            num_runs: int,
            deterministic: bool,
        ):
            multi_rnn_states = torch.zeros(
                (args.num_multi_agents, multi_rnn_size),
                dtype=torch.float32,
                device=DEVICE,
            )
            solo_rnn_states = torch.zeros(
                (1, solo_rnn_size),
                dtype=torch.float32,
                device=DEVICE,
            )
            obs, _ = deterministic_reset(env, args.seed, stored_states)
            snapshot = safe_capture_env_snapshot(env)
            temp_env = clone_env_from_snapshot(snapshot, restore_rng=True)
            try:
                return run_multi_agents(
                    temp_env,
                    obs,
                    args.num_multi_agents,
                    multi_actor,
                    multi_rnn_states,
                    solo_actor,
                    solo_rnn_states,
                    solo_obs_dim,
                    pred_trajectories,
                    solo_action_fn,
                    max_steps=args.episode_length,
                    num_runs=num_runs,
                    deterministic=deterministic,
                )
            finally:
                temp_env.close()

        if args.init_predictions is None:
            print("Predicting trajectories")
            dummy_predictions = np.zeros(
                (args.num_multi_agents, args.episode_length, 6),
                dtype=np.float64,
            )
            prediction_logs = collect_rollouts(
                dummy_predictions,
                fall_down,
                num_runs=1,
                deterministic=True,
            )
            pred_trajectories = []
            for agent_id in range(args.num_multi_agents):
                pred_trajectories.append(
                    np.concatenate(
                        [
                            prediction_logs[agent_id][0]["position"],
                            prediction_logs[agent_id][0]["velocity"],
                        ],
                        axis=1,
                    )
                )
        else:
            pred_path = os.path.join(
                args.predictions_dir,
                args.init_predictions,
                "pred_trajectories.npz",
            )
            with np.load(pred_path) as pred_data:
                pred_trajectories = pred_data["pred_trajectories"].astype(
                    np.float32
                )

        pred_trajectories = _validate_predictions(
            pred_trajectories,
            args.num_multi_agents,
            args.episode_length,
        )

        min_radius = env.quad_arm * COLLISION_HITBOX_RADIUS
        alpha_bar = get_alpha_bar(
            args.alpha,
            args.delta,
            args.num_trajectories,
        )
        if not 0.0 <= alpha_bar < 1.0:
            raise ValueError(
                "The adjusted alpha is outside [0, 1); increase "
                "--num_trajectories or adjust --alpha/--delta."
            )

        def collect_for_radius(radius: float, num_runs: int):
            radii = np.full(
                args.num_multi_agents,
                radius,
                dtype=np.float64,
            )
            return collect_rollouts(
                pred_trajectories,
                make_cbf_filter(radii),
                num_runs=num_runs,
                deterministic=args.deterministic,
            )

        def calibrate(radius: float) -> float:
            logs = collect_for_radius(radius, args.num_trajectories)
            radii = conformal_radii(
                logs,
                args.num_multi_agents,
                pred_trajectories,
                alpha_bar,
                args.episode_length,
            )
            return float(np.max(radii))

        metrics = _empty_metrics()

        def evaluate_and_record(radius: float, qj: float):
            logs = collect_for_radius(radius, args.num_eval_trajs)
            summary = summarize_rollouts(
                logs,
                pred_trajectories,
                radius,
                min_radius,
                args.num_multi_agents,
                args.episode_length,
                args.num_eval_trajs,
            )
            _record_episode(
                metrics,
                qj,
                radius,
                summary,
                pred_trajectories,
            )
            print(
                f"Cum rew: {summary['cumulative_reward']} "
                f"Tube cov: {summary['tube_coverage']} "
                f"Crashes: {summary['crashes']} "
                f"Bad crashes: {summary['bad_crashes']}"
            )
            return logs

        if args.update_procedure in ("robust", "naive"):
            radius = MAX_RADIUS
            episode_ids = np.arange(args.num_episodes)
            for episode in episode_ids:
                print("EPISODE", episode)
                qj = calibrate(radius)
                new_radius = next_radius(
                    args.update_procedure,
                    radius,
                    qj,
                    KAPPA,
                    min_radius,
                    MAX_RADIUS,
                )
                print(
                    "radius",
                    radius,
                    "qj",
                    qj,
                    "new radius",
                    new_radius,
                )
                radius = new_radius
                logs = evaluate_and_record(radius, qj)

                if args.update_predictions and episode == 0:
                    for agent_id in range(args.num_multi_agents):
                        pred_trajectories[agent_id][
                            : args.episode_length,
                            :3,
                        ] = logs[agent_id][0]["position"]
                        pred_trajectories[agent_id][
                            : args.episode_length,
                            3:6,
                        ] = logs[agent_id][0]["velocity"]
        else:
            q0 = calibrate(MAX_RADIUS)
            evaluate_and_record(MAX_RADIUS, q0)

            radius = q0
            print("r0", MAX_RADIUS, "q0", q0, "r1", radius)
            q1 = calibrate(radius)
            print("r1", radius, "q1", q1)
            evaluate_and_record(radius, q1)

            radius = calibrate_once_final_radius(
                MAX_RADIUS,
                q0,
                KAPPA,
                args.num_episodes,
            )
            q_final = calibrate(radius)
            print("radius", radius, "qj", q_final)
            evaluate_and_record(radius, q_final)
            episode_ids = np.array([0, 1, args.num_episodes + 1])

        metrics_path = _save_metrics(
            experiment_dir,
            args,
            episode_ids,
            metrics,
            alpha_bar,
        )
        print(f"[conformal] Saved per-episode metrics to {metrics_path}")
        print(
            "Number of total crashes:",
            sum(metrics["crashes_per_episode"]),
        )
        print(
            "Number of total crashes outside of conformal tubes:",
            sum(metrics["bad_crashes_per_episode"]),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
