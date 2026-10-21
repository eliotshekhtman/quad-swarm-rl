from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot

from project_utils.restart_utils import QuadState, extract_positions_velocities
from project_utils.utils import OBS_KEY


EVALUATION_SCENARIO = "go_to_goal"
PREDICTION_SCHEMA_VERSION = 1


def collect_homogeneous_trajectories(
    env,
    obs: np.ndarray,
    actor,
    rnn_states: torch.Tensor,
    max_steps: int,
) -> np.ndarray:
    """Collect one deterministic rollout with the same actor on every quad."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    obs_run = np.asarray(obs, dtype=np.float32).copy()
    if obs_run.shape[0] != rnn_states.shape[0]:
        raise ValueError("Observation and recurrent-state agent counts do not match.")

    snapshot = safe_capture_env_snapshot(env)
    env_run = clone_env_from_snapshot(snapshot, restore_rng=True)
    run_rnn_states = rnn_states.clone()
    positions = []
    velocities = []

    try:
        done = False
        while not done and len(positions) < max_steps:
            obs_dict = {OBS_KEY: obs_run}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(actor, obs_dict)
                policy_output = actor(normalized_obs, run_rnn_states)
            run_rnn_states = policy_output["new_rnn_states"]
            actions = argmax_actions(actor.action_distribution())
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)

            obs_run, _rewards, dones, _infos = env_run.step(
                actions.detach().cpu().numpy()
            )
            obs_run = np.asarray(obs_run, dtype=np.float32)
            pos, vel = extract_positions_velocities(env_run.unwrapped)
            positions.append(pos.copy())
            velocities.append(vel.copy())
            done = bool(np.all(dones))
    finally:
        env_run.close()

    if len(positions) != max_steps:
        raise RuntimeError(
            f"Prediction rollout ended after {len(positions)} of {max_steps} steps."
        )

    positions_arr = np.stack(positions, axis=1)
    velocities_arr = np.stack(velocities, axis=1)
    return np.concatenate([positions_arr, velocities_arr], axis=2).astype(
        np.float32
    )


def save_prediction_artifact(
    path: str,
    pred_trajectories: np.ndarray,
    all_trajectories: np.ndarray,
    initial_states: Sequence[QuadState],
    *,
    policy_experiment: str,
    policy_checkpoint: str,
    seed: int,
) -> None:
    predictions = np.asarray(pred_trajectories, dtype=np.float32)
    all_predictions = np.asarray(all_trajectories, dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[2] != 6:
        raise ValueError("pred_trajectories must have shape (agents, steps, 6).")
    if all_predictions.ndim != 3 or all_predictions.shape[2] != 6:
        raise ValueError("all_trajectories must have shape (agents, steps, 6).")
    if all_predictions.shape[0] != predictions.shape[0] + 1:
        raise ValueError("Homogeneous collection must include one additional ego quad.")
    if all_predictions.shape[1] != predictions.shape[1]:
        raise ValueError("Prediction horizons do not match.")
    if len(initial_states) != all_predictions.shape[0]:
        raise ValueError("Initial-state count does not match the policy-agent count.")
    if not np.isfinite(all_predictions).all():
        raise ValueError("Prediction trajectories contain non-finite values.")

    initial_positions = np.stack(
        [np.asarray(state.position, dtype=np.float64) for state in initial_states]
    )
    initial_goals = np.stack(
        [np.asarray(state.goal, dtype=np.float64) for state in initial_states]
    )
    np.savez_compressed(
        path,
        pred_trajectories=predictions,
        all_trajectories=all_predictions,
        initial_positions=initial_positions,
        initial_goals=initial_goals,
        schema_version=np.int64(PREDICTION_SCHEMA_VERSION),
        scenario=np.asarray(EVALUATION_SCENARIO),
        homogeneous=np.bool_(True),
        policy_experiment=np.asarray(policy_experiment),
        policy_checkpoint=np.asarray(os.path.basename(policy_checkpoint)),
        seed=np.int64(seed),
        num_multi_agents=np.int64(predictions.shape[0]),
        num_policy_agents=np.int64(all_predictions.shape[0]),
        episode_length=np.int64(predictions.shape[1]),
    )


def load_prediction_artifact(
    path: str,
    *,
    expected_num_multi_agents: int,
    expected_episode_length: int,
    expected_seed: int,
    expected_policy_experiment: str,
    expected_policy_checkpoint: str,
    expected_initial_states: Sequence[QuadState],
) -> np.ndarray:
    required = {
        "pred_trajectories",
        "all_trajectories",
        "initial_positions",
        "initial_goals",
        "schema_version",
        "scenario",
        "homogeneous",
        "policy_experiment",
        "policy_checkpoint",
        "seed",
        "num_multi_agents",
        "num_policy_agents",
        "episode_length",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                "Prediction artifact lacks go_to_goal provenance metadata: "
                + ", ".join(sorted(missing))
            )

        expected_scalars = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "scenario": EVALUATION_SCENARIO,
            "homogeneous": True,
            "policy_experiment": expected_policy_experiment,
            "policy_checkpoint": os.path.basename(expected_policy_checkpoint),
            "seed": expected_seed,
            "num_multi_agents": expected_num_multi_agents,
            "num_policy_agents": expected_num_multi_agents + 1,
        }
        for key, expected in expected_scalars.items():
            actual = data[key].item()
            if actual != expected:
                raise ValueError(
                    f"Prediction artifact {key} mismatch: expected {expected!r}, "
                    f"found {actual!r}."
                )

        artifact_horizon = int(data["episode_length"].item())
        if artifact_horizon < expected_episode_length:
            raise ValueError(
                "Prediction artifact is shorter than the requested episode length."
            )

        predictions = data["pred_trajectories"].astype(np.float32)
        all_predictions = data["all_trajectories"].astype(np.float32)
        initial_positions = data["initial_positions"].astype(np.float64)
        initial_goals = data["initial_goals"].astype(np.float64)

    expected_positions = np.stack(
        [np.asarray(state.position, dtype=np.float64) for state in expected_initial_states]
    )
    expected_goals = np.stack(
        [np.asarray(state.goal, dtype=np.float64) for state in expected_initial_states]
    )
    if not np.array_equal(initial_positions, expected_positions):
        raise ValueError("Prediction artifact initial positions do not match the environment.")
    if not np.array_equal(initial_goals, expected_goals):
        raise ValueError("Prediction artifact initial goals do not match the environment.")

    expected_prediction_shape = (
        expected_num_multi_agents,
        artifact_horizon,
        6,
    )
    expected_all_shape = (
        expected_num_multi_agents + 1,
        artifact_horizon,
        6,
    )
    if predictions.shape != expected_prediction_shape:
        raise ValueError(
            f"Prediction artifact has shape {predictions.shape}, expected "
            f"{expected_prediction_shape}."
        )
    if all_predictions.shape != expected_all_shape:
        raise ValueError(
            f"All-agent trajectories have shape {all_predictions.shape}, expected "
            f"{expected_all_shape}."
        )
    if not np.array_equal(predictions, all_predictions[:-1]):
        raise ValueError(
            "Stored non-ego predictions do not match the homogeneous rollout."
        )
    if not np.isfinite(all_predictions).all():
        raise ValueError("Prediction artifact contains non-finite values.")
    return predictions


__all__ = [
    "EVALUATION_SCENARIO",
    "collect_homogeneous_trajectories",
    "load_prediction_artifact",
    "save_prediction_artifact",
]
