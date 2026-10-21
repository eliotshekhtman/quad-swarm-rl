from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from swarm_rl.env_snapshot import clone_env_from_snapshot, safe_capture_env_snapshot

from project_utils.restart_utils import extract_positions_velocities
from project_utils.utils import OBS_KEY, get_swarm_state


def get_alpha_bar(alpha: float, delta: float, num_trajectories: int) -> float:
    return alpha - np.sqrt(np.log(1.0 / delta) / (2.0 * num_trajectories))


def joint_rollout_error(
    logs,
    pred_trajectories,
    num_multi_agents: int,
    episode_length: int,
    run_id: int = 0,
) -> float:
    """Return the maximum position error over all environment agents and time."""
    max_error = 0.0
    for agent_id in range(num_multi_agents):
        actual = np.asarray(logs[agent_id][run_id]["position"])
        predicted = np.asarray(pred_trajectories[agent_id])
        if actual.shape[0] < episode_length or predicted.shape[0] < episode_length:
            raise ValueError("Rollout or prediction is shorter than episode_length.")
        errors = np.linalg.norm(
            actual[:episode_length, :3] - predicted[:episode_length, :3],
            axis=1,
        )
        max_error = max(max_error, float(np.max(errors)))
    return max_error


def joint_conformal_radius(
    logs,
    num_multi_agents: int,
    pred_trajectories,
    alpha: float,
    episode_length: int,
    min_radius: float,
) -> float:
    """Calibrate the deployed radius from one joint score per rollout."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1).")

    num_runs = len(logs[0])
    if num_runs == 0:
        raise ValueError("At least one rollout is required for calibration.")

    scores = sorted(
        min_radius
        + joint_rollout_error(
            logs,
            pred_trajectories,
            num_multi_agents,
            episode_length,
            run_id,
        )
        for run_id in range(num_runs)
    )
    quantile_index = int(np.ceil(num_runs * (1.0 - alpha)) - 1)
    return float(scores[quantile_index])


def explicit_radius_update(
    prev_radius: float,
    conf_radius: float,
    kappa: float,
    min_radius: float,
    max_radius: float,
) -> float:
    if conf_radius <= prev_radius:
        radius = (conf_radius + kappa * prev_radius) / (1.0 + kappa)
    else:
        radius = (conf_radius - kappa * prev_radius) / (1.0 - kappa)
    return float(np.clip(radius, min_radius, max_radius))


def next_radius(
    update_procedure: str,
    prev_radius: float,
    conf_radius: float,
    kappa: float,
    min_radius: float,
    max_radius: float,
) -> float:
    if update_procedure == "robust":
        return explicit_radius_update(
            prev_radius,
            conf_radius,
            kappa,
            min_radius,
            max_radius,
        )
    if update_procedure == "naive":
        return float(np.clip(conf_radius, min_radius, max_radius))
    raise ValueError(f"Unsupported iterative update procedure: {update_procedure}")


def calibrate_once_final_radius(
    initial_radius: float,
    calibrated_radius: float,
    kappa: float,
    num_updates: int,
) -> float:
    if num_updates < 0:
        raise ValueError("num_updates must be non-negative.")

    radius = float(calibrated_radius)
    delta_radius = abs(radius - initial_radius)
    for _ in range(num_updates):
        new_radius = calibrated_radius + kappa * delta_radius
        delta_radius = abs(new_radius - radius)
        radius = new_radius
    return float(radius)


def summarize_rollouts(
    logs,
    pred_trajectories,
    radius: float,
    min_radius: float,
    num_multi_agents: int,
    episode_length: int,
    num_runs: int,
) -> dict:
    """Summarize rollouts for a radius that includes physical separation."""
    cumulative_reward_per_run = []
    tube_coverage_per_run = []
    num_crashes_per_run = []
    num_bad_crashes_per_run = []
    solo_agent_id = num_multi_agents
    tube_radius = max(radius - min_radius, 0.0)

    for run_id in range(num_runs):
        num_nonswap_steps = 0
        cumulative_reward = 0.0
        in_tube = [True] * num_multi_agents
        num_crashes = 0
        num_bad_crashes = 0

        for agent_id in range(num_multi_agents):
            agent_loc = logs[agent_id][run_id]["position"][0]
            pred_loc = pred_trajectories[agent_id][0][:3]
            if np.linalg.norm(agent_loc - pred_loc) > tube_radius:
                in_tube[agent_id] = False

        for step in range(1, episode_length):
            solo_log = logs[solo_agent_id][run_id]
            if not solo_log["goal_swap"][step]:
                num_nonswap_steps += 1
                delta_distance = (
                    solo_log["goal_dist"][step - 1]
                    - solo_log["goal_dist"][step]
                )
                if delta_distance > 0:
                    cumulative_reward += delta_distance

            solo_loc = solo_log["position"][step]
            for agent_id in range(num_multi_agents):
                agent_loc = logs[agent_id][run_id]["position"][step]
                pred_loc = pred_trajectories[agent_id][step][:3]
                if np.linalg.norm(agent_loc - pred_loc) > tube_radius:
                    in_tube[agent_id] = False
                if np.linalg.norm(solo_loc - agent_loc) <= min_radius:
                    num_crashes += 1
                    if not in_tube[agent_id]:
                        num_bad_crashes += 1

        if num_nonswap_steps == 0:
            raise ValueError("Cannot compute progress when every evaluation step swaps goals.")

        cumulative_reward_per_run.append(
            cumulative_reward / num_nonswap_steps * (episode_length - 1)
        )
        tube_coverage_per_run.append(float(all(in_tube)))
        num_crashes_per_run.append(num_crashes)
        num_bad_crashes_per_run.append(num_bad_crashes)

    return {
        "cumulative_reward": sum(cumulative_reward_per_run) / num_runs,
        "tube_coverage": sum(tube_coverage_per_run) / num_runs,
        "crashes": sum(num_crashes_per_run) / num_runs,
        "bad_crashes": sum(num_bad_crashes_per_run) / num_runs,
        "safety": 1.0
        - sum(crashes > 0 for crashes in num_bad_crashes_per_run) / num_runs,
        "cumulative_reward_per_run": np.asarray(
            cumulative_reward_per_run,
            dtype=np.float32,
        ),
        "agent_locs": [
            logs[agent_id][0]["position"]
            for agent_id in range(num_multi_agents + 1)
        ],
    }


def fall_down(base_action, env_state, swarm_state):
    del base_action, env_state, swarm_state
    return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)


def _agent_goals(env):
    return [
        np.asarray(quad.goal, dtype=np.float64).copy()
        for quad in env.unwrapped.envs
    ]


def run_multi_agents(
    env,
    obs,
    num_multi_agents,
    multi_actor,
    multi_rnn_states,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim,
    pred_trajectories,
    solo_action_fn: Callable,
    max_steps=1500,
    num_runs=1,
    deterministic=False,
    first_run_deterministic=True,
    show_progress=True,
):
    """Run ordered rollouts from one snapshot and return state and goal logs."""
    snapshot = safe_capture_env_snapshot(env)
    logs = {i: [None] * num_runs for i in range(num_multi_agents + 1)}

    def _run_single_episode(run_idx: int):
        run_logs = {
            agent_id: {
                "position": [],
                "velocity": [],
                "goal_dist": [],
                "goal_swap": [],
            }
            for agent_id in range(num_multi_agents + 1)
        }
        env_run = clone_env_from_snapshot(snapshot)
        try:
            obs_run = np.array(obs, copy=True, dtype=np.float32)
            done = False
            step_num = 0
            run_multi_rnn_states = multi_rnn_states.clone()
            run_solo_rnn_states = solo_rnn_states.clone()
            run_max_dist = 0.0

            goals = _agent_goals(env_run)

            while not done and step_num < max_steps:
                obs_multi_dict = {OBS_KEY: obs_run[:num_multi_agents]}
                with torch.no_grad():
                    normalized_obs = prepare_and_normalize_obs(
                        multi_actor,
                        obs_multi_dict,
                    )
                    policy_output = multi_actor(
                        normalized_obs,
                        run_multi_rnn_states,
                    )
                actions_multi = policy_output["actions"]
                run_multi_rnn_states = policy_output["new_rnn_states"]
                if deterministic or (first_run_deterministic and run_idx == 0):
                    actions_multi = argmax_actions(
                        multi_actor.action_distribution()
                    )
                if actions_multi.dim() == 1:
                    actions_multi = actions_multi.unsqueeze(-1)
                actions_multi = actions_multi.detach().cpu().numpy()

                obs_solo_self = obs_run[-1, :solo_obs_dim]
                obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
                with torch.no_grad():
                    normalized_solo = prepare_and_normalize_obs(
                        solo_actor,
                        obs_solo_dict,
                    )
                    policy_solo = solo_actor(
                        normalized_solo,
                        run_solo_rnn_states,
                    )
                run_solo_rnn_states = policy_solo["new_rnn_states"]
                action_solo = argmax_actions(solo_actor.action_distribution())
                if action_solo.dim() == 1:
                    action_solo = action_solo.unsqueeze(0)
                action_solo = action_solo.detach().cpu().numpy()[0]

                swarm_state = get_swarm_state(env_run.unwrapped)
                for agent_id in range(num_multi_agents):
                    swarm_state.positions[agent_id, :] = (
                        pred_trajectories[agent_id][step_num][:3]
                    )
                    swarm_state.velocities[agent_id, :] = (
                        pred_trajectories[agent_id][step_num][3:]
                    )
                action_solo = solo_action_fn(
                    base_action=action_solo,
                    env_state=env_run.unwrapped,
                    swarm_state=swarm_state,
                )

                actions = np.vstack([actions_multi, action_solo[None, :]])
                obs_run, _rewards, dones, _infos = env_run.step(actions)
                obs_run = np.array(obs_run, dtype=np.float32)

                pos, vel = extract_positions_velocities(env_run.unwrapped)
                current_goals = _agent_goals(env_run)
                for agent_id in range(num_multi_agents + 1):
                    target = current_goals[agent_id]
                    run_logs[agent_id]["position"].append(pos[agent_id])
                    run_logs[agent_id]["velocity"].append(vel[agent_id])
                    run_logs[agent_id]["goal_dist"].append(
                        np.linalg.norm(pos[agent_id] - target)
                    )
                    goal_changed = not np.allclose(target, goals[agent_id])
                    run_logs[agent_id]["goal_swap"].append(goal_changed)
                    if goal_changed:
                        goals[agent_id] = target.copy()
                    if agent_id < num_multi_agents:
                        run_max_dist = max(
                            run_max_dist,
                            np.linalg.norm(
                                pos[agent_id]
                                - swarm_state.positions[agent_id, :]
                            ),
                        )
                done = np.all(dones)
                step_num += 1
        finally:
            env_run.close()

        return run_idx, run_logs, run_max_dist

    max_dist = 0.0
    progress_bar = tqdm(total=num_runs, disable=not show_progress)
    for run_idx in range(num_runs):
        run_idx, run_logs, run_max_dist = _run_single_episode(run_idx)
        for agent_id in range(num_multi_agents + 1):
            logs[agent_id][run_idx] = run_logs[agent_id]
        max_dist = max(max_dist, run_max_dist)
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"max dist={max_dist:.3f}")
    progress_bar.close()

    return logs


__all__ = [
    "calibrate_once_final_radius",
    "explicit_radius_update",
    "fall_down",
    "get_alpha_bar",
    "joint_conformal_radius",
    "joint_rollout_error",
    "next_radius",
    "run_multi_agents",
    "summarize_rollouts",
]
