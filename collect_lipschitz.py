#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size

from project_utils.cbf_utils import cbf_dynamics, real_dynamics
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import OBS_KEY, latest_checkpoint, load_actor, load_cfg
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

DEVICE = torch.device("cpu")
EPS = 1e-6

def _pack_state(position: np.ndarray, velocity: np.ndarray, rotation: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Flatten one simulator state into a fixed 18D vector:
    [pos(3), vel(3), rot(9), omega(3)].

    Keeping a single packed layout makes pairwise distance computations simple and fast.
    """
    return np.concatenate(
        [
            np.asarray(position, dtype=np.float64).reshape(-1),
            np.asarray(velocity, dtype=np.float64).reshape(-1),
            np.asarray(rotation, dtype=np.float64).reshape(-1),
            np.asarray(omega, dtype=np.float64).reshape(-1),
        ],
        axis=0,
    )


def _pack_next_state_tuple(next_state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Convert dynamics output tuple (pos, vel, rot, omega) into the same 18D format."""
    pos, vel, rot, omega = next_state
    return _pack_state(pos, vel, rot, omega)


def _action_to_unit(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Map an action from env bounds [low, high] to normalized [0, 1]^d coordinates.

    All Lipschitz constants in this script are computed in this normalized action space.
    """
    scale = np.maximum(high - low, 1e-12)
    return np.clip((action - low) / scale, 0.0, 1.0)


def _set_single_agent_start_goal(env, start_point: np.ndarray, goal_point: np.ndarray) -> np.ndarray:
    """
    Hard-reset the single quad's physical state and active goal.

    We enforce deterministic start/goal per trajectory even though the scenario
    may generate random patrol goals internally.
    """
    base_env = env.unwrapped
    quad = base_env.envs[0]
    dynamics = quad.dynamics

    velocity = np.zeros(3, dtype=np.float64)
    omega = np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)

    dynamics.set_state(start_point, velocity, rotation, omega)
    dynamics.reset()
    dynamics.on_floor = False
    dynamics.crashed_floor = dynamics.crashed_wall = dynamics.crashed_ceiling = False

    quad.goal = goal_point.copy()
    quad.tick = 0
    quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]

    # Keep patrol_dual_goal scenario aligned with the manually assigned goal.
    scenario = getattr(base_env, "scenario", None)
    if scenario is not None and hasattr(scenario, "goal_pairs"):
        if scenario.goal_pairs is not None:
            scenario.goal_pairs[0, 0] = goal_point.copy()
            scenario.goal_pairs[0, 1] = goal_point.copy()
        if getattr(scenario, "active_goal_index", None) is not None:
            scenario.active_goal_index[0] = 0
        if getattr(scenario, "steps_since_switch", None) is not None:
            scenario.steps_since_switch[0] = 0
        if getattr(scenario, "goals", None) is not None:
            scenario.goals[0] = goal_point.copy()

    base_env.pos[0, :] = dynamics.pos
    base_env.vel[0, :] = dynamics.vel

    obs = [quad.state_vector(quad)]
    if getattr(base_env, "num_use_neighbor_obs", 0) > 0:
        obs = base_env.add_neighborhood_obs(obs)
    return np.asarray(obs, dtype=np.float32)


def _compute_pairwise_action_lipschitz(
    actions_unit: np.ndarray,
    real_next: np.ndarray,
    residual_next: np.ndarray,
    ratios_u: np.ndarray,
    ratios_eu: np.ndarray,
    write_u: int,
    write_eu: int,
) -> Tuple[int, int]:
    """
    For one fixed state, compute action-based ratios across all unordered action pairs:

    L_u contribution: ||f_real(x,u_i)-f_real(x,u_j)|| / ||u_i-u_j||
    L_eu contribution: ||e(x,u_i)-e(x,u_j)|| / ||u_i-u_j||

    Results are appended into preallocated buffers for efficiency.
    """
    pair_i, pair_j = np.triu_indices(actions_unit.shape[0], k=1)

    delta_u = np.linalg.norm(actions_unit[pair_i] - actions_unit[pair_j], axis=1)
    valid = delta_u >= EPS
    if not np.any(valid):
        return write_u, write_eu

    pair_i = pair_i[valid]
    pair_j = pair_j[valid]
    delta_u = delta_u[valid]

    delta_real = np.linalg.norm(real_next[pair_i] - real_next[pair_j], axis=1)
    delta_resid = np.linalg.norm(residual_next[pair_i] - residual_next[pair_j], axis=1)

    count = delta_u.shape[0]
    ratios_u[write_u : write_u + count] = (delta_real / delta_u).astype(np.float32)
    ratios_eu[write_eu : write_eu + count] = (delta_resid / delta_u).astype(np.float32)
    return write_u + count, write_eu + count


def _stats_from_ratios(values: np.ndarray) -> Dict[str, float]:
    """Summarize a ratio set with max and robust high quantiles."""
    if values.size == 0:
        return {
            "max": 0.0,
            "top95_max": 0.0,
            "top90_max": 0.0,
            "num_ratios": 0,
        }
    return {
        "max": float(np.max(values)),
        "top95_max": float(np.quantile(values, 0.95)),
        "top90_max": float(np.quantile(values, 0.90)),
        "num_ratios": int(values.size),
    }


def _build_state_pairs(
    traj_ids: np.ndarray,
    step_ids: np.ndarray,
    same_traj_window: int,
    cross_traj_window: int,
    cross_samples_per_step: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build candidate state index pairs (i, j) for state-based Lipschitz estimates.

    Pairing policy:
    - Same trajectory: only local neighbors within `same_traj_window`.
    - Cross trajectory: only near-in-time neighbors within `cross_traj_window`,
      optionally subsampled per source state using `cross_samples_per_step`.
    - Global cap `max_pairs` to prevent quadratic explosion.
    """
    pair_i = []
    pair_j = []

    n = traj_ids.shape[0]

    # Same-trajectory local temporal pairs: encourages comparisons among similar phases.
    if same_traj_window > 0:
        unique_trajs = np.unique(traj_ids)
        for traj in unique_trajs:
            idx = np.where(traj_ids == traj)[0]
            if idx.size <= 1:
                continue
            for local_i in range(idx.size):
                local_j_end = min(idx.size, local_i + same_traj_window + 1)
                if local_j_end <= local_i + 1:
                    continue
                i_global = idx[local_i]
                js = idx[local_i + 1 : local_j_end]
                pair_i.extend([i_global] * js.size)
                pair_j.extend(js.tolist())

    # Cross-trajectory pairs constrained by step proximity.
    if cross_traj_window > 0 and cross_samples_per_step > 0:
        step_map = {}
        for idx in range(n):
            step_map.setdefault(int(step_ids[idx]), []).append(idx)

        for i in range(n):
            this_traj = traj_ids[i]
            this_step = int(step_ids[i])
            candidates = []
            for ds in range(-cross_traj_window, cross_traj_window + 1):
                step_key = this_step + ds
                for j in step_map.get(step_key, []):
                    if j <= i:
                        continue
                    if traj_ids[j] == this_traj:
                        continue
                    candidates.append(j)
            if not candidates:
                continue
            if len(candidates) > cross_samples_per_step:
                sel = rng.choice(len(candidates), size=cross_samples_per_step, replace=False)
                candidates = [candidates[k] for k in sel]
            pair_i.extend([i] * len(candidates))
            pair_j.extend(candidates)

    if len(pair_i) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)

    if pair_i.size > max_pairs:
        keep = rng.choice(pair_i.size, size=max_pairs, replace=False)
        pair_i = pair_i[keep]
        pair_j = pair_j[keep]

    return pair_i, pair_j


def _compute_state_based_lipschitz(
    states: np.ndarray,
    traj_ids: np.ndarray,
    step_ids: np.ndarray,
    real_next_random: np.ndarray,
    residual_random: np.ndarray,
    policy_actions: np.ndarray,
    unperturbed_mask: np.ndarray,
    same_traj_window: int,
    cross_traj_window: int,
    cross_samples_per_step: int,
    max_pairs: int,
    state_close_radius: float,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, float]]:
    """
    Compute state-conditioned Lipschitz constants:
    - L_x: real dynamics sensitivity to state (fixed action).
    - L_ex: residual sensitivity to state (fixed action).
    - L_pi: policy sensitivity to state (unperturbed rollouts only).
    """
    i_all, j_all = _build_state_pairs(
        traj_ids=traj_ids,
        step_ids=step_ids,
        same_traj_window=same_traj_window,
        cross_traj_window=cross_traj_window,
        cross_samples_per_step=cross_samples_per_step,
        max_pairs=max_pairs,
        rng=rng,
    )

    if i_all.size == 0:
        zero = _stats_from_ratios(np.empty(0, dtype=np.float32))
        return {"L_x": zero, "L_ex": zero, "L_pi": zero}

    # Denominator for state-based constants.
    dx = np.linalg.norm(states[i_all] - states[j_all], axis=1)
    valid = dx >= EPS
    if state_close_radius > 0:
        valid &= dx <= state_close_radius

    i_all = i_all[valid]
    j_all = j_all[valid]
    dx = dx[valid]

    if i_all.size == 0:
        zero = _stats_from_ratios(np.empty(0, dtype=np.float32))
        return {"L_x": zero, "L_ex": zero, "L_pi": zero}

    ratios_x = []
    ratios_ex = []

    # Same-action comparisons use the global shared random probe actions.
    # This guarantees action overlap across different states.
    for action_idx in tqdm(
        range(real_next_random.shape[0]),
        desc="L_x/L_ex over probe actions",
        leave=False,
    ):
        dy_real = np.linalg.norm(real_next_random[action_idx, i_all] - real_next_random[action_idx, j_all], axis=1)
        dy_res = np.linalg.norm(residual_random[action_idx, i_all] - residual_random[action_idx, j_all], axis=1)
        ratios_x.append((dy_real / dx).astype(np.float32))
        ratios_ex.append((dy_res / dx).astype(np.float32))

    ratios_x = np.concatenate(ratios_x, axis=0) if ratios_x else np.empty(0, dtype=np.float32)
    ratios_ex = np.concatenate(ratios_ex, axis=0) if ratios_ex else np.empty(0, dtype=np.float32)

    # L_pi is measured only on states from base (non-perturbed) rollouts.
    idx_unpert = np.where(unperturbed_mask)[0]
    if idx_unpert.size > 1:
        i_u, j_u = _build_state_pairs(
            traj_ids=traj_ids[idx_unpert],
            step_ids=step_ids[idx_unpert],
            same_traj_window=same_traj_window,
            cross_traj_window=cross_traj_window,
            cross_samples_per_step=cross_samples_per_step,
            max_pairs=max_pairs,
            rng=rng,
        )
        i_u = idx_unpert[i_u]
        j_u = idx_unpert[j_u]
        dx_pi = np.linalg.norm(states[i_u] - states[j_u], axis=1)
        valid_pi = dx_pi >= EPS
        if state_close_radius > 0:
            valid_pi &= dx_pi <= state_close_radius
        i_u = i_u[valid_pi]
        j_u = j_u[valid_pi]
        dx_pi = dx_pi[valid_pi]
        if i_u.size > 0:
            du = np.linalg.norm(policy_actions[i_u] - policy_actions[j_u], axis=1)
            ratios_pi = (du / dx_pi).astype(np.float32)
        else:
            ratios_pi = np.empty(0, dtype=np.float32)
    else:
        ratios_pi = np.empty(0, dtype=np.float32)

    return {
        "L_x": _stats_from_ratios(ratios_x),
        "L_ex": _stats_from_ratios(ratios_ex),
        "L_pi": _stats_from_ratios(ratios_pi),
    }


@dataclass
class CollectionResult:
    """Paths + metrics returned by `collect_lipschitz`."""
    metrics: Dict[str, Dict[str, float]]
    dataset_path: str
    metrics_path: str


def collect_lipschitz(
    perturbation_scale: float,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    *,
    solo_train_dir: str,
    solo_experiment: str,
    episode_length: int,
    seed: int,
    same_traj_window: int,
    cross_traj_window: int,
    cross_samples_per_step: int,
    max_state_pairs: int,
    state_close_radius: float,
    output_dir: str,
) -> CollectionResult:
    """
    Execute rollouts, evaluate next-state models, and estimate empirical constants.

    High-level flow:
    1. Build single-agent env + load solo policy.
    2. Run 200 trajectories (100 unperturbed, 100 perturbed).
    3. At each timestep, evaluate real/CBF dynamics on:
       - executed action
       - 10 fixed random probe actions
    4. Aggregate ratio statistics for L_u, L_x, L_eu, L_ex, L_pi.
    """
    if not (0.0 <= perturbation_scale <= 1.0):
        raise ValueError("perturbation_scale must be in [0, 1]")

    register_swarm_components()

    cfg_solo = load_cfg(solo_train_dir, solo_experiment)

    # Match conformal-style evaluation setup with patrol_dual_goal, but single agent only.
    # We pick episode duration long enough to cover the requested fixed horizon.
    control_dt_guess = 0.015
    required_seconds = max(15.0, episode_length * control_dt_guess + 1.0)
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        "--quads_num_agents=1",
        f"--quads_neighbor_visible_num={cfg_solo.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_solo.quads_neighbor_obs_type}",
        "--quads_use_numba=False",
        "--quads_render=False",
        "--max_num_episodes=1",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)

    try:
        solo_ckpt = latest_checkpoint(solo_train_dir, solo_experiment, policy_index=0)
        solo_actor = load_actor(cfg_solo, env.observation_space, env.action_space, solo_ckpt, DEVICE)
        solo_rnn_size = get_rnn_size(cfg_solo)

        # Initial reset/snapshot to establish deterministic baseline state buffers.
        obs, stored_states = deterministic_reset(env, seed, None)
        obs = _set_single_agent_start_goal(env, start_point, goal_point)

        action_low = np.asarray(env.action_space.low, dtype=np.float64)
        action_high = np.asarray(env.action_space.high, dtype=np.float64)
        action_dim = int(action_low.shape[-1])

        rng = np.random.default_rng(seed)

        # Pre-sample fixed random probe actions once so all states share identical probes.
        random_actions_raw = rng.uniform(low=action_low, high=action_high, size=(10, action_dim)).astype(np.float64)
        random_actions_unit = _action_to_unit(random_actions_raw, action_low, action_high)

        num_trajectories = 200
        num_total_samples = num_trajectories * episode_length

        # Main trajectory-level buffers. These are preallocated to avoid repeated reallocations.
        states = np.empty((num_total_samples, 18), dtype=np.float32)
        traj_ids = np.empty(num_total_samples, dtype=np.int32)
        step_ids = np.empty(num_total_samples, dtype=np.int32)
        policy_actions_unit = np.empty((num_total_samples, action_dim), dtype=np.float32)
        actual_actions_unit = np.empty((num_total_samples, action_dim), dtype=np.float32)

        real_next_actual = np.empty((num_total_samples, 18), dtype=np.float32)
        cbf_next_actual = np.empty((num_total_samples, 18), dtype=np.float32)
        real_next_random = np.empty((10, num_total_samples, 18), dtype=np.float32)
        cbf_next_random = np.empty((10, num_total_samples, 18), dtype=np.float32)

        max_ratios_per_state = 11 * 10 // 2
        # Upper bound on pair count per state for 11 actions is C(11,2)=55.
        ratios_u = np.empty(num_total_samples * max_ratios_per_state, dtype=np.float32)
        ratios_eu = np.empty(num_total_samples * max_ratios_per_state, dtype=np.float32)
        write_u = 0
        write_eu = 0

        sample_idx = 0
        collect_pbar = tqdm(total=num_trajectories, desc="Collecting trajectories", unit="traj")

        for traj in range(num_trajectories):
            # Deterministic per-trajectory reset then explicit override to requested start/goal.
            obs, stored_states = deterministic_reset(env, seed + traj, stored_states)
            obs = _set_single_agent_start_goal(env, start_point, goal_point)

            run_rnn = torch.zeros((1, solo_rnn_size), dtype=torch.float32, device=DEVICE)
            is_perturbed_rollout = traj >= 100

            for step in range(episode_length):
                obs_np = np.asarray(obs, dtype=np.float32)
                obs_self = obs_np[0]
                obs_dict = {OBS_KEY: obs_self[None, :]}

                # Policy inference for solo action.
                with torch.no_grad():
                    normalized = prepare_and_normalize_obs(solo_actor, obs_dict)
                    policy_output = solo_actor(normalized, run_rnn)
                run_rnn = policy_output["new_rnn_states"]

                base_action = argmax_actions(solo_actor.action_distribution())
                if base_action.dim() == 1:
                    base_action = base_action.unsqueeze(0)
                base_action_raw = base_action.detach().cpu().numpy()[0].astype(np.float64)

                if is_perturbed_rollout:
                    # Perturbed half: additive bounded uniform noise before clipping to env bounds.
                    noise = rng.uniform(low=-1.0, high=1.0, size=action_dim)
                    actual_action_raw = np.clip(base_action_raw + perturbation_scale * noise, action_low, action_high)
                else:
                    actual_action_raw = base_action_raw

                base_action_unit = _action_to_unit(base_action_raw, action_low, action_high)
                actual_action_unit = _action_to_unit(actual_action_raw, action_low, action_high)

                # Snapshot current simulator state x_t before stepping.
                dynamics = env.unwrapped.envs[0].dynamics
                current_state = _pack_state(dynamics.pos, dynamics.vel, dynamics.rot, dynamics.omega)

                # Probe set = {executed action} U {10 shared random actions}.
                action_bank_unit = np.vstack([actual_action_unit[None, :], random_actions_unit])
                real_bank = np.empty((11, 18), dtype=np.float64)
                cbf_bank = np.empty((11, 18), dtype=np.float64)
                for a_idx in range(11):
                    cmd_unit = action_bank_unit[a_idx]
                    real_next = real_dynamics(cmd_unit, dynamics, env.unwrapped.control_dt)
                    cbf_next = cbf_dynamics(cmd_unit, dynamics, env.unwrapped.control_dt)
                    real_bank[a_idx] = _pack_next_state_tuple(real_next)
                    cbf_bank[a_idx] = _pack_next_state_tuple(cbf_next)

                # Residual dynamics e(x,u) = real(x,u) - cbf(x,u).
                residual_bank = real_bank - cbf_bank
                write_u, write_eu = _compute_pairwise_action_lipschitz(
                    actions_unit=action_bank_unit,
                    real_next=real_bank,
                    residual_next=residual_bank,
                    ratios_u=ratios_u,
                    ratios_eu=ratios_eu,
                    write_u=write_u,
                    write_eu=write_eu,
                )

                states[sample_idx] = current_state.astype(np.float32)
                traj_ids[sample_idx] = traj
                step_ids[sample_idx] = step
                policy_actions_unit[sample_idx] = base_action_unit.astype(np.float32)
                actual_actions_unit[sample_idx] = actual_action_unit.astype(np.float32)

                real_next_actual[sample_idx] = real_bank[0].astype(np.float32)
                cbf_next_actual[sample_idx] = cbf_bank[0].astype(np.float32)
                real_next_random[:, sample_idx, :] = real_bank[1:].astype(np.float32)
                cbf_next_random[:, sample_idx, :] = cbf_bank[1:].astype(np.float32)

                # Execute one real simulator step with the selected action.
                obs, _, _, _, _ = env.step(actual_action_raw[None, :].astype(np.float32))
                sample_idx += 1
            collect_pbar.update(1)
        collect_pbar.close()

        # Trim unused preallocated ratio tail.
        ratios_u = ratios_u[:write_u]
        ratios_eu = ratios_eu[:write_eu]

        # Residuals for random probe actions used by L_ex computation.
        residual_random = real_next_random - cbf_next_random
        unperturbed_mask = traj_ids < 100
        lipschitz_pbar = tqdm(total=3, desc="Computing Lipschitz constants", unit="phase")
        lipschitz_pbar.update(1)

        state_metrics = _compute_state_based_lipschitz(
            states=states,
            traj_ids=traj_ids,
            step_ids=step_ids,
            real_next_random=real_next_random,
            residual_random=residual_random,
            policy_actions=policy_actions_unit,
            unperturbed_mask=unperturbed_mask,
            same_traj_window=same_traj_window,
            cross_traj_window=cross_traj_window,
            cross_samples_per_step=cross_samples_per_step,
            max_pairs=max_state_pairs,
            state_close_radius=state_close_radius,
            rng=rng,
        )
        lipschitz_pbar.update(1)

        # Final scalar summaries.
        metrics = {
            "config": {
                "perturbation_scale": float(perturbation_scale),
                "start_point": start_point.tolist(),
                "goal_point": goal_point.tolist(),
                "episode_length": int(episode_length),
                "num_trajectories": 200,
                "num_unperturbed": 100,
                "num_perturbed": 100,
                "same_traj_window": int(same_traj_window),
                "cross_traj_window": int(cross_traj_window),
                "cross_samples_per_step": int(cross_samples_per_step),
                "max_state_pairs": int(max_state_pairs),
                "state_close_radius": float(state_close_radius),
                "action_space": "[0, 1]^4 (normalized)",
            },
            "L_u": _stats_from_ratios(ratios_u),
            "L_eu": _stats_from_ratios(ratios_eu),
            "L_x": state_metrics["L_x"],
            "L_ex": state_metrics["L_ex"],
            "L_pi": state_metrics["L_pi"],
        }
        lipschitz_pbar.update(1)
        lipschitz_pbar.close()

        os.makedirs(output_dir, exist_ok=True)
        dataset_path = os.path.join(output_dir, "lipschitz_dataset.npz")
        metrics_path = os.path.join(output_dir, "lipschitz_metrics.json")

        # Save full arrays for post-hoc diagnostics / alternative estimators.
        np.savez_compressed(
            dataset_path,
            states=states,
            traj_ids=traj_ids,
            step_ids=step_ids,
            policy_actions_unit=policy_actions_unit,
            actual_actions_unit=actual_actions_unit,
            random_actions_raw=random_actions_raw.astype(np.float32),
            random_actions_unit=random_actions_unit.astype(np.float32),
            real_next_actual=real_next_actual,
            cbf_next_actual=cbf_next_actual,
            real_next_random=real_next_random,
            cbf_next_random=cbf_next_random,
        )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

        return CollectionResult(metrics=metrics, dataset_path=dataset_path, metrics_path=metrics_path)

    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    """CLI options for reproducible rollout collection and estimator controls."""
    parser = argparse.ArgumentParser(description="Collect rollout state-action data and estimate empirical Lipschitz constants.")
    parser.add_argument("--solo_train_dir", default="train_dir", help="Directory containing solo policy experiment.")
    parser.add_argument("--solo_experiment", required=True, help="Solo policy experiment name.")
    parser.add_argument("--perturbation_scale", type=float, required=True, help="Perturbation scale in [0, 1].")
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="train_dir/lipschitz_collection")

    # State-pair search controls for L_x, L_ex, and L_pi.
    parser.add_argument("--same_traj_window", type=int, default=8)
    parser.add_argument("--cross_traj_window", type=int, default=0)
    parser.add_argument("--cross_samples_per_step", type=int, default=2)
    parser.add_argument("--max_state_pairs", type=int, default=500000)
    parser.add_argument("--state_close_radius", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Entry point with default start/goal values requested by the user."""
    args = parse_args()

    # Requested defaults for patrol_dual_goal-like 3D points.
    start_point = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    goal_point = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    result = collect_lipschitz(
        perturbation_scale=args.perturbation_scale,
        start_point=start_point,
        goal_point=goal_point,
        solo_train_dir=args.solo_train_dir,
        solo_experiment=args.solo_experiment,
        episode_length=args.episode_length,
        seed=args.seed,
        same_traj_window=args.same_traj_window,
        cross_traj_window=args.cross_traj_window,
        cross_samples_per_step=args.cross_samples_per_step,
        max_state_pairs=args.max_state_pairs,
        state_close_radius=args.state_close_radius,
        output_dir=args.output_dir,
    )

    print("Saved dataset:", result.dataset_path)
    print("Saved metrics:", result.metrics_path)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
