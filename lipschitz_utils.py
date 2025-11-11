from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from swarm_rl.env_snapshot import (
    clone_env_from_snapshot,
    restore_rng_state,
    safe_capture_env_snapshot,
    snapshot_rng_state,
)

from cbf_utils import CBF_K0, CBF_K1
from utils import OBS_KEY, get_swarm_state


DELTA_T = 0.015
MAX_RADIUS = 2.0

# ---------------------------------------------------------------------------
# Estimate Lipschitz constants
# ---------------------------------------------------------------------------

def _extract_agent_state(env, agent_index=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (position, velocity, rotation_matrix, angular_velocity) for the chosen agent."""
    base_env = getattr(env, "unwrapped", env)
    quad = base_env.envs[agent_index]
    dynamics = quad.dynamics
    position = np.asarray(dynamics.pos, dtype=np.float64)
    velocity = np.asarray(dynamics.vel, dtype=np.float64)
    rotation = np.asarray(dynamics.rot, dtype=np.float64)
    omega = np.asarray(dynamics.omega, dtype=np.float64)
    return position, velocity, rotation, omega

def _pack_agent_state(position, velocity, rotation, omega) -> np.ndarray:
    """Flatten the provided components into a single vector for distance computations."""
    return np.concatenate(
        [
            position.reshape(-1),
            velocity.reshape(-1),
            rotation.reshape(-1),
            omega.reshape(-1),
        ],
        axis=0,
    ).astype(np.float64)

def _project_to_so3(rotation: np.ndarray) -> np.ndarray:
    """Project an arbitrary 3x3 matrix onto SO(3) via SVD."""
    rot_mat = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(rot_mat)
    proj = U @ Vt
    if np.linalg.det(proj) < 0.0:
        U[:, -1] *= -1.0
        proj = U @ Vt
    return proj

def _unpack_agent_state(state_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of `_pack_agent_state`."""
    flat = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    if flat.size != 18:
        raise ValueError(f"Expected 18-dimensional state vector, received {flat.size}")
    position = flat[0:3]
    velocity = flat[3:6]
    rotation = flat[6:15].reshape(3, 3)
    omega = flat[15:18]
    return position, velocity, rotation, omega

def _apply_agent_state(env, agent_index: int, position, velocity, rotation, omega) -> None:
    """Write the supplied state into the chosen agent and update cached arrays."""
    base_env = getattr(env, "unwrapped", env)
    quad = base_env.envs[agent_index]
    # Assume that the rotation is valid
    quad.dynamics.set_state(position, velocity, rotation, omega)
    base_env.pos[agent_index, :] = quad.dynamics.pos
    base_env.vel[agent_index, :] = quad.dynamics.vel

def _collect_observations_from_env(env) -> np.ndarray:
    """Rebuild the per-agent observations from the environment's current physical state."""
    base_env = getattr(env, "unwrapped", env)
    obs = [quad.state_vector(quad) for quad in base_env.envs]
    if getattr(base_env, "num_use_neighbor_obs", 0) > 0:
        obs = base_env.add_neighborhood_obs(obs)
    return np.asarray(obs, dtype=np.float32)

def _extract_full_swarm_state_vector(env, num_multi_agents: int) -> np.ndarray:
    """
    Flatten (pos, vel, rot, omega) for every multi-agent quad into a single vector.
    Assumes the solo agent occupies the last slot and is therefore excluded.
    """
    base_env = getattr(env, "unwrapped", env)
    if num_multi_agents <= 0:
        return np.empty(0, dtype=np.float64)
    packed = []
    for agent_idx in range(num_multi_agents):
        quad = base_env.envs[agent_idx]
        dynamics = quad.dynamics
        packed.append(
            _pack_agent_state(
                np.asarray(dynamics.pos, dtype=np.float64),
                np.asarray(dynamics.vel, dtype=np.float64),
                np.asarray(dynamics.rot, dtype=np.float64),
                np.asarray(dynamics.omega, dtype=np.float64),
            )
        )
    return np.concatenate(packed, axis=0)

def _apply_full_swarm_state(env, state_vec: np.ndarray, num_multi_agents: int) -> None:
    """
    Overwrite each multi-agent quad's state using the flattened vector representation.
    Leaves the solo agent (last index) untouched.
    """
    if num_multi_agents <= 0:
        return
    base_env = getattr(env, "unwrapped", env)
    state_array = np.asarray(state_vec, dtype=np.float64)
    try:
        per_agent = state_array.reshape(num_multi_agents, 18)
    except ValueError as exc:
        raise ValueError("State vector length must be num_multi_agents * 18") from exc
    for agent_idx in range(num_multi_agents):
        entry = per_agent[agent_idx]
        position = entry[0:3]
        velocity = entry[3:6]
        rotation = entry[6:15].reshape(3, 3)
        omega = entry[15:18]
        _apply_agent_state(base_env, agent_idx, position, velocity, rotation, omega)

def simulate_env_from_snapshot(
    snapshot,
    base_env_state_vec: np.ndarray,
    base_solo_state_vec: np.ndarray,
    env_state_perturb: Optional[np.ndarray],
    solo_state_perturb: Optional[np.ndarray],
    num_multi_agents: int,
    multi_actor,
    multi_rnn_states,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single environment step from a snapshot while applying optional perturbations.
    Returns (multi_agents_state_vec, solo_state_vec) after stepping.
    """
    rng_backup = snapshot_rng_state()
    restore_rng_state(snapshot.rng)
    env_clone = clone_env_from_snapshot(snapshot)
    try:
        solo_reference_vec = base_solo_state_vec
        if env_state_perturb is not None:
            candidate_env_state = base_env_state_vec + env_state_perturb
            _apply_full_swarm_state(env_clone, candidate_env_state, num_multi_agents)
            solo_reference_vec = _pack_agent_state(*_extract_agent_state(env_clone, -1))
        if solo_state_perturb is not None:
            candidate_solo = solo_reference_vec + solo_state_perturb
            position, velocity, rotation, omega = _unpack_agent_state(candidate_solo)
            _apply_agent_state(env_clone, -1, position, velocity, rotation, omega)

        obs_clone = _collect_observations_from_env(env_clone)

        run_multi_states = multi_rnn_states.clone()
        obs_multi_dict = {OBS_KEY: obs_clone[:num_multi_agents]}
        with torch.no_grad():
            normalized_multi = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
            policy_multi = multi_actor(normalized_multi, run_multi_states)
        actions_multi = argmax_actions(multi_actor.action_distribution())
        if actions_multi.dim() == 1:
            actions_multi = actions_multi.unsqueeze(-1)
        actions_multi = actions_multi.detach().cpu().numpy()

        run_solo_states = solo_rnn_states.clone()
        solo_obs = obs_clone[-1, :solo_obs_dim]
        obs_solo_dict = {OBS_KEY: solo_obs[None, :]}
        with torch.no_grad():
            normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
            solo_actor(normalized_solo, run_solo_states)
        action_solo = argmax_actions(solo_actor.action_distribution())
        if action_solo.dim() == 1:
            action_solo = action_solo.unsqueeze(0)
        action_solo = action_solo.detach().cpu().numpy()[0]

        actions = np.vstack([actions_multi, action_solo[None, :]])
        env_clone.step(actions)
        swarm_state_vec = _extract_full_swarm_state_vector(env_clone, num_multi_agents)
        solo_state_vec = _pack_agent_state(*_extract_agent_state(env_clone, -1))
        return swarm_state_vec, solo_state_vec
    finally:
        env_clone.close()
        restore_rng_state(rng_backup)

def estimate_LXu(
    temp_env,
    obs,
    num_multi_agents,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim,
) -> float:
    """
    Empirically estimate a local Lipschitz constant L such that
    ||x_{t+1}(u_1) - x_{t+1}(u_2)|| <= L ||u_1 - u_2|| for the solo agent.
    """


    obs_np = np.asarray(obs, dtype=np.float32)
    snapshot = safe_capture_env_snapshot(temp_env)

    # Run solo agent to get the base action / state
    run_solo_rnn_states = solo_rnn_states.clone()
    obs_solo_self = obs_np[-1, :solo_obs_dim]
    obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
    with torch.no_grad():
        normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
        solo_actor(normalized_solo, run_solo_rnn_states)
    base_action = argmax_actions(solo_actor.action_distribution())
    if base_action.dim() == 1:
        base_action = base_action.unsqueeze(0)
    base_action = base_action.detach().cpu().numpy()[0].astype(np.float32)
    action_dim = int(base_action.shape[-1])

    # zero thrust for teammates
    actions_multi = np.zeros((num_multi_agents, action_dim), dtype=np.float32)

    def simulate_next_state(action: np.ndarray) -> np.ndarray:
        rng_backup = snapshot_rng_state()
        restore_rng_state(snapshot.rng)
        env_clone = clone_env_from_snapshot(snapshot)
        try:
            stacked_actions = np.vstack([actions_multi, action[None, :]])
            env_clone.step(stacked_actions)
            position, velocity, rotation, omega = _extract_agent_state(env_clone.unwrapped, -1)
            return _pack_agent_state(position, velocity, rotation, omega)
        finally:
            env_clone.close()
            restore_rng_state(rng_backup)

    base_state = simulate_next_state(base_action)

    # Collect directions to act in
    required_samples = max(32, 3 * action_dim)
    directions: List[np.ndarray] = []
    for axis in range(action_dim):
        basis = np.zeros(action_dim, dtype=np.float32)
        basis[axis] = 1.0
        directions.append(basis)
        directions.append(-basis)
    while len(directions) < required_samples:
        directions.append(np.random.uniform(low=-1., high=1., size=action_dim).astype(np.float32))

    # Printing what happens with NO perturbation
    # pert_state = simulate_next_state(base_action)
    # print("DELTA X: ", np.linalg.norm(pert_state - base_state))

    # Collect max ratio
    lipschitz = 0.0
    for direction in directions:
        delta_u = np.linalg.norm(direction - base_action)
        if delta_u < 1e-9:
            continue
        pert_state = simulate_next_state(direction)
        delta_x = np.linalg.norm(pert_state - base_state)
        if delta_x <= 0.0:
            continue
        lipschitz = max(lipschitz, delta_x / delta_u)

    return float(lipschitz)


def estimate_LXx(
    temp_env,
    obs,
    num_multi_agents,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim,
    perturbation_radius,
) -> float:
    """
    Estimate a local Lipschitz constant relating solo-state perturbations to the
    next solo state.
    """
    if perturbation_radius <= 0.0:
        return 0.0

    obs_np = np.asarray(obs, dtype=np.float32)
    snapshot = safe_capture_env_snapshot(temp_env)
    solo_position, solo_velocity, solo_rotation, solo_omega = _extract_agent_state(temp_env.unwrapped, -1)
    base_state_vec = _pack_agent_state(solo_position, solo_velocity, solo_rotation, solo_omega)
    state_dim = base_state_vec.size

    run_solo_rnn_states = solo_rnn_states.clone()
    obs_solo_self = obs_np[-1, :solo_obs_dim]
    obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
    with torch.no_grad():
        normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
        solo_actor(normalized_solo, run_solo_rnn_states)
    base_action = argmax_actions(solo_actor.action_distribution())
    if base_action.dim() == 1:
        base_action = base_action.unsqueeze(0)
    base_action = base_action.detach().cpu().numpy()[0]
    action_dim = base_action.shape[-1]
    actions_multi = np.zeros((num_multi_agents, action_dim), dtype=np.float32)

    def simulate_next_state(state_vec: np.ndarray) -> np.ndarray:
        rng_backup = snapshot_rng_state()
        restore_rng_state(snapshot.rng)
        env_clone = clone_env_from_snapshot(snapshot)
        try:
            pos, vel, rot, omega = _unpack_agent_state(state_vec)
            _apply_agent_state(env_clone, -1, pos, vel, rot, omega)

            stacked_actions = np.vstack([actions_multi, base_action[None, :]])
            env_clone.step(stacked_actions)
            next_pos, next_vel, next_rot, next_omega = _extract_agent_state(env_clone.unwrapped, -1)
            return _pack_agent_state(next_pos, next_vel, next_rot, next_omega)
        finally:
            env_clone.close()
            restore_rng_state(rng_backup)

    base_next_state = simulate_next_state(base_state_vec)

    required_samples = max(32, 3 * state_dim)
    directions: List[np.ndarray] = []
    for axis in range(state_dim):
        basis = np.zeros(state_dim, dtype=np.float64)
        basis[axis] = 1.0
        directions.append(basis)
        directions.append(-basis)
    while len(directions) < required_samples:
        directions.append(np.random.uniform(low=-1.0, high=1.0, size=state_dim))

    lipschitz = 0.0
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        perturb = (perturbation_radius * direction / norm).astype(np.float64)
        candidate_state = base_state_vec + perturb
        pos, vel, rot, omega = _unpack_agent_state(candidate_state)
        rot = _project_to_so3(rot)
        candidate_state = _pack_agent_state(pos, vel, rot, omega)
        perturb = candidate_state - base_state_vec
        delta_x = np.linalg.norm(perturb)
        if delta_x < 1e-9:
            continue
        pert_state = simulate_next_state(candidate_state)
        delta_y = np.linalg.norm(pert_state - base_next_state)
        if delta_y <= 0.0:
            continue
        lipschitz = max(lipschitz, delta_y / delta_x)

    return float(lipschitz)

def estimate_LYx(
    temp_env,
    obs,
    num_multi_agents,
    multi_actor,
    multi_rnn_states,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim,
    perturbation_radius,
) -> float:
    """
    Estimate a local Lipschitz constant relating perturbations in the solo quad's
    physical state to the next environment-wide state (all teammates included).
    """
    if perturbation_radius <= 0.0:
        return 0.0

    snapshot = safe_capture_env_snapshot(temp_env)
    solo_position, solo_velocity, solo_rotation, solo_omega = _extract_agent_state(temp_env.unwrapped, -1)
    base_state_vec = _pack_agent_state(solo_position, solo_velocity, solo_rotation, solo_omega)
    base_env_state_vec = _extract_full_swarm_state_vector(temp_env.unwrapped, num_multi_agents)
    # Size of total solo agent state
    state_dim = base_state_vec.size

    # The total concatenated environment state after a timestep
    base_next_state, _ = simulate_env_from_snapshot(
        snapshot,
        base_env_state_vec,
        base_state_vec,
        env_state_perturb=None,
        solo_state_perturb=None,
        num_multi_agents=num_multi_agents,
        multi_actor=multi_actor,
        multi_rnn_states=multi_rnn_states,
        solo_actor=solo_actor,
        solo_rnn_states=solo_rnn_states,
        solo_obs_dim=solo_obs_dim,
    )

    required_samples = max(32, 3 * state_dim)
    directions: List[np.ndarray] = []
    for axis in range(state_dim):
        basis = np.zeros(state_dim, dtype=np.float64)
        basis[axis] = 1.0
        directions.append(basis)
        directions.append(-basis)
    while len(directions) < required_samples:
        directions.append(np.random.uniform(low=-1., high=1., size=state_dim))

    # Printing what happens with NO perturbation
    # perturb = directions[0] * 0.0
    # perturbed_state, _ = simulate_env_from_snapshot(
    #     snapshot,
    #     base_env_state_vec,
    #     base_state_vec,
    #     env_state_perturb=None,
    #     solo_state_perturb=perturb,
    #     num_multi_agents=num_multi_agents,
    #     multi_actor=multi_actor,
    #     multi_rnn_states=multi_rnn_states,
    #     solo_actor=solo_actor,
    #     solo_rnn_states=solo_rnn_states,
    #     solo_obs_dim=solo_obs_dim,
    # )
    # delta_y = np.linalg.norm(perturbed_state - base_next_state)
    # print("DELTA Y: ", delta_y)

    lipschitz = 0.0
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        perturb = (perturbation_radius * direction / norm).astype(np.float64)
        # Make it so that the perturbation itself respects SO(3)
        candidate = base_state_vec + perturb
        position, velocity, rotation, omega = _unpack_agent_state(candidate)
        rotation = _project_to_so3(rotation)
        candidate = _pack_agent_state(position, velocity, rotation, omega)
        perturb = candidate - base_state_vec
        delta_x = np.linalg.norm(perturb)
        if delta_x < 1e-9:
            continue
        # Run environment one step and collect the new state
        perturbed_state, _ = simulate_env_from_snapshot(
            snapshot,
            base_env_state_vec,
            base_state_vec,
            env_state_perturb=None,
            solo_state_perturb=perturb,
            num_multi_agents=num_multi_agents,
            multi_actor=multi_actor,
        multi_rnn_states=multi_rnn_states,
        solo_actor=solo_actor,
        solo_rnn_states=solo_rnn_states,
        solo_obs_dim=solo_obs_dim,
        )
        delta_y = np.linalg.norm(perturbed_state - base_next_state)
        if delta_y <= 0.0:
            continue
        # print(delta_y, delta_x)
        lipschitz = max(lipschitz, delta_y / delta_x)

    return float(lipschitz)


def estimate_LYu(
    temp_env,
    obs,
    num_multi_agents,
    multi_actor,
    multi_rnn_states,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim,
    perturbation_radius,
) -> float:
    """
    Estimate a local Lipschitz constant relating solo-action perturbations to the
    next teammate state (solo quad excluded).
    """
    if perturbation_radius <= 0.0 or num_multi_agents <= 0:
        return 0.0

    obs_np = np.asarray(obs, dtype=np.float32)
    snapshot = safe_capture_env_snapshot(temp_env)

    run_solo_rnn_states = solo_rnn_states.clone()
    obs_solo_self = obs_np[-1, :solo_obs_dim]
    obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
    with torch.no_grad():
        normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
        solo_actor(normalized_solo, run_solo_rnn_states)
    base_action = argmax_actions(solo_actor.action_distribution())
    if base_action.dim() == 1:
        base_action = base_action.unsqueeze(0)
    base_action = base_action.detach().cpu().numpy()[0].astype(np.float32)
    action_dim = int(base_action.shape[-1])

    def simulate_teammate_state(action: np.ndarray) -> np.ndarray:
        rng_backup = snapshot_rng_state()
        restore_rng_state(snapshot.rng)
        env_clone = clone_env_from_snapshot(snapshot)
        try:
            obs_clone = _collect_observations_from_env(env_clone)

            run_multi_states = multi_rnn_states.clone()
            obs_multi_dict = {OBS_KEY: obs_clone[:num_multi_agents]}
            with torch.no_grad():
                normalized_multi = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_multi = multi_actor(normalized_multi, run_multi_states)
            actions_multi = policy_multi["actions"]
            actions_multi = argmax_actions(multi_actor.action_distribution())
            if actions_multi.dim() == 1:
                actions_multi = actions_multi.unsqueeze(-1)
            actions_multi = actions_multi.detach().cpu().numpy()

            stacked_actions = np.vstack([actions_multi, action[None, :]])
            env_clone.step(stacked_actions)
            return _extract_full_swarm_state_vector(env_clone.unwrapped, num_multi_agents)
        finally:
            env_clone.close()
            restore_rng_state(rng_backup)

    base_env_state = simulate_teammate_state(base_action)

    required_samples = max(32, 3 * action_dim)
    directions: List[np.ndarray] = []
    for axis in range(action_dim):
        basis = np.zeros(action_dim, dtype=np.float32)
        basis[axis] = 1.0
        directions.append(basis)
        directions.append(-basis)
    while len(directions) < required_samples:
        directions.append(np.random.uniform(low=-1.0, high=1.0, size=action_dim).astype(np.float32))
    
    # Printing what happens with NO perturbation
    pert_env_state = simulate_teammate_state(base_action)
    print("DELTA Y: ", np.linalg.norm(pert_env_state - base_env_state))

    lipschitz = 0.0
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        perturb = (perturbation_radius * direction / norm).astype(np.float32)
        candidate_action = np.clip(base_action + perturb, -1.0, 1.0)
        delta_u = np.linalg.norm(candidate_action - base_action)
        if delta_u < 1e-9:
            continue
        pert_state = simulate_teammate_state(candidate_action)
        delta_y = np.linalg.norm(pert_state - base_env_state)
        if delta_y <= 0.0:
            continue
        lipschitz = max(lipschitz, delta_y / delta_u)

    return float(lipschitz)


def estimate_LYy(
    temp_env,
    obs,
    num_multi_agents,
    multi_actor,
    multi_rnn_states,
    solo_actor,
    solo_rnn_states,
    solo_obs_dim,
    perturbation_radius,
) -> float:
    """
    Estimate a local Lipschitz constant relating teammate-state perturbations to
    the subsequent teammate state.
    """
    if perturbation_radius <= 0.0 or num_multi_agents <= 0:
        return 0.0

    snapshot = safe_capture_env_snapshot(temp_env)
    base_env_state_vec = _extract_full_swarm_state_vector(temp_env.unwrapped, num_multi_agents)
    solo_position, solo_velocity, solo_rotation, solo_omega = _extract_agent_state(temp_env.unwrapped, -1)
    base_solo_state_vec = _pack_agent_state(solo_position, solo_velocity, solo_rotation, solo_omega)
    state_dim = base_env_state_vec.size

    base_next_state, _ = simulate_env_from_snapshot(
        snapshot,
        base_env_state_vec,
        base_solo_state_vec,
        env_state_perturb=None,
        solo_state_perturb=None,
        num_multi_agents=num_multi_agents,
        multi_actor=multi_actor,
        multi_rnn_states=multi_rnn_states,
        solo_actor=solo_actor,
        solo_rnn_states=solo_rnn_states,
        solo_obs_dim=solo_obs_dim,
    )

    required_samples = max(32, 3 * state_dim)
    directions: List[np.ndarray] = []
    for axis in range(state_dim):
        basis = np.zeros(state_dim, dtype=np.float64)
        basis[axis] = 1.0
        directions.append(basis)
        directions.append(-basis)
    while len(directions) < required_samples:
        directions.append(np.random.uniform(low=-1.0, high=1.0, size=state_dim))

    lipschitz = 0.0
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        raw_perturb = (perturbation_radius * direction / norm).astype(np.float64)
        candidate_state = base_env_state_vec + raw_perturb
        candidate_state = candidate_state.reshape(num_multi_agents, 18)
        projected = []
        for agent_state in candidate_state:
            pos = agent_state[0:3]
            vel = agent_state[3:6]
            rot = agent_state[6:15].reshape(3, 3)
            omega = agent_state[15:18]
            rot = _project_to_so3(rot)
            projected.append(_pack_agent_state(pos, vel, rot, omega))
        candidate_state = np.concatenate(projected, axis=0)
        perturb = candidate_state - base_env_state_vec
        delta_x = np.linalg.norm(perturb)
        if delta_x < 1e-9:
            continue
        perturbed_state, _ = simulate_env_from_snapshot(
            snapshot,
            base_env_state_vec,
            base_solo_state_vec,
            env_state_perturb=perturb,
            solo_state_perturb=None,
            num_multi_agents=num_multi_agents,
            multi_actor=multi_actor,
            multi_rnn_states=multi_rnn_states,
            solo_actor=solo_actor,
            solo_rnn_states=solo_rnn_states,
            solo_obs_dim=solo_obs_dim,
        )
        delta_y = np.linalg.norm(perturbed_state - base_next_state)
        if delta_y <= 0.0:
            continue
        lipschitz = max(lipschitz, delta_y / delta_x)

    return float(lipschitz)

def closed_form_estimate_LXx(temp_env):
    dynamics = temp_env.envs[-1].dynamics
    solo_position, solo_velocity, solo_rotation, solo_omega = _extract_agent_state(temp_env.unwrapped, -1)
    cond_number = np.linalg.cond(dynamics.model.I_com, 2)
    omega_norm = np.linalg.norm(solo_omega)
    max_thrust = np.linalg.norm(dynamics.thrust_max)
    L_x = 1
    L_v = np.sqrt(1 + DELTA_T ** 2)
    L_R = np.sqrt((DELTA_T / dynamics.mass * max_thrust) ** 2 + (1 + DELTA_T * omega_norm) ** 2)
    L_omega = np.sqrt(DELTA_T ** 2 + (1 + 2 * DELTA_T * cond_number * omega_norm) ** 2)
    return 2 * max([L_x, L_v, L_R, L_omega])

def estimate_LU(temp_env, num_multi_agents):
    dynamics = temp_env.envs[-1].dynamics
    solo_position, solo_velocity, solo_rotation, solo_omega = _extract_agent_state(temp_env.unwrapped, -1)
    swarm_state = get_swarm_state(temp_env.unwrapped)
    positions = swarm_state.positions[:-1] # Multi agent positions
    a_k = []
    mass_factor = 2.0 / float(dynamics.mass)
    for agent_id in range(num_multi_agents):
        difference = solo_position - positions[agent_id]
        a = mass_factor * np.dot(solo_rotation[:, 2], difference) * np.ones(3)
        a_k.append(np.linalg.norm(a)) # From ||a_const * 1||_2
    return 2 * MAX_RADIUS * CBF_K0 * CBF_K1 / min(a_k)

def calculate_beta_T(L_Xx, L_Xu, L_Yy, L_Yx, L_Yu, episode_length):
    A_T = L_Xu * sum([L_Xx ** t for t in range(episode_length)])
    expsum_Yy = sum([L_Yy ** t for t in range(episode_length)])
    return (L_Yx * A_T + L_Yu) * expsum_Yy
