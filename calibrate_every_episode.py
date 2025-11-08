#!/usr/bin/env python3
"""
Collect quadrotor trajectories in the patrol_dual_goal scenario with deterministic resets.

This script mirrors the evaluation flow of ``mixed_enjoy.py`` while enforcing:
    * A user-specified random seed that is re-applied before every environment reset so the
      patrol scenario regenerates the same goals each episode.
    * Manual restoration of the initial quad states (position, velocity, rotation, body rates)
      after every reset, ensuring agents respawn exactly where they started on the first episode.
    * Per-timestep logging of each quad's position and velocity to an ``NPZ`` archive.
    * Loading of a pretrained recurrent predictor (unused for control, but instantiated and ready).

Initial states detected on the first episode are persisted to ``initial_states.json`` inside the
output experiment directory so subsequent runs can reproduce the exact spawn configuration.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from tqdm import tqdm

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.huggingface.huggingface_utils import generate_replay_video

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_snapshot import *

from pretrain_rnn_predictor import RNNPredictor, load_rnn_checkpoint

from utils import *
from cbf_utils import make_cbf_filter
from restart_utils import (
    QuadState,
    deterministic_reset,
    extract_positions_velocities,
    quad_state_from_dict,
    quad_state_to_serialisable,
)

DEVICE = torch.device("cpu")
DELTA_T = 0.015


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


# ---------------------------------------------------------------------------
# Conformal utilities
# ---------------------------------------------------------------------------

def safe_capture_env_snapshot(env):
    """
    Capture a snapshot without copying OpenGL scene objects that own module references.
    """
    base_env = env.unwrapped
    had_scenes = hasattr(base_env, "scenes")
    saved_scenes = base_env.scenes if had_scenes else None
    if had_scenes:
        base_env.scenes = []
    try:
        return capture_env_snapshot(env)
    finally:
        if had_scenes:
            base_env.scenes = saved_scenes

def fall_down(base_action, env_state, swarm_state):
    return np.array([-1., -1., -1., -1.], dtype=np.float64)

def run_multi_agents(env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories, solo_action_fn,
                    max_steps=1600, num_runs=1, deterministic=False):
    '''
    Run the environment for [max_steps] steps, where the multi agents act like normal
    but the solo agent plays a fixed action, and return positions and velocities.
    Does not log the initial state.
    '''
    snapshot = safe_capture_env_snapshot(env)

    logs = {}
    for i in range(num_multi_agents):
        logs[i] = []
    for run in range(num_runs):
        # Start logging the new run
        for i in range(num_multi_agents):
            logs[i].append({ "position" : [], "velocity" : [] })
        # Make sure I spin up a new env and obs
        env_run = clone_env_from_snapshot(snapshot)
        obs_run = np.array(obs, copy=True, dtype=np.float32)
        done = False
        step_num = 0
        run_multi_rnn_states = multi_rnn_states.clone()
        run_solo_rnn_states = solo_rnn_states.clone()
        while not done and step_num < max_steps:
            obs_multi_dict = {OBS_KEY: obs_run[:num_multi_agents]}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_output = multi_actor(normalized_obs, run_multi_rnn_states)
            actions_multi = policy_output["actions"]
            run_multi_rnn_states = policy_output["new_rnn_states"]
            if deterministic:
                actions_multi = argmax_actions(multi_actor.action_distribution())
            if actions_multi.dim() == 1:
                actions_multi = actions_multi.unsqueeze(-1)
            actions_multi = actions_multi.detach().cpu().numpy()

            obs_solo_self = obs_run[-1, :solo_obs_dim]
            obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
            with torch.no_grad():
                normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
                policy_solo = solo_actor(normalized_solo, run_solo_rnn_states)
            run_solo_rnn_states = policy_solo["new_rnn_states"]
            # Running the actual execution as deterministic
            action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]
            swarm_state = get_swarm_state(env_run.unwrapped)
            # We care about where we think they'll be next timestep: that's what
            # the conformal radius is built on
            for agent_id in range(num_multi_agents):
                swarm_state.positions[agent_id, :] = pred_trajectories[agent_id][step_num][:3]
                swarm_state.velocities[agent_id, :] = pred_trajectories[agent_id][step_num][3:]
            # Apply CBF to the ego agent based on radius determined earlier
            action_solo = solo_action_fn(
                base_action=action_solo,
                env_state=env_run.unwrapped,
                swarm_state=swarm_state
            )

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs_run, rewards, dones, infos = env_run.step(actions)
            obs_run = np.array(obs_run, dtype=np.float32)

            pos, vel = extract_positions_velocities(env_run.unwrapped)
            for i in range(num_multi_agents):
                logs[i][-1]["position"].append(pos[i])
                logs[i][-1]["velocity"].append(vel[i])
            done = np.all(dones)
            step_num += 1
        env_run.close()

    return logs



def finetune_rnn(logs, num_multi_agents, predictor_checkpoint):

    # Fine-tune it just a little for the actual path
    predictors = []
    for agent_id in range(num_multi_agents):
        predictor = load_rnn_checkpoint(predictor_checkpoint, DEVICE)
        deterministic_run = np.concatenate(
            [logs[agent_id][-1]["position"], logs[agent_id][-1]["velocity"]],
            axis=-1,
        ).astype(np.float32)

        predictor.train()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=5e-4)
        criterion = nn.MSELoss()
        print(f'Training agent {agent_id}')
        with torch.enable_grad():
            for _ in tqdm(range(3)):
                sequence_tensor = torch.from_numpy(deterministic_run).to(
                    device=DEVICE, dtype=torch.float32
                )
                inputs = sequence_tensor[:-1].unsqueeze(0)
                targets = sequence_tensor[1:].unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                preds = predictor(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                optimizer.step()
        predictor.eval()
        predictors.append(predictor)
    return predictors

def roll_out_predictor(histories, predictors, agent_id, rollout_length):
    history_log = histories[agent_id]
    # Original true history array
    history_array = np.concatenate(
        [history_log["position"], history_log["velocity"]], axis=-1
    ).astype(np.float32)
    # Track history as a Python list so we can append predictions
    history = [entry.copy() for entry in history_array]
    history_len = len(history)
    # Roll out a predicted trajectory
    for i in range(rollout_length):
        # Predictor wasn't doing that well so just replaced with predicting from velocity
        # New history array generated from appended-to history
        # history_np = np.asarray(history, dtype=np.float32)
        # history_tensor = torch.from_numpy(history_np).unsqueeze(0).to(
        #     device = DEVICE, dtype = torch.float32)
        # pred = predictor(history_tensor)[0, -1].detach().cpu().numpy()
        pred = np.concatenate([
            history[-1][:3] + history[-1][3:] * DELTA_T, history[-1][3:]], 
            axis=-1)
        history.append(pred.astype(np.float32))
    return history[history_len:]

def get_alpha_bar(alpha, delta, num_trajectories):
    return alpha - np.sqrt(np.log(1 / delta) / (2 * num_trajectories))

def conformal_radii(logs, num_multi_agents, pred_trajectories, alpha, episode_length):
    radii = np.full(num_multi_agents, 0, dtype=np.float64) # Probs set to arm len
    # Need a radius for each agent
    for agent_id in range(num_multi_agents):
        predictions = pred_trajectories[agent_id]
        # Collect trajectory-level nonconformity scores
        scores = []
        for run_log in logs[agent_id]:
            score = 0  # Lowerbound on possible nonconformity score
            run = np.concatenate(
                [run_log["position"], run_log["velocity"]],
                axis=-1,
            ).astype(np.float32)
            for i in range(episode_length):
                # Largest distance across timesteps in the episode
                score = max(score, np.linalg.norm(predictions[i][:3] - run[i][:3]))
            scores.append(score)
        scores.sort()
        # Just want to visually check that this makes sense
        # print(f'Scores for agent {agent_id}: ', scores)
        conformal_radius = scores[int(np.ceil(len(scores) * (1 - alpha)) - 1)]
        radii[agent_id] = conformal_radius
    return radii


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic patrol_dual_goal data collection.")
    parser.add_argument("--multi_train_dir", required=True, help="Directory containing the trained multi-agent policy.")
    parser.add_argument("--multi_experiment", required=True, help="Experiment name for the multi-agent policy.")
    parser.add_argument("--solo_train_dir", required=True, help="Directory containing the trained single-agent policy.")
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--train_dir", required=True, help="Base directory to store the new conformal experiment.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--predictor_checkpoint", required=True, help="Path to a pretrained RNN predictor checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Seed applied before every reset to reproduce goals.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Desired probability of conformal error")
    parser.add_argument("--delta", type=float, default=0.1, help="Desired probability of a bad draw")
    parser.add_argument("--video_name", default="conformal_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=10)
    parser.add_argument("--num_trajectories", type=int, default=200)
    return parser.parse_args()


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    register_swarm_components()

    if os.path.isabs(args.video_name):
        video_dir = os.path.dirname(args.video_name) or "."
        video_file = os.path.basename(args.video_name)
    else:
        video_dir = os.path.join(args.multi_train_dir, args.multi_experiment)
        video_file = args.video_name
    os.makedirs(video_dir, exist_ok=True)
    video_frames: List[np.ndarray] = []

    experiment_dir = ensure_experiment_dir(args.train_dir, args.experiment_name)

    # Load multi config early since it has some useful info
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={cfg_multi.quads_num_agents + 1}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_collision_reward=8.0",
        "--quads_collision_hitbox_radius=2.5",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=True",
        "--quads_view_mode=topdown",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    render_mode = "rgb_array"
    
    num_multi_agents = int(eval_cfg.quads_num_agents - 1)

    # Load in multi-agents
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=render_mode)
    multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
    multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, DEVICE)
    multi_rnn_size = get_rnn_size(cfg_multi)
    multi_rnn_states = torch.zeros((num_multi_agents, multi_rnn_size), dtype=torch.float32, device=DEVICE)

    # Add in ego agent
    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=DEVICE)

    # Save initial state so we can return to it later
    obs, stored_states = deterministic_reset(env, args.seed, None)

    # Finetune a predictor for each multi-agent
    # snapshot = safe_capture_env_snapshot(env)
    # temp_env = clone_env_from_snapshot(snapshot)
    # logs = run_multi_agents(temp_env, obs, num_multi_agents, 
    #                 multi_actor, multi_rnn_states, 
    #                 solo_actor, solo_rnn_states, solo_obs_dim, fall_down,
    #                 deterministic=True)
    # predictors = finetune_rnn(logs, num_multi_agents, args.predictor_checkpoint)
    # temp_env.close()
    predictors = [None] * num_multi_agents

    # Collect arm length for default radius and dt for time btn steps
    arm_len = env.quad_arm
    DELTA_T = env.control_dt
    bar_alpha = get_alpha_bar(args.alpha, args.delta, args.num_trajectories)

    # Make sure no resets are needed for the actual run
    num_episodes = 1500 // args.episode_length - 1
    progress_bar = tqdm(range(num_episodes))
    # Init r0 to some large value that ought to be safe
    radii = np.full(num_multi_agents, 2, dtype=np.float64)
    filter = make_cbf_filter(radii)
    # Collect histories for each agent to pass to their respective predictors
    pos, vel = extract_positions_velocities(env.unwrapped)
    histories = [] # list of pos,vel histories, each entry is a quad
    solo_collision_count = 0
    for i in range(num_multi_agents + 1):
        histories.append({ "position" : [pos[i]], "velocity" : [vel[i]] })
    for episode in progress_bar:
        # progress_bar.set_postfix_str("Setting radii")
        # Find qj using old pi_j
        snapshot = safe_capture_env_snapshot(env)
        temp_env = clone_env_from_snapshot(snapshot)
        # Collect predictions of where we think they'll go using our naive predictor
        pred_trajectories = [roll_out_predictor(histories, predictors, agent_id, args.episode_length) for agent_id in range(num_multi_agents)]
        # Collect actual rollouts to compare against
        logs = run_multi_agents(temp_env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories, filter,
                    max_steps=args.episode_length, 
                    num_runs=args.num_trajectories, 
                    deterministic=False)
        # Set radius depending on how bad our prediction was
        perturbation_radius = 0.1
        qj = conformal_radii(logs, num_multi_agents, pred_trajectories, bar_alpha, args.episode_length)
        L_Xu = estimate_LXu(temp_env, obs, num_multi_agents, 
            solo_actor, solo_rnn_states, solo_obs_dim)
        L_Yx = estimate_LYx(temp_env, obs, num_multi_agents, 
            multi_actor, multi_rnn_states, 
            solo_actor, solo_rnn_states, solo_obs_dim, perturbation_radius)
        L_Yy = estimate_LYy(temp_env, obs, num_multi_agents, 
            multi_actor, multi_rnn_states,
            solo_actor, solo_rnn_states, solo_obs_dim, perturbation_radius,
)
        print(L_Xu, L_Yx, L_Yy)
        # get Deltaj
        # get rhoj
        # set total radius
        radii = np.full(num_multi_agents, 2 * arm_len, dtype=np.float64)
        radii += qj # and Deltaj and rhoj
        filter = make_cbf_filter(radii) # pi_{j+1}
        temp_env.close()
        # progress_bar.set_postfix_str(f"crashes={solo_collision_count}")
        for step in range(args.episode_length):
            # Actually run the normal execution
            obs_np = np.asarray(obs)

            obs_multi_dict = {OBS_KEY: obs_np[:num_multi_agents]}
            with torch.no_grad():
                normalized_multi = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_multi = multi_actor(normalized_multi, multi_rnn_states)
            actions_multi = policy_multi["actions"]
            multi_rnn_states = policy_multi["new_rnn_states"]
            # Running the actual execution as deterministic
            actions_multi = argmax_actions(multi_actor.action_distribution())
            if actions_multi.dim() == 1:
                actions_multi = actions_multi.unsqueeze(-1)
            actions_multi = actions_multi.detach().cpu().numpy()

            obs_solo_self = obs_np[-1, :solo_obs_dim]
            obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
            with torch.no_grad():
                normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
                policy_solo = solo_actor(normalized_solo, solo_rnn_states)
            action_solo = policy_solo["actions"]
            solo_rnn_states = policy_solo["new_rnn_states"]
            # Running the actual execution as deterministic
            action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]

            swarm_state = get_swarm_state(env.unwrapped)
            # We care about where we think they'll be next timestep: that's what
            # the conformal radius is built on
            for agent_id in range(num_multi_agents):
                swarm_state.positions[agent_id, :] = pred_trajectories[agent_id][step][:3]
                swarm_state.velocities[agent_id, :] = pred_trajectories[agent_id][step][3:]
            # Apply CBF to the ego agent based on radius determined earlier
            action_solo = filter(
                base_action=action_solo,
                env_state=env.unwrapped,
                swarm_state=swarm_state
            )
            closest_dist = 100 # arbitrarily big
            solo_pos = swarm_state.positions[-1]
            for teammate_pos in swarm_state.positions[:-1]:
                dist = np.linalg.norm(solo_pos - teammate_pos)
                closest_dist = min(dist, closest_dist)
            progress_bar.set_postfix_str(f"dist>={closest_dist}")

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs, rewards, terminated, truncated, infos = env.step(actions)

            solo_info_rewards = infos[-1].get("rewards", {})
            if solo_info_rewards.get("rewraw_quadcol", 0.0) < 0.0:
                solo_collision_count += 1
                progress_bar.set_postfix_str(f"crashes={solo_collision_count}")
            
            pos, vel = extract_positions_velocities(env.unwrapped)
            for i in range(num_multi_agents + 1): # Save for all quads
                histories[i]["position"].append(pos[i])
                histories[i]["velocity"].append(vel[i])

            terminated = np.asarray(terminated)
            truncated = np.asarray(truncated)
            dones = np.logical_or(terminated, truncated)

            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    env.close()

    if len(video_frames) > 0:
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_dir, video_file))
        print(f"[conformal_enjoy] Video saved to {final_path}")

    print(f"[conformal_enjoy] Solo drone collisions: {solo_collision_count}")



if __name__ == "__main__":
    main()
