"""
Shared data structures and environment reset helpers used by calibration scripts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class QuadState:
    """State snapshot for a single quadrotor."""

    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3)
    omega: np.ndarray  # (3,)
    goal: np.ndarray  # (3,)


def capture_initial_states(env) -> List[QuadState]:
    """Snapshot the current state (pos, vel, rot, omega, goal) of each quad."""
    states = []
    for quad in env.envs:
        dynamics = quad.dynamics
        states.append(
            QuadState(
                position=np.asarray(dynamics.pos, dtype=np.float64).copy(),
                velocity=np.asarray(dynamics.vel, dtype=np.float64).copy(),
                rotation=np.asarray(dynamics.rot, dtype=np.float64).copy(),
                omega=np.asarray(dynamics.omega, dtype=np.float64).copy(),
                goal=np.asarray(quad.goal, dtype=np.float64).copy(),
            )
        )
    return states


def quad_state_to_serialisable(state: QuadState) -> Dict[str, list]:
    """Convert a QuadState into JSON-friendly lists."""
    return {
        "position": state.position.tolist(),
        "velocity": state.velocity.tolist(),
        "rotation": state.rotation.tolist(),
        "omega": state.omega.tolist(),
        "goal": state.goal.tolist(),
    }


def quad_state_from_dict(data: Dict[str, Sequence[float]]) -> QuadState:
    """Rehydrate a QuadState from ``initial_states.json``."""
    return QuadState(
        position=np.asarray(data["position"], dtype=np.float64),
        velocity=np.asarray(data["velocity"], dtype=np.float64),
        rotation=np.asarray(data["rotation"], dtype=np.float64),
        omega=np.asarray(data["omega"], dtype=np.float64),
        goal=np.asarray(data["goal"], dtype=np.float64),
    )


def apply_initial_states(env, states: List[QuadState]) -> List[np.ndarray]:
    """
    Force each quadrotor to the provided state and rebuild the observation list
    returned to the control loop.
    """
    if len(states) != len(env.envs):
        raise ValueError("Number of stored initial states does not match number of agents.")

    for idx, (quad, state) in enumerate(zip(env.envs, states)):
        quad.dynamics.set_state(state.position, state.velocity, state.rotation, state.omega)
        quad.dynamics.reset()
        quad.dynamics.on_floor = False
        quad.dynamics.crashed_floor = quad.dynamics.crashed_wall = quad.dynamics.crashed_ceiling = False
        quad.tick = 0
        quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]
        quad.goal = state.goal.copy()
        env.pos[idx, :] = quad.dynamics.pos
        env.vel[idx, :] = quad.dynamics.vel

    obs = [quad.state_vector(quad) for quad in env.envs]
    if env.num_use_neighbor_obs > 0:
        obs = env.add_neighborhood_obs(obs)
    return obs


def set_global_seed(seed: int) -> None:
    """Synchronise Python, NumPy, and Torch RNGs to the chosen seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reseed_quads(env, base_seed: int) -> None:
    """Reseed each QuadrotorSingle for deterministic spawn reproduction."""
    for idx, quad in enumerate(env.envs):
        quad._seed(base_seed + idx)


def deterministic_reset(
    env, seed: int, stored_states: Optional[List[QuadState]]
) -> Tuple[np.ndarray, List[QuadState]]:
    """
    Reset the environment while enforcing deterministic scenario generation and
    optionally reapplying saved initial states.
    """
    set_global_seed(seed)
    reseed_quads(env, seed)
    obs, _ = env.reset()
    if stored_states is None:
        states = capture_initial_states(env)
        return np.asarray(obs), states
    obs = apply_initial_states(env.unwrapped, stored_states)
    return np.asarray(obs), stored_states


def extract_positions_velocities(env) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked positions and velocities (num_agents, 3)."""
    positions, velocities = [], []
    for quad in env.envs:
        dynamics = quad.dynamics
        positions.append(np.asarray(dynamics.pos, dtype=np.float64))
        velocities.append(np.asarray(dynamics.vel, dtype=np.float64))
    return np.stack(positions, axis=0), np.stack(velocities, axis=0)


__all__ = [
    "QuadState",
    "apply_initial_states",
    "capture_initial_states",
    "deterministic_reset",
    "extract_positions_velocities",
    "quad_state_from_dict",
    "quad_state_to_serialisable",
    "reseed_quads",
    "set_global_seed",
]
