from __future__ import annotations

from typing import Dict, List, Tuple

import warnings
import cvxpy as cp
import numpy as np

from project_utils.full_cbf_utils import (
    CBF_K0,
    CBF_K1,
    CBF_SLACK_WEIGHT,
    EPSILON,
    GRAVITY_VECTOR,
    cbf_dynamics,
    real_dynamics,
    _normalized_to_thrust,
    _thrust_to_normalized,
)
from project_utils.utils import SwarmState, get_swarm_state


class _CachedJointCBF:
    def __init__(self, num_agents: int, gamma: float):
        self.num_agents = int(num_agents)
        self.gamma = float(gamma)
        self.pair_indices: List[Tuple[int, int]] = [
            (i, j) for i in range(self.num_agents) for j in range(i + 1, self.num_agents)
        ]
        self.expected_dim = 4 * self.num_agents
        self.num_pairs = len(self.pair_indices)

        self.A = cp.Parameter((self.num_pairs, self.expected_dim))
        self.rhs = cp.Parameter(self.num_pairs)
        self.u_ref = cp.Parameter(self.expected_dim)
        self.u_min = cp.Parameter()
        self.u_max = cp.Parameter()
        self.u_var = cp.Variable(self.expected_dim)
        self.slack = cp.Variable()

        constraints = [
            self.A @ self.u_var + self.slack >= self.rhs,
            self.u_var >= self.u_min,
            self.u_var <= self.u_max,
            self.slack >= 0,
        ]
        objective = cp.sum_squares(self.u_var - self.u_ref) + CBF_SLACK_WEIGHT * cp.square(self.slack)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)


_JOINT_CBF_CACHE: Dict[Tuple[int, float], _CachedJointCBF] = {}


def _get_cached_joint_cbf(num_agents: int, gamma: float) -> _CachedJointCBF:
    key = (int(num_agents), float(gamma))
    cache = _JOINT_CBF_CACHE.get(key)
    if cache is None:
        cache = _CachedJointCBF(num_agents=num_agents, gamma=gamma)
        _JOINT_CBF_CACHE[key] = cache
    return cache


def _build_joint_cbf_affine_terms(
    swarm_state: SwarmState,
    u_ref_thrust: np.ndarray,
    separation_radius: float,
    mass: float,
    dt: float,
    gamma: float,
    cache: _CachedJointCBF,
    r: float,
) -> Tuple[np.ndarray, np.ndarray]:
    u_ref_thrust = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    positions = np.asarray(swarm_state.positions, dtype=np.float64)
    velocities = np.asarray(swarm_state.velocities, dtype=np.float64)
    rotations = np.asarray(swarm_state.rotations, dtype=np.float64)
    num_agents = int(positions.shape[0])

    if cache.num_pairs == 0 or cache.expected_dim == 0:
        return np.zeros((0, cache.expected_dim), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    dt = float(dt) / 2.0
    ones4 = np.ones(4, dtype=np.float64)
    sep = float(separation_radius)

    acc_scales = np.empty((num_agents, 3, 4), dtype=np.float64)
    for i in range(num_agents):
        acc_scales[i] = (1.0 / float(mass)) * np.outer(rotations[i][:, 2], ones4)

    A = np.zeros((cache.num_pairs, cache.expected_dim), dtype=np.float64)
    rhs = np.zeros((cache.num_pairs,), dtype=np.float64)

    for row_idx, (i, j) in enumerate(cache.pair_indices):
        rel_pos = positions[i] - positions[j]
        rel_vel = velocities[i] - velocities[j]
        acc_scale_i = acc_scales[i]
        acc_scale_j = acc_scales[j]

        u_ref_i = u_ref_thrust[4 * i: 4 * (i + 1)]
        u_ref_j = u_ref_thrust[4 * j: 4 * (j + 1)]
        rel_acc_ref = acc_scale_i @ u_ref_i - acc_scale_j @ u_ref_j

        z_ref = rel_pos + dt * rel_vel + dt * (rel_vel + dt * rel_acc_ref)
        z_ref_norm = float(np.linalg.norm(z_ref))
        if z_ref_norm < 1e-6:
            z_ref_norm = 1.0e-6
        grad = z_ref / z_ref_norm

        h_value = float(np.linalg.norm(rel_pos) - sep)
        a_const = rel_pos + 2.0 * dt * rel_vel
        # Keep separate per-agent blocks so the cached affine form matches
        # the original full CBF linearization exactly.
        row_i = (dt * dt) * (grad @ acc_scale_i)
        row_j = -(dt * dt) * (grad @ acc_scale_j)

        const_term = float(grad @ a_const + z_ref_norm - grad @ z_ref - sep - (1.0 - gamma) * h_value)
        A[row_idx, 4 * i: 4 * (i + 1)] = row_i
        A[row_idx, 4 * j: 4 * (j + 1)] = row_j
        rhs[row_idx] = float(r - const_term)

    return A, rhs


def _solve_cbf_qp(
    gamma,
    u_ref_thrust: np.ndarray,
    swarm_state: SwarmState,
    r,
    separation_radius: float,
    mass: float,
    dt,
    thrust_bounds: Tuple[float, float],
) -> np.ndarray:
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    u_min = float(thrust_bounds[0])
    u_max = float(thrust_bounds[1])
    num_agents = int(swarm_state.positions.shape[0])
    expected_dim = 4 * num_agents
    if u_ref.shape[0] != expected_dim:
        raise ValueError(f"u_ref_thrust has shape {u_ref.shape}, expected ({expected_dim},)")
    if u_min > u_max:
        raise ValueError(f"Invalid scalar thrust bounds: u_min={u_min} > u_max={u_max}")

    if expected_dim == 0 or num_agents <= 1:
        return np.clip(u_ref, u_min, u_max)

    cache = _get_cached_joint_cbf(num_agents=num_agents, gamma=gamma)
    A, rhs = _build_joint_cbf_affine_terms(
        swarm_state=swarm_state,
        u_ref_thrust=u_ref,
        separation_radius=separation_radius,
        mass=mass,
        dt=dt,
        gamma=float(gamma),
        cache=cache,
        r=float(r),
    )

    cache.A.value = A
    cache.rhs.value = rhs
    cache.u_ref.value = u_ref
    cache.u_min.value = u_min
    cache.u_max.value = u_max

    try:
        cache.problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
    except cp.SolverError:
        approx = cache.u_var.value
        if approx is None:
            approx = u_ref
        clipped = np.clip(np.asarray(approx, dtype=np.float64), u_min, u_max)
        print("QP timed out; returning last iterate:", clipped)
        return clipped

    if cache.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print("OTHER ISSUE", cache.problem.status)
        return np.clip(u_ref, u_min, u_max)

    return np.array(cache.u_var.value, dtype=np.float64)


def apply_cbf_filter(
    base_action: np.ndarray,
    env_state,
    r,
    separation_radius: float,
    gamma,
    debug=False,
) -> np.ndarray:
    """
    Wrap raw policy actions with pairwise swarm CBF safety filter.

    Parameters
    ----------
    base_action : np.ndarray
        Motor commands in [-1, 1]^{N x 4} (or shape (4,) for N=1).
    env_state :
        Vectorised environment used to access the simulator dynamics.
    r :
        Pairwise separation radius.
    gamma :
        CBF contraction factor in (0, 1].
    """
    actions = np.asarray(base_action, dtype=np.float64)
    squeeze_out = False
    if actions.ndim == 1:
        actions = actions.reshape(1, 4)
        squeeze_out = True

    num_agents = len(env_state.envs)
    if actions.shape != (num_agents, 4):
        raise ValueError(f"base_action shape {actions.shape} does not match expected ({num_agents}, 4)")

    swarm_state = get_swarm_state(env_state)

    u_refs = []
    per_agent_thrust_max = []
    for agent_idx, quad in enumerate(env_state.envs):
        dynamics = quad.dynamics
        normalized = np.clip(0.5 * (actions[agent_idx] + 1.0), 0.0, 1.0)
        u_ref_thrust, _ = _normalized_to_thrust(normalized, dynamics)
        u_refs.append(u_ref_thrust)
        per_agent_thrust_max.append(float(np.min(np.asarray(dynamics.thrust_max, dtype=np.float64))))

    u_ref_concat = np.concatenate(u_refs, axis=0)
    u_min_scalar = 0.0
    u_max_scalar = float(np.min(np.asarray(per_agent_thrust_max, dtype=np.float64)))

    mass = float(env_state.envs[0].dynamics.mass)
    dt = float(env_state.control_dt)

    # Hopefully stop the "Solution may be inaccurate" warnings since they're unactionable
    with warnings.catch_warnings(action="ignore"):
        safe_thrust_concat = _solve_cbf_qp(
            gamma=gamma,
            u_ref_thrust=u_ref_concat,
            swarm_state=swarm_state,
            r=r,
            separation_radius=separation_radius,
            mass=mass,
            dt=dt,
            thrust_bounds=(u_min_scalar, u_max_scalar),
        )

    safe_actions = np.zeros((num_agents, 4), dtype=np.float32)
    for agent_idx, quad in enumerate(env_state.envs):
        dynamics = quad.dynamics
        thrust_block = safe_thrust_concat[4 * agent_idx: 4 * (agent_idx + 1)]
        safe_normalized = _thrust_to_normalized(thrust_block, dynamics)
        safe_action = 2.0 * safe_normalized - 1.0
        safe_actions[agent_idx] = np.clip(safe_action.astype(np.float32), -1.0, 1.0)

    if squeeze_out:
        return safe_actions[0]
    return safe_actions


def make_cbf_filter(r: float, separation_radius: float, gamma: float):
    def filter(base_action: np.ndarray, env_state, debug=False):
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_state,
            r=r,
            separation_radius=separation_radius,
            gamma=gamma,
            debug=debug,
        )

    return filter
