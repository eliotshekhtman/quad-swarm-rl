from __future__ import annotations

from typing import List, Tuple

import cvxpy as cp
import numpy as np

from utils import *

CBF_K1 = 8.0
CBF_K0 = 2.0
CBF_SLACK_WEIGHT = 1.0e4
CBF_QP_DIAGONAL = np.ones(4, dtype=np.float64)
CBF_QP_SOLVER = "OSQP"

GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81], dtype=np.float64)


# ---------------------------------------------------------------------------
# Motor command conversions
# ---------------------------------------------------------------------------

def _normalized_to_thrust(norm_cmds: np.ndarray, dynamics) -> np.ndarray:
    """
    Convert environment actions in [0, 1] into per-rotor thrust magnitudes (Newtons).

    The quadrotor dynamics use a convex combination of linear and quadratic curves
    to map the high-level command to thrust.  We delegate the core conversion to
    ``QuadrotorDynamics.angvel2thrust`` so the QP shares the exact actuator model.
    """
    commands = np.clip(np.asarray(norm_cmds, dtype=np.float64), 0.0, 1.0)
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    linearity = np.broadcast_to(np.atleast_1d(linearity), commands.shape).astype(np.float64)
    return thrust_max * dynamics.angvel2thrust(commands, linearity=linearity)


def _thrust_to_normalized(thrusts: np.ndarray, dynamics) -> np.ndarray:
    """
    Invert ``_normalized_to_thrust`` by numerically recovering the [0, 1] motor
    commands whose actuator model yields the requested thrust magnitudes.
    """
    thrusts = np.asarray(thrusts, dtype=np.float64)
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    max_safe = np.clip(thrust_max, 1e-6, None)
    ratio = np.clip(thrusts / max_safe, 0.0, 1.0)
    linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    linearity = np.broadcast_to(np.atleast_1d(linearity), ratio.shape).astype(np.float64)

    def _invert_single(r: float, lin: float) -> float:
        """
        Use a monotone bisection driven entirely by ``angvel2thrust`` so the
        inverse stays coupled to the simulator's actuator curves.
        """
        if abs(lin - 1.0) < 1e-9:
            return r
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            val = float(dynamics.angvel2thrust(np.array([mid]), linearity=np.array([lin]))[0])
            if val < r:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    vectorized = np.vectorize(_invert_single, otypes=[np.float64])
    normalized = vectorized(ratio, linearity)
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ECBF helpers
# ---------------------------------------------------------------------------

def _ecbf_coefficients(
    *,
    solo_pos: np.ndarray,
    solo_vel: np.ndarray,
    solo_rot: np.ndarray,
    teammate_pos: np.ndarray,
    teammate_vel: np.ndarray,
    radius: float,
    mass: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute the ``(a, b, h)`` triple for the ECBF constraint ``a^T u ≥ -b - slack``.

    The derivation follows the supplied formulation:

        h(x) = ||x - p||² - r²
        ḣ(x) = 2 zᵀ v_rel
        ḧ(x) = 2 ||v_rel||² + 2 zᵀ g + (2/m) (zᵀ R e₃) 1ᵀ u

    where:

        z      := solo_pos - teammate_pos
        v_rel  := solo_vel - teammate_vel   (moving obstacle extension)
        g      := GRAVITY_VECTOR
        R e₃   := third column of the body-to-world rotation

    The moving-obstacle extension treats the teammate as a point translating with
    velocity ``teammate_vel``.  This keeps the barrier conservative: if the
    teammate is stationary it reduces to the exact textbook form.
    """
    z_vec = solo_pos - teammate_pos # (x-p)
    v_rel = solo_vel - teammate_vel
    h_value = float(np.dot(z_vec, z_vec) - radius**2) # h
    z_dot_gravity = float(np.dot(z_vec, GRAVITY_VECTOR)) # (x-p)ᵀg
    z_dot_v_rel = float(np.dot(z_vec, solo_vel)) # (x-p)ᵀv
    v_rel_sq = float(np.dot(solo_vel, solo_vel)) # vᵀv
    thrust_axis_world = solo_rot[:, 2] # R e₃
    thrust_alignment = float(np.dot(z_vec, thrust_axis_world)) # (x-p)ᵀRe₃
    c_scale = (2.0 / mass) * thrust_alignment # (2/m) (x-p)ᵀRe₃
    a_vec = c_scale * np.ones(4, dtype=np.float64) # (2/m) (x-p)ᵀRe₃ 1ᵀ
    b_scalar = ( # 2vᵀv + 2vᵀg + a1(2(x-p)ᵀv + a0(h))
        2.0 * v_rel_sq 
        + 2.0 * z_dot_gravity
        + 2.0 * CBF_K1 * z_dot_v_rel
        + CBF_K1 * CBF_K0 * h_value
    )
    return a_vec, b_scalar, h_value


def _solve_cbf_qp(
    *,
    u_ref_thrust: np.ndarray,
    swarm_state: SwarmState,
    radii: np.ndarray,
    mass: float,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Build and solve the ECBF quadratic program described in the task statement.

    Decision variables
    ------------------
    - ``u`` ∈ ℝ⁴ : per-motor thrusts.
    - ``slack`` ≥ 0 : shared softening variable.

    Objective
    ---------
    minimise ‖u - u_ref‖²_W + CBF_SLACK_WEIGHT · slack

    Constraints
    -----------
    - One inequality per teammate: ``a_iᵀ u ≥ -b_i - slack``.
    - Elementwise thrust bounds ``u_min ≤ u ≤ u_max``.
    - ``slack ≥ 0``.
    """
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64)
    u_min, u_max = thrust_bounds
    solo_pos = swarm_state.positions[-1]
    solo_vel = swarm_state.velocities[-1]
    solo_rot = swarm_state.rotations[-1]
    num_multi_agents = len(radii)

    constraints: List[cp.Constraint] = []
    u_var = cp.Variable(4)
    slack = cp.Variable()

    for teammate_idx in range(num_multi_agents):
        radius = float(radii[teammate_idx])
        if radius <= 0.0:
            continue
        teammate_pos = swarm_state.positions[teammate_idx]
        teammate_vel = swarm_state.velocities[teammate_idx]
        a_vec, b_scalar, _ = _ecbf_coefficients(
            solo_pos=solo_pos,
            solo_vel=solo_vel,
            solo_rot=solo_rot,
            teammate_pos=teammate_pos,
            teammate_vel=teammate_vel,
            radius=radius,
            mass=mass,
        )
        constraints.append(a_vec @ u_var >= -b_scalar - slack)

    if len(constraints) == 0:
        return np.clip(u_ref, u_min, u_max)

    weight_matrix = np.diag(CBF_QP_DIAGONAL)
    objective = cp.quad_form(u_var - u_ref, weight_matrix) + CBF_SLACK_WEIGHT * slack
    constraints.extend(
        [
            u_var >= u_min,
            u_var <= u_max,
            slack >= 0,
        ]
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver = getattr(cp, CBF_QP_SOLVER, cp.OSQP)
    try:
        problem.solve(solver=solver, warm_start=True, verbose=False)
    except cp.SolverError:
        return np.clip(u_ref, u_min, u_max)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return np.clip(u_ref, u_min, u_max)
    # print('Slack: ', slack.value)
    solution = np.array(u_var.value, dtype=np.float64)
    return np.clip(solution, u_min, u_max)


def apply_cbf_filter(
    base_action: np.ndarray,
    radii: np.ndarray,
    env_state,
    swarm_state: SwarmState,
) -> np.ndarray:
    """
    Wrap the raw solo-policy action with the ECBF safety filter.
    Assumed that the protected solo agent is at the last index.

    Parameters
    ----------
    base_action : np.ndarray
        Motor command in [-1, 1]⁴ produced by the solo policy.
    env_state :
        Vectorised environment used to access the simulator dynamics.
    swarm_state : SwarmState
        Pre-computed positions, velocities, and orientations for the current step.
    """
    quad = env_state.envs[-1]
    dynamics = quad.dynamics

    # Action space conversions: [-1, 1] → [0, 1] → Newton thrusts.
    base_action = np.asarray(base_action, dtype=np.float64)
    normalized = np.clip(0.5 * (base_action + 1.0), 0.0, 1.0)
    u_ref_thrust = _normalized_to_thrust(normalized, dynamics)

    u_min = np.zeros(4, dtype=np.float64)
    u_max = np.asarray(dynamics.thrust_max, dtype=np.float64)
    safe_thrust = _solve_cbf_qp(
        u_ref_thrust=u_ref_thrust,
        swarm_state=swarm_state,
        radii=radii,
        mass=float(dynamics.mass),
        thrust_bounds=(u_min, u_max),
    )

    # Convert Newton thrust back to the environment's action space.
    safe_normalized = _thrust_to_normalized(safe_thrust, dynamics)
    safe_action = 2.0 * safe_normalized - 1.0
    return np.clip(safe_action.astype(np.float32), -1.0, 1.0)

def make_cbf_filter(radii: np.ndarray):
    def filter(base_action: np.ndarray, env_state, swarm_state: SwarmState):
        return apply_cbf_filter(base_action, radii, env_state, swarm_state)
    return filter

