from __future__ import annotations

from typing import List, Tuple

import cvxpy as cp
import numpy as np

from utils import *

CBF_K1 = 0.1
CBF_K0 = 0.1
CBF_SLACK_WEIGHT = 1.0e4
EPSILON = 1e-3

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
    motor_tau_down = np.asarray(dynamics.motor_tau_down, dtype=np.float64)
    motor_tau = dynamics.motor_tau_up * np.ones([4, ])
    motor_tau[norm_cmds < dynamics.thrust_cmds_damp] = motor_tau_down
    motor_tau[motor_tau > 1.] = 1.
    thrust_rot = norm_cmds ** 0.5
    thrust_rot_damp = motor_tau * (thrust_rot - dynamics.thrust_rot_damp) + dynamics.thrust_rot_damp
    thrust_cmds_damp = thrust_rot_damp ** 2
    
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    return thrust_max * dynamics.angvel2thrust(thrust_cmds_damp, linearity=linearity)

def _thrust_to_normalized(thrusts: np.ndarray, dynamics) -> np.ndarray:
    def _invert_single(index):
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            test_norm = np.ones(4) * mid 
            val = _normalized_to_thrust(test_norm, dynamics)[index]
            if val < thrusts[index]:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)
    norm_cmds = np.zeros(4)
    for i in range(4):
        norm_cmds[i] = _invert_single(i)
    return norm_cmds

# def _thrust_to_normalized(thrusts: np.ndarray, dynamics) -> np.ndarray:
#     """
#     Invert ``_normalized_to_thrust`` by numerically recovering the [0, 1] motor
#     commands whose actuator model yields the requested thrust magnitudes.
#     """
#     thrusts = np.asarray(thrusts, dtype=np.float64)
#     thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
#     ratio = np.clip(thrusts / thrust_max, 0.0, 1.0)
#     linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
#     linearity = np.broadcast_to(np.atleast_1d(linearity), ratio.shape).astype(np.float64)

#     def _invert_single(r: float, lin: float) -> float:
#         """
#         Use a monotone bisection driven entirely by ``angvel2thrust`` so the
#         inverse stays coupled to the simulator's actuator curves.
#         """
#         low, high = 0.0, 1.0
#         for _ in range(30):
#             mid = 0.5 * (low + high)
#             val = float(dynamics.angvel2thrust(np.array([mid]), linearity=np.array([lin]))[0])
#             if val < r:
#                 low = mid
#             else:
#                 high = mid
#         return 0.5 * (low + high)

#     vectorized = np.vectorize(_invert_single, otypes=[np.float64])
#     normalized = vectorized(ratio, linearity)
#     return np.clip(normalized, 0.0, 1.0)


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
    pos_rel = solo_pos - teammate_pos # (x-p)
    # v_rel = solo_vel - teammate_vel
    h_value = float(np.dot(pos_rel, pos_rel) - radius**2) # h
    relpos_dot_gravity = float(np.dot(pos_rel, GRAVITY_VECTOR)) # (x-p)ᵀg
    relpos_dot_v = float(np.dot(pos_rel, solo_vel)) # (x-p)ᵀv
    v_sq = float(np.dot(solo_vel, solo_vel)) # vᵀv
    thrust_axis_world = solo_rot[:, 2] # R e₃
    thrust_alignment = float(np.dot(pos_rel, thrust_axis_world)) # (x-p)ᵀRe₃
    c_scale = (2.0 / mass) * thrust_alignment # (2/m) (x-p)ᵀRe₃
    LgLfh = c_scale * np.ones(4, dtype=np.float64) # (2/m) (x-p)ᵀRe₃ 1ᵀ
    Lf2h = 2.0 * v_sq + 2.0 * relpos_dot_gravity # 2vᵀv + 2(x-p)ᵀg
    Lfh = 2 * relpos_dot_v # 2(x-p)ᵀv
    return h_value, Lfh, Lf2h, LgLfh


def _solve_cbf_qp(
    *,
    u_ref_thrust: np.ndarray,
    swarm_state: SwarmState,
    radii: np.ndarray,
    mass: float,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
    debug=False
) -> np.ndarray:
    """
    Build and solve the ECBF quadratic program described in the task statement.

    Decision variables
    ------------------
    - ``u`` ∈ ℝ⁴ : per-motor thrusts.
    - ``slack`` ≥ 0 : shared softening variable.

    Objective
    ---------
    minimise ‖u - u_ref‖² + CBF_SLACK_WEIGHT · slack²

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

    hdd_list = []
    hd_list = []
    h_list = []

    for teammate_idx in range(num_multi_agents):
        radius = float(radii[teammate_idx])
        if radius <= 0.0:
            continue
        teammate_pos = swarm_state.positions[teammate_idx]
        teammate_vel = swarm_state.velocities[teammate_idx]
        h_value, Lfh, Lf2h, LgLfh = _ecbf_coefficients(
            solo_pos=solo_pos,
            solo_vel=solo_vel,
            solo_rot=solo_rot,
            teammate_pos=teammate_pos,
            teammate_vel=teammate_vel,
            radius=radius,
            mass=mass,
        )
        hdd = Lf2h + LgLfh @ u_var
        hd = Lfh
        hdd_list.append(hdd)
        hd_list.append(hd)
        h_list.append(h_value)
        constraints.append(hdd + CBF_K1 * (hd + CBF_K0 * h_value) >= - slack) #  EPSILON

    if len(constraints) == 0:
        return np.clip(u_ref, u_min, u_max)

    objective = cp.sum_squares(u_var - u_ref) + CBF_SLACK_WEIGHT * cp.square(slack)
    constraints.extend(
        [
            u_var >= u_min,
            u_var <= u_max,
            slack >= 0,
        ]
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)

    try:
        problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
    except cp.SolverError:
        approx = u_var.value
        if approx is None:
            approx = u_ref # No iteratre returned
        clipped = np.clip(approx, u_min, u_max)
        print("QP timed out; returning last iterate:", clipped)
        if debug:
            return clipped, h_list, hd_list, hdd_list
        return clipped
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print('OTHER ISSUE', problem.status)
        return np.clip(u_ref, u_min, u_max)
    if debug:
        for agent_id in range(num_multi_agents):
            print('C0:', h_list[agent_id])
            print('C1:', hd_list[agent_id] + CBF_K0 * (h_list[agent_id]))
            print('C2:', hdd_list[agent_id].value + CBF_K1 * (hd_list[agent_id] + CBF_K0 * (h_list[agent_id])))
        # print('Slack: ', slack.value, 'u dist:', np.linalg.norm(u_ref - u_var.value), u_var.value)
        # print(np.linalg.norm(u_var.value - u_min), np.linalg.norm(u_var.value - u_max))
    solution = np.array(u_var.value, dtype=np.float64)
    # solution = u_ref / np.sum(u_ref) * np.sum(solution) # 
    if debug:
        return solution, h_list, hd_list, hdd_list
    return solution # np.clip(solution, u_min, u_max)


def apply_cbf_filter(
    base_action: np.ndarray,
    radii: np.ndarray,
    env_state,
    swarm_state: SwarmState,
    debug=False
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
    outputs = _solve_cbf_qp(
        u_ref_thrust=u_ref_thrust,
        swarm_state=swarm_state,
        radii=radii,
        mass=float(dynamics.mass),
        thrust_bounds=(u_min, u_max),
        debug=debug
    )
    if debug:
        safe_thrust, h_list, hd_list, hdd_list = outputs
    else:
        safe_thrust = outputs

    # Convert Newton thrust back to the environment's action space.
    safe_normalized = _thrust_to_normalized(safe_thrust, dynamics)
    safe_action = 2.0 * safe_normalized - 1.0
    clipped_action = np.clip(safe_action.astype(np.float32), -1.0, 1.0)
    if debug:
        return clipped_action, h_list, hd_list, hdd_list
    else:
        return clipped_action

def make_cbf_filter(radii: np.ndarray):
    def filter(base_action: np.ndarray, env_state, swarm_state: SwarmState, debug=False):
        return apply_cbf_filter(base_action, radii, env_state, swarm_state, debug=debug)
    return filter

