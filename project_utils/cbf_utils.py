from __future__ import annotations

from typing import List, Tuple

import warnings
import cvxpy as cp
import numpy as np

from project_utils.utils import *

CBF_K1 = 1
CBF_K0 = 1
CBF_SLACK_WEIGHT = 1.0e4
EPSILON = 1e-3

GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81], dtype=np.float64)


# ---------------------------------------------------------------------------
# Local vector helpers (avoid dependency on quadrotor_dynamics imports)
# ---------------------------------------------------------------------------

def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _cross_mx4(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.cross(np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64))


def _cross_vec_mx4(v: np.ndarray, mx4: np.ndarray) -> np.ndarray:
    mx4_arr = np.asarray(mx4, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    tiled_v = np.tile(v_arr.reshape(1, 3), (mx4_arr.shape[0], 1))
    return np.cross(tiled_v, mx4_arr)


# ---------------------------------------------------------------------------
# Motor command conversions
# ---------------------------------------------------------------------------

def _normalized_to_thrust(norm_cmds: np.ndarray, dynamics, give_damped_cmds=False) -> np.ndarray:
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
    thrusts = thrust_max * dynamics.angvel2thrust(thrust_cmds_damp, linearity=linearity)

    torques = dynamics.prop_crossproducts * thrusts[:, None]  
    torques[:, 2] += dynamics.torque_max * dynamics.prop_ccw * thrust_cmds_damp

    if give_damped_cmds:
        return thrusts, torques, thrust_cmds_damp
    else:
        return thrusts, torques

def _thrust_to_normalized(thrusts: np.ndarray, dynamics) -> np.ndarray:
    def _invert_single(index):
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            test_norm = np.ones(4) * mid 
            test_thrusts, _ = _normalized_to_thrust(test_norm, dynamics)
            val = test_thrusts[index]
            if val < thrusts[index]:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)
    norm_cmds = np.zeros(4)
    for i in range(4):
        norm_cmds[i] = _invert_single(i)
    return norm_cmds

def cbf_dynamics(norm_cmds, dynamics, dt):
    thrusts, torques = _normalized_to_thrust(norm_cmds, dynamics)

    torque = np.sum(torques, axis=0)
    thrust = np.array([0, 0, np.sum(thrusts)])

    # ROTATIONAL DYNAMICS
    # Integrating rotations (based on current values)
    omega_vec = np.matmul(dynamics.rot, dynamics.omega)  # Change from body to world frame
    wx, wy, wz = omega_vec
    omega_norm = np.linalg.norm(omega_vec)
    if omega_norm != 0:
        # See [7]
        K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
        rot_angle = omega_norm * dt
        dRdt = np.eye(3) + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
        rot = dRdt @ dynamics.rot
    else:
        rot = dynamics.rot

    # COMPUTING OMEGA UPDATE
    omega_dot = ((1.0 / dynamics.inertia) * (_cross(-dynamics.omega, dynamics.inertia * dynamics.omega) + torque))
    omega = dynamics.omega + dt * omega_dot
    omega = np.clip(omega, a_min=-dynamics.omega_max, a_max=dynamics.omega_max)

    # TRANSLATIONAL DYNAMICS
    # Computing position
    pos = dynamics.pos + dt * dynamics.vel
    force = np.matmul(rot, thrust)
    acc = [0., 0., -9.81] + (1.0 / dynamics.mass) * force

    # Computing velocities
    vel = dynamics.vel + dt * acc
    return pos, vel, rot, omega # What I'm determining to be the state

def real_dynamics(norm_cmds, dynamics, dt):
    thrusts, torques, thrust_cmds_damp = _normalized_to_thrust(norm_cmds, dynamics, True)

    thrust_torque = np.sum(torques, axis=0)

    # Rotor drag and Rolling forces and moments
    # See Ref[1] Sec:2.1 for details
    if dynamics.C_rot_drag != 0 or dynamics.C_rot_roll != 0:
        vel_body = dynamics.rot.T @ dynamics.vel
        v_rotor = vel_body + _cross_vec_mx4(dynamics.omega, dynamics.model.prop_pos)
        v_rotor[:, 2] = 0.  # Projection to the rotor plane

        # Drag/Roll of rotors (both in body frame)
        rotor_drag_fi = - dynamics.C_rot_drag * np.sqrt(thrust_cmds_damp)[:, None] * v_rotor
        rotor_drag_force = np.sum(rotor_drag_fi, axis=0)
        rotor_drag_ti = _cross_mx4(rotor_drag_fi, dynamics.model.prop_pos)
        rotor_drag_torque = np.sum(rotor_drag_ti, axis=0)

        rotor_roll_torque = \
            - dynamics.C_rot_roll * dynamics.prop_ccw[:, None] * np.sqrt(thrust_cmds_damp)[:, None] * v_rotor
        rotor_roll_torque = np.sum(rotor_roll_torque, axis=0)
        rotor_visc_torque = rotor_drag_torque + rotor_roll_torque

        # Constraints (prevent numerical instabilities)
        vel_norm = np.linalg.norm(vel_body)
        rdf_norm = np.linalg.norm(rotor_drag_force)
        rdf_norm_clip = np.clip(rdf_norm, a_min=0., a_max=vel_norm * dynamics.mass / (2 * dt))
        if rdf_norm > EPS:
            rotor_drag_force = (rotor_drag_force / rdf_norm) * rdf_norm_clip

        # omega_norm = np.linalg.norm(dynamics.omega)
        rvt_norm = np.linalg.norm(rotor_visc_torque)
        rvt_norm_clipped = np.clip(rvt_norm, a_min=0., a_max=np.linalg.norm(dynamics.omega * dynamics.inertia) / (2 * dt))
        if rvt_norm > EPS:
            rotor_visc_torque = (rotor_visc_torque / rvt_norm) * rvt_norm_clipped
    else:
        rotor_visc_torque = rotor_drag_force = np.zeros(3)

    # (Square) Damping using torques (in case we would like to add damping using torques)
    # damping_torque = - 0.3 * dynamics.omega * np.fabs(dynamics.omega)
    torque = thrust_torque + rotor_visc_torque
    thrust = np.array([0, 0, np.sum(thrusts)])

    # ROTATIONAL DYNAMICS
    # Integrating rotations (based on current values)
    omega_vec = np.matmul(dynamics.rot, dynamics.omega)  # Change from body to world frame
    wx, wy, wz = omega_vec
    omega_norm = np.linalg.norm(omega_vec)
    if omega_norm != 0:
        # See [7]
        K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
        rot_angle = omega_norm * dt
        dRdt = np.eye(3) + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
        rot = dRdt @ dynamics.rot
    else:
        rot = dynamics.rot

    # COMPUTING OMEGA UPDATE
    omega_dot = ((1.0 / dynamics.inertia) * (_cross(-dynamics.omega, dynamics.inertia * dynamics.omega) + torque))
    # Quadratic damping
    # 0.03 corresponds to roughly 1 revolution per sec
    omega_damp_quadratic = np.clip(dynamics.damp_omega_quadratic * dynamics.omega ** 2, a_min=0.0, a_max=1.0)
    omega = dynamics.omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
    omega = np.clip(omega, a_min=-dynamics.omega_max, a_max=dynamics.omega_max)

    # TRANSLATIONAL DYNAMICS
    # Computing position
    pos = dynamics.pos + dt * dynamics.vel
    force = np.matmul(rot, thrust)
    acc = [0., 0., -9.81] + (1.0 / dynamics.mass) * force

    # Computing velocities
    vel = (1.0 - dynamics.vel_damp) * dynamics.vel + dt * acc
    return pos, vel, rot, omega # What I'm determining to be the state


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
        if radius < 0.0:
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
    u_ref_thrust, _ = _normalized_to_thrust(normalized, dynamics)

    u_min = np.zeros(4, dtype=np.float64)
    u_max = np.asarray(dynamics.thrust_max, dtype=np.float64)
    # Hopefully stop the "Solution may be innacurate" warnings since they're unactionable
    with warnings.catch_warnings(action='ignore'):
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
