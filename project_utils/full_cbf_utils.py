from __future__ import annotations

import itertools
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
# CBF helpers
# ---------------------------------------------------------------------------

def _calculate_Lh(
    i: int,
    j: int,
    swarm_state: SwarmState,
    mass: float,
    dt: float,
    thrust_bounds: Tuple[float, float],
    r: float,
) -> float:
    """
    Pairwise conservative finite-search upper bound for:
      (1 + 2 * CBF_K0) * ||next_pos_rel|| + ||next_vel_rel||

    where relative terms are for agent pair (i, j), and the maximization is over:
      - u_i, u_j at control-box corners with scalar bounds [u_min, u_max]
      - additive velocity disturbance direction s_v in {-1, 1}^3 scaled by r
    """
    u_min, u_max = float(thrust_bounds[0]), float(thrust_bounds[1])
    r = float(r)

    positions = swarm_state.positions
    velocities = swarm_state.velocities
    rotations = swarm_state.rotations

    pos_rel = np.asarray(positions[i] - positions[j], dtype=np.float64)
    vel_rel = np.asarray(velocities[i] - velocities[j], dtype=np.float64)
    rot_i = np.asarray(rotations[i], dtype=np.float64)
    rot_j = np.asarray(rotations[j], dtype=np.float64)

    # Relative position one step ahead (current model, independent of control).
    next_pos_rel = pos_rel + dt * vel_rel
    pos_term = (1.0 + 2.0 * CBF_K0) * np.linalg.norm(next_pos_rel)

    # (1/m) R e3 1^T in R^{3x4} for each agent.
    acc_scale_i = (1.0 / mass) * np.outer(rot_i[:, 2], np.ones(4, dtype=np.float64))
    acc_scale_j = (1.0 / mass) * np.outer(rot_j[:, 2], np.ones(4, dtype=np.float64))

    # Corners for each agent's 4 motor thrusts under scalar bounds.
    u_corners = np.array(list(itertools.product([u_min, u_max], repeat=4)), dtype=np.float64)
    dirs = np.array(np.meshgrid([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])).T.reshape(-1, 3)

    best = 0.0
    for u_i in u_corners:
        acc_i = GRAVITY_VECTOR + acc_scale_i @ u_i
        for u_j in u_corners:
            acc_j = GRAVITY_VECTOR + acc_scale_j @ u_j
            base_next_vel_rel = vel_rel + dt * (acc_i - acc_j)
            for s_v in dirs:
                next_vel_rel = base_next_vel_rel + r * s_v
                val = pos_term + np.linalg.norm(next_vel_rel)
                if val > best:
                    best = val

    return float(best)

def _cbf_h_values(
    i, j,
    swarm_state: SwarmState,
    radius: float,
    mass: float,
    dt: float,
    u_vars: cp.Variable,
):
    """
    Pairwise CBF terms for agents i and j.
    """
    positions = swarm_state.positions
    velocities = swarm_state.velocities
    rotations = swarm_state.rotations

    u_i = u_vars[4 * i: 4 * (i + 1)]
    u_j = u_vars[4 * j: 4 * (j + 1)]

    pos_rel = positions[i] - positions[j] # (x-p)
    vel_rel = velocities[i] - velocities[j] 
    next_pos_rel = pos_rel + dt * vel_rel
    acc_scale_i = (1.0 / mass) * np.outer(rotations[i][:, 2], np.ones(4))
    acc_scale_j = (1.0 / mass) * np.outer(rotations[j][:, 2], np.ones(4))
    acc_i = GRAVITY_VECTOR + acc_scale_i @ u_i
    acc_j = GRAVITY_VECTOR + acc_scale_j @ u_j
    acc_rel = acc_i - acc_j
    next_vel_rel = vel_rel + dt * acc_rel
    next_h = CBF_K0 * (next_pos_rel @ next_pos_rel - radius**2) + next_pos_rel @ next_vel_rel
    h_value = CBF_K0 * (pos_rel @ pos_rel - radius**2) + pos_rel @ vel_rel
    return h_value, next_h


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
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    u_min = float(thrust_bounds[0])
    u_max = float(thrust_bounds[1])
    num_agents = int(swarm_state.positions.shape[0])
    expected_dim = 4 * num_agents
    if u_ref.shape[0] != expected_dim:
        raise ValueError(f"u_ref_thrust has shape {u_ref.shape}, expected ({expected_dim},)")
    if u_min > u_max:
        raise ValueError(f"Invalid scalar thrust bounds: u_min={u_min} > u_max={u_max}")

    constraints: List[cp.Constraint] = []
    u_var = cp.Variable(expected_dim)
    slack = cp.Variable()

    # Pairwise constraints across all agent pairs.
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            h_value, h_next = _cbf_h_values(i, j, swarm_state, separation_radius, mass, dt, u_var)
            Lh_ij = _calculate_Lh(
                i=i,
                j=j,
                swarm_state=swarm_state,
                mass=mass,
                dt=dt,
                thrust_bounds=(u_min, u_max),
                r=r,
            )
            constraints.append(h_next - (1 - gamma) * h_value >= Lh_ij * r - slack)

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
            approx = u_ref # No iterate returned
        clipped = np.clip(approx, u_min, u_max)
        print("QP timed out; returning last iterate:", clipped)
        return clipped
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print("OTHER ISSUE", problem.status)
        return np.clip(u_ref, u_min, u_max)
    solution = np.array(u_var.value, dtype=np.float64)
    return solution 


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
    with warnings.catch_warnings(action='ignore'):
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
    clipped_action = safe_actions
    return clipped_action

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
