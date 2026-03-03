from __future__ import annotations

import itertools
from dataclasses import dataclass
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

def _normalized_to_thrust(norm_cmds: np.ndarray, dynamics, steps: int = 1) -> np.ndarray:
    """
    Convert environment actions in [0, 1] into per-rotor thrust magnitudes (Newtons).

    The quadrotor dynamics use a convex combination of linear and quadratic curves
    to map the high-level command to thrust.  We delegate the core conversion to
    ``QuadrotorDynamics.angvel2thrust`` so the QP shares the exact actuator model.
    """
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    norm_cmds = np.asarray(norm_cmds, dtype=np.float64)
    thrust_rot = norm_cmds ** 0.5
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    motor_tau_down = np.asarray(dynamics.motor_tau_down, dtype=np.float64)

    thrust_rot_damp = np.asarray(dynamics.thrust_rot_damp, dtype=np.float64).copy()
    thrust_cmds_damp = np.asarray(dynamics.thrust_cmds_damp, dtype=np.float64).copy()
    outputs = []
    for _ in range(steps):
        motor_tau = dynamics.motor_tau_up * np.ones([4, ], dtype=np.float64)
        motor_tau[norm_cmds < thrust_cmds_damp] = motor_tau_down
        motor_tau[motor_tau > 1.0] = 1.0
        thrust_rot_damp = motor_tau * (thrust_rot - thrust_rot_damp) + thrust_rot_damp
        thrust_cmds_damp = thrust_rot_damp ** 2

        thrusts = thrust_max * dynamics.angvel2thrust(thrust_cmds_damp, linearity=linearity)
        torques = dynamics.prop_crossproducts * thrusts[:, None]
        torques[:, 2] += dynamics.torque_max * dynamics.prop_ccw * thrust_cmds_damp
        outputs.append((thrusts, torques, thrust_cmds_damp.copy()))

    if steps == 1:
        thrusts, torques, _ = outputs[0]
        return thrusts, torques
    return outputs

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

def cbf_dynamics(norm_cmds, dynamics, dt, steps=2):
    dt = dt / steps
    thrusts, torques = _normalized_to_thrust(norm_cmds, dynamics)
    rot = dynamics.rot 
    omega = dynamics.omega 
    pos = dynamics.pos 
    vel = dynamics.vel 

    for step in range(steps):
        torque = np.sum(torques, axis=0)
        thrust = np.array([0, 0, np.sum(thrusts)])

        # ROTATIONAL DYNAMICS
        # Integrating rotations (based on current values)
        omega_vec = np.matmul(rot, omega)  # Change from body to world frame
        wx, wy, wz = omega_vec
        omega_norm = np.linalg.norm(omega_vec)
        if omega_norm != 0:
            # See [7]
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            rot_angle = omega_norm * dt
            dRdt = np.eye(3) + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
            rot = dRdt @ rot
        else:
            rot = rot

        # COMPUTING OMEGA UPDATE
        omega_dot = ((1.0 / dynamics.inertia) * (_cross(-omega, dynamics.inertia * omega) + torque))
        omega = omega + dt * omega_dot
        omega = np.clip(omega, a_min=-dynamics.omega_max, a_max=dynamics.omega_max)

        # TRANSLATIONAL DYNAMICS
        # Computing position
        pos = pos + dt * vel
        force = np.matmul(rot, thrust)
        acc = [0., 0., -9.81] + (1.0 / dynamics.mass) * force

        # Computing velocities
        vel = vel + dt * acc
    return pos, vel, rot, omega # What I'm determining to be the state

def real_dynamics(norm_cmds, dynamics, dt, steps=2):
    dt = dt / steps
    norm_cmds = np.asarray(norm_cmds, dtype=np.float64)

    if steps == 1:
        step_outputs = _normalized_to_thrust(norm_cmds, dynamics, steps=2)
        step_outputs = [step_outputs[0]]
    else:
        step_outputs = _normalized_to_thrust(norm_cmds, dynamics, steps=steps)

    rot = np.asarray(dynamics.rot, dtype=np.float64).copy()
    omega = np.asarray(dynamics.omega, dtype=np.float64).copy()
    pos = np.asarray(dynamics.pos, dtype=np.float64).copy()
    vel = np.asarray(dynamics.vel, dtype=np.float64).copy()

    for thrusts, torques, thrust_cmds_damp in step_outputs:
        thrust_torque = np.sum(torques, axis=0)

        # Rotor drag and Rolling forces and moments
        if dynamics.C_rot_drag != 0 or dynamics.C_rot_roll != 0:
            vel_body = rot.T @ vel
            v_rotor = vel_body + _cross_vec_mx4(omega, dynamics.model.prop_pos)
            v_rotor[:, 2] = 0.0

            rotor_drag_fi = -dynamics.C_rot_drag * np.sqrt(thrust_cmds_damp)[:, None] * v_rotor
            rotor_drag_force = np.sum(rotor_drag_fi, axis=0)
            rotor_drag_ti = _cross_mx4(rotor_drag_fi, dynamics.model.prop_pos)
            rotor_drag_torque = np.sum(rotor_drag_ti, axis=0)

            rotor_roll_torque = -dynamics.C_rot_roll * dynamics.prop_ccw[:, None] * np.sqrt(thrust_cmds_damp)[:, None] * v_rotor
            rotor_roll_torque = np.sum(rotor_roll_torque, axis=0)
            rotor_visc_torque = rotor_drag_torque + rotor_roll_torque

            vel_norm = np.linalg.norm(vel_body)
            rdf_norm = np.linalg.norm(rotor_drag_force)
            rdf_norm_clip = np.clip(rdf_norm, a_min=0.0, a_max=vel_norm * dynamics.mass / (2 * dt))
            if rdf_norm > EPS:
                rotor_drag_force = (rotor_drag_force / rdf_norm) * rdf_norm_clip

            rvt_norm = np.linalg.norm(rotor_visc_torque)
            rvt_norm_clipped = np.clip(rvt_norm, a_min=0.0, a_max=np.linalg.norm(omega * dynamics.inertia) / (2 * dt))
            if rvt_norm > EPS:
                rotor_visc_torque = (rotor_visc_torque / rvt_norm) * rvt_norm_clipped
        else:
            rotor_visc_torque = np.zeros(3)
            rotor_drag_force = np.zeros(3)

        torque = thrust_torque + rotor_visc_torque
        thrust = np.array([0.0, 0.0, np.sum(thrusts)], dtype=np.float64)

        omega_vec = np.matmul(rot, omega)
        wx, wy, wz = omega_vec
        omega_norm = np.linalg.norm(omega_vec)
        if omega_norm != 0:
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            rot_angle = omega_norm * dt
            dRdt = np.eye(3) + np.sin(rot_angle) * K + (1.0 - np.cos(rot_angle)) * (K @ K)
            rot = dRdt @ rot

        omega_dot = ((1.0 / dynamics.inertia) * (_cross(-omega, dynamics.inertia * omega) + torque))
        omega_damp_quadratic = np.clip(dynamics.damp_omega_quadratic * omega ** 2, a_min=0.0, a_max=1.0)
        omega = omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
        omega = np.clip(omega, a_min=-dynamics.omega_max, a_max=dynamics.omega_max)

        pos = pos + dt * vel
        force = np.matmul(rot, thrust) + np.matmul(rot, rotor_drag_force)
        acc = GRAVITY_VECTOR + (1.0 / dynamics.mass) * force
        vel = (1.0 - dynamics.vel_damp) * vel + dt * acc

    return pos, vel, rot, omega


# ---------------------------------------------------------------------------
# CBF helpers
# ---------------------------------------------------------------------------

@dataclass
class CBFObstacle:
    position: np.ndarray
    radius: float
    velocity: np.ndarray

def _cbf_h_values(
    pos: np.ndarray,
    vel: np.ndarray,
    rot: np.ndarray,
    obs_pos: np.ndarray,
    radius: float,
    mass: float,
    dt: float,
    u_var,
    u_ref_thrust: np.ndarray):
    """
    """
    dt = dt / 2.0  # each control step is 2 simulator steps
    acc_scale = (1.0 / mass) * np.outer(rot[:, 2], np.ones(4, dtype=np.float64))

    # Predicted next position under optimization variable.
    acc = GRAVITY_VECTOR + acc_scale @ u_var
    next_vel = vel + dt * acc
    next_pos = pos + dt * vel + dt * next_vel

    # Nominal next position around reference thrust (for first-order lower bound).
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64)
    acc_ref = GRAVITY_VECTOR + acc_scale @ u_ref
    next_vel_ref = vel + dt * acc_ref
    next_pos_ref = pos + dt * vel + dt * next_vel_ref

    z = next_pos - obs_pos
    z_ref = np.asarray(next_pos_ref - obs_pos, dtype=np.float64)
    z_ref_norm = max(float(np.linalg.norm(z_ref)), 1e-6)
    grad = z_ref / z_ref_norm
    norm_lb = z_ref_norm + grad @ (z - z_ref)

    h_value = np.linalg.norm(pos - obs_pos) - radius
    h_next = norm_lb - radius
    return h_value, h_next


def _solve_cbf_qp(
    gamma,
    u_ref_thrust: np.ndarray,
    state,
    obstacles, # Contains obstacle positions and radii
    r,
    mass: float,
    dt,
    thrust_bounds: Tuple[np.ndarray, np.ndarray]
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
    pos = state[:3]
    vel = state[3:6]
    rot = state[6:15].reshape(3, 3)

    constraints: List[cp.Constraint] = []
    u_var = cp.Variable(4)
    slack = cp.Variable()

    for obs_idx in range(len(obstacles)):
        obs = obstacles[obs_idx]
        radius = float(obs["radius"])
        obs_pos = np.asarray(obs["position"], dtype=np.float64)
        h_value, h_next = _cbf_h_values(pos, vel, rot, obs_pos, radius, mass, dt, u_var, u_ref)
        constraints.append(h_next - (1 - gamma) * h_value >= r - slack) # Lh = 1

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
        print('OTHER ISSUE', problem.status, 'Lh*r=', r)
        return np.clip(u_ref, u_min, u_max)
    solution = np.array(u_var.value, dtype=np.float64)
    return solution 


def apply_cbf_filter(
    base_action: np.ndarray,
    env_state,
    quad_state,
    obstacles,
    r,
    gamma,
    debug=False
) -> np.ndarray:
    """
    Wrap the raw solo-policy action with the obstacle-based CBF safety filter.

    Parameters
    ----------
    base_action : np.ndarray
        Motor command in [-1, 1]⁴ produced by the solo policy.
    env_state :
        Vectorised environment used to access the simulator dynamics.
    quad_state :
        Packed solo state [pos(3), vel(3), rot(9), omega(3)].
    obstacles :
        Sequence of obstacles with position, radius, velocity.
    r :
        Conformal upper bound on model mismatch used by robust CBF.
    gamma :
        CBF contraction factor in (0, 1].
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
        safe_thrust = _solve_cbf_qp(
            gamma=gamma,
            u_ref_thrust=u_ref_thrust,
            state=quad_state,
            obstacles=obstacles,
            r=r,
            mass=float(dynamics.mass),
            dt=float(env_state.control_dt),
            thrust_bounds=(u_min, u_max),
        )

    # Convert Newton thrust back to the environment's action space.
    safe_normalized = _thrust_to_normalized(safe_thrust, dynamics)
    safe_action = 2.0 * safe_normalized - 1.0
    clipped_action = np.clip(safe_action.astype(np.float32), -1.0, 1.0)
    return clipped_action

def make_cbf_filter(r: float, gamma: float):
    def filter(base_action: np.ndarray, env_state, quad_state, obstacles, debug=False):
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_state,
            quad_state=quad_state,
            obstacles=obstacles,
            r=r,
            gamma=gamma,
            debug=debug,
        )
    return filter
