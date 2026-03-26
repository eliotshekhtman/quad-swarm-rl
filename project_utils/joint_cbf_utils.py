from __future__ import annotations

from typing import List, Tuple

import warnings
import cvxpy as cp
import numpy as np

from project_utils.utils import *

CBF_K4 = 1
CBF_K1 = 4
CBF_K0 = 1
CBF_SLACK_WEIGHT = 1.0e4
CBF_RELINEARIZATION_PASSES = 2
EPSILON = 1e-3

GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81], dtype=np.float64)
CYLINDER_PROJECTION = np.diag([1.0, 1.0, 0.0]).astype(np.float64)


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


def _skew(v: np.ndarray) -> np.ndarray:
    v_arr = np.asarray(v, dtype=np.float64)
    x, y, z = v_arr
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _project_xy(v: np.ndarray) -> np.ndarray:
    return CYLINDER_PROJECTION @ np.asarray(v, dtype=np.float64)


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
    dt = dt / 2.0  # control_dt is 2*dt since there are 2 env steps every control step
    thrusts, torques = _normalized_to_thrust(norm_cmds, dynamics)
    rot = dynamics.rot
    omega = dynamics.omega
    pos = dynamics.pos
    vel = dynamics.vel

    for _ in range(steps):
        torque = np.sum(torques, axis=0)
        thrust = np.array([0, 0, np.sum(thrusts)])

        omega_vec = np.matmul(rot, omega)
        wx, wy, wz = omega_vec
        omega_norm = np.linalg.norm(omega_vec)
        if omega_norm != 0:
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            rot_angle = omega_norm * dt
            dRdt = np.eye(3) + np.sin(rot_angle) * K + (1.0 - np.cos(rot_angle)) * (K @ K)
            rot = dRdt @ rot

        omega_dot = ((1.0 / dynamics.inertia) * (_cross(-omega, dynamics.inertia * omega) + torque))
        omega = omega + dt * omega_dot
        omega = np.clip(omega, a_min=-dynamics.omega_max, a_max=dynamics.omega_max)

        pos = pos + dt * vel
        force = np.matmul(rot, thrust)
        acc = [0.0, 0.0, -9.81] + (1.0 / dynamics.mass) * force
        vel = vel + dt * acc
    return pos, vel, rot, omega


def real_dynamics(norm_cmds, dynamics, dt, steps=2):
    dt = dt / 2.0  # control_dt is 2*dt since there are 2 env steps every control step
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
# ECBF helpers
# ---------------------------------------------------------------------------

def _agent_ecbf_terms(dynamics) -> dict:
    """
    Per-agent coefficients for the zero-order-hold translational derivatives
    used by the pairwise 4th-order ECBF.

        x¨   = g + H u
        x⁽³⁾ = J u
        x⁽⁴⁾ = (1ᵀu) (K u + k₀)

    where ``u`` is the 4-vector of rotor thrust magnitudes.
    """
    rot = np.asarray(dynamics.rot, dtype=np.float64)
    omega_body = np.asarray(dynamics.omega, dtype=np.float64)
    omega_world = rot @ omega_body
    mass = float(dynamics.mass)
    inertia = np.asarray(dynamics.inertia, dtype=np.float64)
    inv_inertia = 1.0 / inertia

    s = rot[:, 2]
    G_tau = np.asarray(dynamics.prop_crossproducts, dtype=np.float64).T
    C_tau = inv_inertia[:, None] * G_tau
    c0 = inv_inertia * _cross(-omega_body, inertia * omega_body)

    j1 = _cross(omega_world, s) / mass
    K = -(1.0 / mass) * (_skew(s) @ rot @ C_tau)
    k0 = (_cross(rot @ c0, s) + _cross(omega_world, _cross(omega_world, s))) / mass

    return {
        "mass": mass,
        "s": s,
        "j1": j1,
        "K": K,
        "k0": k0,
    }


def _ecbf_coefficients(
    *,
    i: int,
    j: int,
    swarm_state: SwarmState,
    agent_terms: List[dict],
    separation_radius: float,
    u_ref_thrust: np.ndarray,
    lambda4: float,
    lambda0: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> Tuple[float, np.ndarray, dict]:
    """
    Compute the affine pairwise ECBF inequality for agents ``i`` and ``j``:

        phi_const + phi_grad_localᵀ [u_i; u_j] >= -slack

    using the 4th-order continuous-time barrier

        h(x_i, x_j) = ||x_i - x_j|| - separation_radius.

    The exact relative derivatives under zero-order hold are

        z     = x_i - x_j
        ż     = v_i - v_j
        z̈(u)  = H_i u_i + H_j u_j
        z⁽³⁾(u)= J_i u_i + J_j u_j
        z⁽⁴⁾(u)= (1ᵀu_i)(K_i u_i + k₀_i) - (1ᵀu_j)(K_j u_j + k₀_j).

    As in the single-agent obstacle ECBF, ``ḧ`` and ``h⁽³⁾`` are kept exact,
    while only ``h⁽⁴⁾`` is linearised around ``u_ref_thrust``.
    """
    positions = np.asarray(swarm_state.positions, dtype=np.float64)
    velocities = np.asarray(swarm_state.velocities, dtype=np.float64)
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    ones4 = np.ones(4, dtype=np.float64)

    z = positions[i] - positions[j]
    v = velocities[i] - velocities[j]
    rho = max(float(np.linalg.norm(z)), EPSILON)

    terms_i = agent_terms[i]
    terms_j = agent_terms[j]

    H_i = np.outer(terms_i["s"] / terms_i["mass"], ones4)
    H_j = -np.outer(terms_j["s"] / terms_j["mass"], ones4)
    J_i = np.outer(terms_i["j1"], ones4)
    J_j = -np.outer(terms_j["j1"], ones4)

    u_ref_i = u_ref[4 * i: 4 * (i + 1)]
    u_ref_j = u_ref[4 * j: 4 * (j + 1)]
    u_ref_local = np.concatenate([u_ref_i, u_ref_j], axis=0)

    a_ref = H_i @ u_ref_i + H_j @ u_ref_j
    j_ref = J_i @ u_ref_i + J_j @ u_ref_j

    T_ref_i = float(ones4 @ u_ref_i)
    T_ref_j = float(ones4 @ u_ref_j)
    snap_ref = (
        T_ref_i * (terms_i["K"] @ u_ref_i + terms_i["k0"])
        - T_ref_j * (terms_j["K"] @ u_ref_j + terms_j["k0"])
    )

    A = float(z @ v)

    g_B_i = H_i.T @ z
    g_B_j = H_j.T @ z
    g_B = np.concatenate([g_B_i, g_B_j], axis=0)
    B_const = float(v @ v)
    B_ref = B_const + float(g_B_i @ u_ref_i) + float(g_B_j @ u_ref_j)

    g_C_i = 3.0 * (H_i.T @ v) + J_i.T @ z
    g_C_j = 3.0 * (H_j.T @ v) + J_j.T @ z
    g_C = np.concatenate([g_C_i, g_C_j], axis=0)
    C_ref = float(g_C_i @ u_ref_i) + float(g_C_j @ u_ref_j)

    q_i = terms_i["K"].T @ z
    q_j = terms_j["K"].T @ z

    D_ref = float(3.0 * (a_ref @ a_ref) + 4.0 * (v @ j_ref) + (z @ snap_ref))

    grad_D_i = (
        6.0 * (H_i.T @ a_ref)
        + (4.0 * float(v @ terms_i["j1"]) + float(z @ terms_i["k0"])) * ones4
        + float(q_i @ u_ref_i) * ones4
        + T_ref_i * q_i
    )
    grad_D_j = (
        6.0 * (H_j.T @ a_ref)
        - (4.0 * float(v @ terms_j["j1"]) + float(z @ terms_j["k0"])) * ones4
        - float(q_j @ u_ref_j) * ones4
        - T_ref_j * q_j
    )
    grad_D = np.concatenate([grad_D_i, grad_D_j], axis=0)

    h_value = rho - float(separation_radius)
    h_dot = A / rho

    h_ddot_const = B_const / rho - (A * A) / (rho ** 3)
    h_ddot_grad = g_B / rho
    h_ddot_ref = h_ddot_const + float(h_ddot_grad @ u_ref_local)

    h_dddot_const = -3.0 * A * B_const / (rho ** 3) + 3.0 * (A ** 3) / (rho ** 5)
    h_dddot_grad = g_C / rho - 3.0 * A * g_B / (rho ** 3)
    h_dddot_ref = h_dddot_const + float(h_dddot_grad @ u_ref_local)

    h_ddddot_ref = (
        D_ref / rho
        - 4.0 * A * C_ref / (rho ** 3)
        - 3.0 * (B_ref ** 2) / (rho ** 3)
        + 18.0 * (A ** 2) * B_ref / (rho ** 5)
        - 15.0 * (A ** 4) / (rho ** 7)
    )
    h_ddddot_grad = (
        grad_D / rho
        - 4.0 * A * g_C / (rho ** 3)
        - 6.0 * B_ref * g_B / (rho ** 3)
        + 18.0 * (A ** 2) * g_B / (rho ** 5)
    )
    h_ddddot_lin_const = h_ddddot_ref - float(h_ddddot_grad @ u_ref_local)

    phi_const = (
        lambda4 * h_ddddot_lin_const
        + lambda3 * h_dddot_const
        + lambda2 * h_ddot_const
        + lambda1 * h_dot
        + lambda0 * h_value
    )
    phi_grad_local = lambda4 * h_ddddot_grad + lambda3 * h_dddot_grad + lambda2 * h_ddot_grad

    phi_ref = (
        lambda4 * h_ddddot_ref
        + lambda3 * h_dddot_ref
        + lambda2 * h_ddot_ref
        + lambda1 * h_dot
        + lambda0 * h_value
    )

    debug_terms = {
        "pair": (i, j),
        "rho": rho,
        "h0": h_value,
        "h1": h_dot,
        "h2_ref": h_ddot_ref,
        "h3_ref": h_dddot_ref,
        "h4_ref": h_ddddot_ref,
        "phi_ref": phi_ref,
    }
    return phi_const, phi_grad_local, debug_terms


def _solve_cbf_qp_single(
    *,
    u_ref_thrust: np.ndarray,
    swarm_state: SwarmState,
    agent_terms: List[dict],
    separation_radius: float,
    r_mismatch: float,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
    debug=False,
) -> np.ndarray:
    """
    Build and solve one affine approximation of the joint pairwise ECBF QP.

    Decision variables
    ------------------
    - ``u`` ∈ ℝ⁴ᴺ : stacked per-rotor thrusts for all agents.
    - ``slack`` ≥ 0 : shared softening variable.

    Constraints
    -----------
    For every pair ``(i, j)``,

        phi_ij_lin(u) >= lambda1 * r_mismatch - slack

    where ``phi_ij_lin`` is the affine re-linearisation of the exact 4th-order
    pairwise ECBF at the current iterate.
    """
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    u_min = np.asarray(thrust_bounds[0], dtype=np.float64).reshape(-1)
    u_max = np.asarray(thrust_bounds[1], dtype=np.float64).reshape(-1)
    num_agents = int(swarm_state.positions.shape[0])
    expected_dim = 4 * num_agents
    if u_ref.shape[0] != expected_dim:
        raise ValueError(f"u_ref_thrust has shape {u_ref.shape}, expected ({expected_dim},)")
    if u_min.shape[0] != expected_dim or u_max.shape[0] != expected_dim:
        raise ValueError(
            f"thrust bounds must have shape ({expected_dim},); got {u_min.shape} and {u_max.shape}"
        )

    alpha = float(CBF_K1)
    lambda4 = float(CBF_K4)
    lambda3 = 4.0 * alpha
    lambda2 = 6.0 * (alpha ** 2)
    lambda1 = 4.0 * (alpha ** 3)
    lambda0 = float(CBF_K0) * (alpha ** 4)

    constraints: List[cp.Constraint] = []
    u_var = cp.Variable(expected_dim)
    slack = cp.Variable()

    phi_list = []
    h1_list = []
    h_list = []

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            phi_const, phi_grad_local, debug_terms = _ecbf_coefficients(
                i=i,
                j=j,
                swarm_state=swarm_state,
                agent_terms=agent_terms,
                separation_radius=separation_radius,
                u_ref_thrust=u_ref,
                lambda4=lambda4,
                lambda0=lambda0,
                lambda1=lambda1,
                lambda2=lambda2,
                lambda3=lambda3,
            )

            phi_grad = np.zeros(expected_dim, dtype=np.float64)
            phi_grad[4 * i: 4 * (i + 1)] = phi_grad_local[:4]
            phi_grad[4 * j: 4 * (j + 1)] = phi_grad_local[4:]

            phi_expr = phi_const + phi_grad @ u_var
            constraints.append(phi_expr >= lambda1 * float(r_mismatch) - slack)
            phi_list.append(phi_expr)
            h1_list.append((debug_terms["pair"], debug_terms["h1"]))
            h_list.append((debug_terms["pair"], debug_terms["h0"]))

    if len(constraints) == 0:
        clipped = np.clip(u_ref, u_min, u_max)
        if debug:
            return clipped, h_list, h1_list, phi_list
        return clipped

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
            approx = u_ref
        clipped = np.clip(np.asarray(approx, dtype=np.float64), u_min, u_max)
        print("QP timed out; returning last iterate:", clipped)
        if debug:
            return clipped, h_list, h1_list, phi_list
        return clipped

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print("OTHER ISSUE", problem.status)
        clipped = np.clip(u_ref, u_min, u_max)
        if debug:
            return clipped, h_list, h1_list, phi_list
        return clipped

    if debug:
        for pair_data, h_data, phi_expr in zip(h_list, h1_list, phi_list):
            pair = pair_data[0]
            print("pair:", pair)
            print("h:", pair_data[1])
            print("h_dot:", h_data[1])
            print("phi:", phi_expr.value)

    solution = np.array(u_var.value, dtype=np.float64)
    if debug:
        return solution, h_list, h1_list, phi_list
    return solution


def _solve_cbf_qp(
    *,
    u_ref_thrust: np.ndarray,
    swarm_state: SwarmState,
    agent_terms: List[dict],
    separation_radius: float,
    r_mismatch: float,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
    relinearization_passes: int = 1,
    debug=False,
) -> np.ndarray:
    """
    Solve the joint pairwise ECBF QP, optionally re-linearising h⁽⁴⁾ around the
    latest stacked thrust iterate.
    """
    passes = int(relinearization_passes)
    if passes <= 0:
        raise ValueError("relinearization_passes must be >= 1")

    iterate = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    final_debug = None

    for _ in range(passes):
        outputs = _solve_cbf_qp_single(
            u_ref_thrust=iterate,
            swarm_state=swarm_state,
            agent_terms=agent_terms,
            separation_radius=separation_radius,
            r_mismatch=r_mismatch,
            thrust_bounds=thrust_bounds,
            debug=debug,
        )
        if debug:
            iterate, h_list, h1_list, phi_list = outputs
            final_debug = (h_list, h1_list, phi_list)
        else:
            iterate = outputs
        iterate = np.asarray(iterate, dtype=np.float64).reshape(-1)

    if debug:
        h_list, h1_list, phi_list = final_debug
        return iterate, h_list, h1_list, phi_list
    return iterate


def apply_cbf_filter(
    base_action: np.ndarray,
    env_state,
    r,
    separation_radius: float,
    gamma=None,
    use_repeated_linearization: bool = False,
    debug=False,
) -> np.ndarray:
    """
    Wrap raw joint swarm actions with the pairwise 4th-order ECBF safety filter.

    Parameters
    ----------
    base_action : np.ndarray
        Motor commands in ``[-1, 1]^{N x 4}`` (or ``(4,)`` for ``N=1``).
    env_state :
        Vectorised environment used to access the simulator dynamics.
    r :
        Conformal mismatch margin. This enters as ``lambda1 * r`` on the right
        hand side of every pairwise ECBF inequality.
    separation_radius :
        Desired pairwise Euclidean separation distance.
    gamma :
        Accepted for API compatibility with the discrete-time joint CBF wrappers.
        It is intentionally unused by this continuous-time 4th-order ECBF.
    """
    _ = gamma
    actions = np.asarray(base_action, dtype=np.float64)
    squeeze_out = False
    if actions.ndim == 1:
        actions = actions.reshape(1, 4)
        squeeze_out = True

    num_agents = len(env_state.envs)
    if actions.shape != (num_agents, 4):
        raise ValueError(f"base_action shape {actions.shape} does not match expected ({num_agents}, 4)")

    swarm_state = get_swarm_state(env_state)
    agent_terms = [_agent_ecbf_terms(quad.dynamics) for quad in env_state.envs]

    u_refs = []
    u_min_blocks = []
    u_max_blocks = []
    for agent_idx, quad in enumerate(env_state.envs):
        dynamics = quad.dynamics
        normalized = np.clip(0.5 * (actions[agent_idx] + 1.0), 0.0, 1.0)
        u_ref_thrust, _ = _normalized_to_thrust(normalized, dynamics)
        u_refs.append(np.asarray(u_ref_thrust, dtype=np.float64))
        u_min_blocks.append(np.zeros(4, dtype=np.float64))
        u_max_blocks.append(np.asarray(dynamics.thrust_max, dtype=np.float64))

    u_ref_concat = np.concatenate(u_refs, axis=0)
    u_min = np.concatenate(u_min_blocks, axis=0)
    u_max = np.concatenate(u_max_blocks, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _solve_cbf_qp(
            u_ref_thrust=u_ref_concat,
            swarm_state=swarm_state,
            agent_terms=agent_terms,
            separation_radius=float(separation_radius),
            r_mismatch=float(r),
            thrust_bounds=(u_min, u_max),
            relinearization_passes=(CBF_RELINEARIZATION_PASSES if use_repeated_linearization else 1),
            debug=debug,
        )

    if debug:
        safe_thrust_concat, h_list, h1_list, phi_list = outputs
    else:
        safe_thrust_concat = outputs

    safe_actions = np.zeros((num_agents, 4), dtype=np.float32)
    for agent_idx, quad in enumerate(env_state.envs):
        dynamics = quad.dynamics
        thrust_block = safe_thrust_concat[4 * agent_idx: 4 * (agent_idx + 1)]
        safe_normalized = _thrust_to_normalized(thrust_block, dynamics)
        safe_action = 2.0 * safe_normalized - 1.0
        safe_actions[agent_idx] = np.clip(safe_action.astype(np.float32), -1.0, 1.0)

    if squeeze_out:
        safe_actions = safe_actions[0]

    if debug:
        return safe_actions, h_list, h1_list, phi_list
    return safe_actions


def make_cbf_filter(
    r: float,
    separation_radius: float,
    gamma: float,
    use_repeated_linearization: bool = False,
):
    def filter(base_action: np.ndarray, env_state, debug=False):
        return apply_cbf_filter(
            base_action=base_action,
            env_state=env_state,
            r=r,
            separation_radius=separation_radius,
            gamma=gamma,
            use_repeated_linearization=use_repeated_linearization,
            debug=debug,
        )

    return filter
