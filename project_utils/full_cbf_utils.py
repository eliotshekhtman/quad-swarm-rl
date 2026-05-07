from __future__ import annotations

from typing import List, Tuple

import warnings

import cvxpy as cp
import numpy as np

from project_utils.utils import SwarmState

CBF_K4 = 0.5
CBF_K1 = 1
CBF_K0 = 1
CBF_SLACK_WEIGHT = 1.0e4
CBF_RELINEARIZATION_PASSES = 2
EPSILON = 1e-3

GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Motor command conversions
# ---------------------------------------------------------------------------

def _normalized_to_thrust(norm_cmds: np.ndarray, dynamics) -> np.ndarray:
    """
    Convert environment actions in [0, 1] into per-rotor thrust magnitudes.

    The conversion mirrors the simulator's damped motor model so the ECBF QP
    operates in the same thrust space as the policy's low-level commands.
    """
    motor_tau_down = np.asarray(dynamics.motor_tau_down, dtype=np.float64)
    motor_tau = dynamics.motor_tau_up * np.ones([4, ])
    motor_tau[norm_cmds < dynamics.thrust_cmds_damp] = motor_tau_down
    motor_tau[motor_tau > 1.0] = 1.0
    thrust_rot = np.asarray(norm_cmds, dtype=np.float64) ** 0.5
    thrust_rot_damp = motor_tau * (thrust_rot - dynamics.thrust_rot_damp) + dynamics.thrust_rot_damp
    thrust_cmds_damp = thrust_rot_damp ** 2

    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    return thrust_max * dynamics.angvel2thrust(thrust_cmds_damp, linearity=linearity)


def _thrust_to_normalized(thrusts: np.ndarray, dynamics) -> np.ndarray:
    def _invert_single(index: int) -> float:
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            test_norm = np.ones(4, dtype=np.float64) * mid
            val = _normalized_to_thrust(test_norm, dynamics)[index]
            if val < thrusts[index]:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    norm_cmds = np.zeros(4, dtype=np.float64)
    for i in range(4):
        norm_cmds[i] = _invert_single(i)
    return norm_cmds


# ---------------------------------------------------------------------------
# 4th-order ego-only ECBF helpers
# ---------------------------------------------------------------------------

def _ego_ecbf_terms(dynamics) -> dict:
    """
    Ego translational coefficients under zero-order hold.

        x¨   = g + H u
        x⁽³⁾ = J u
        x⁽⁴⁾ = (1ᵀu) (K u + k₀)

    where ``u`` is the 4-vector of ego rotor thrust magnitudes.
    """
    rot = np.asarray(dynamics.rot, dtype=np.float64)
    omega_body = np.asarray(dynamics.omega, dtype=np.float64)
    omega_world = rot @ omega_body
    mass = float(dynamics.mass)
    inertia = np.asarray(dynamics.inertia, dtype=np.float64)
    inv_inertia = 1.0 / inertia
    ones4 = np.ones(4, dtype=np.float64)

    s = rot[:, 2]
    H = np.outer(s / mass, ones4)

    G_tau = np.asarray(dynamics.prop_crossproducts, dtype=np.float64).T
    C_tau = inv_inertia[:, None] * G_tau
    c0 = inv_inertia * _cross(-omega_body, inertia * omega_body)

    j1 = _cross(omega_world, s) / mass
    J = np.outer(j1, ones4)
    K = -(1.0 / mass) * (_skew(s) @ rot @ C_tau)
    k0 = (_cross(rot @ c0, s) + _cross(omega_world, _cross(omega_world, s))) / mass

    return {
        "H": H,
        "J": J,
        "K": K,
        "k0": k0,
    }


def _ecbf_coefficients(
    *,
    solo_pos: np.ndarray,
    solo_vel: np.ndarray,
    teammate_pos: np.ndarray,
    teammate_vel: np.ndarray,
    ego_terms: dict,
    radius: float,
    u_ref_thrust: np.ndarray,
    lambda4: float,
    lambda0: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> Tuple[float, np.ndarray, dict]:
    """
    Compute the affine 4th-order ECBF inequality for one predicted teammate:

        phi_const + phi_gradᵀ u >= -slack

    using the 3D spherical barrier

        h(x, p) = ||x - p|| - r

    where the ego state is controlled by the thrust vector ``u``, while the
    teammate prediction contributes only position and velocity:

        z      = x - p
        ż      = v - ṗ
        z̈(u)   = g + H u
        z⁽³⁾(u) = J u
        z⁽⁴⁾(u) = (1ᵀu) (K u + k₀)

    As in the root-repo obstacle ECBF, ``ḧ`` and ``h⁽³⁾`` are kept exact, while
    only ``h⁽⁴⁾`` is linearised around ``u_ref_thrust``.
    """
    z = np.asarray(solo_pos, dtype=np.float64) - np.asarray(teammate_pos, dtype=np.float64)
    z_dot = np.asarray(solo_vel, dtype=np.float64) - np.asarray(teammate_vel, dtype=np.float64)
    rho = max(float(np.linalg.norm(z)), EPSILON)
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    ones4 = np.ones(4, dtype=np.float64)

    H = ego_terms["H"]
    J = ego_terms["J"]
    K = ego_terms["K"]
    k0 = ego_terms["k0"]

    # A := zᵀ ż
    A = float(z @ z_dot)

    # B(u) := ||ż||² + zᵀ z̈(u) = B_const + g_Bᵀ u
    g_B = H.T @ z
    B_const = float(z_dot @ z_dot + z @ GRAVITY_VECTOR)
    B_ref = B_const + float(g_B @ u_ref)

    # C(u) := 3 żᵀ z̈(u) + zᵀ z⁽³⁾(u) = C_const + g_Cᵀ u
    g_C = 3.0 * (H.T @ z_dot) + J.T @ z
    C_const = float(3.0 * (z_dot @ GRAVITY_VECTOR))
    C_ref = C_const + float(g_C @ u_ref)

    # D(u) := 3 z̈(u)ᵀ z̈(u) + 4 żᵀ z⁽³⁾(u) + zᵀ z⁽⁴⁾(u)
    a_ref = GRAVITY_VECTOR + H @ u_ref
    j_ref = J @ u_ref
    q = K.T @ z
    c = float(z @ k0)
    T_ref = float(ones4 @ u_ref)
    snap_ref = T_ref * (K @ u_ref + k0)
    D_ref = float(3.0 * (a_ref @ a_ref) + 4.0 * (z_dot @ j_ref) + z @ snap_ref)
    grad_D = (
        6.0 * (H.T @ a_ref)
        + 4.0 * (J.T @ z_dot)
        + (float(q @ u_ref) + c) * ones4
        + T_ref * q
    )

    h_value = rho - float(radius)
    h_dot = A / rho

    h_ddot_const = B_const / rho - (A * A) / (rho ** 3)
    h_ddot_grad = g_B / rho
    h_ddot_ref = h_ddot_const + float(h_ddot_grad @ u_ref)

    h_dddot_const = C_const / rho - 3.0 * A * B_const / (rho ** 3) + 3.0 * (A ** 3) / (rho ** 5)
    h_dddot_grad = g_C / rho - 3.0 * A * g_B / (rho ** 3)
    h_dddot_ref = h_dddot_const + float(h_dddot_grad @ u_ref)

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
    h_ddddot_lin_const = h_ddddot_ref - float(h_ddddot_grad @ u_ref)

    phi_const = (
        lambda4 * h_ddddot_lin_const
        + lambda3 * h_dddot_const
        + lambda2 * h_ddot_const
        + lambda1 * h_dot
        + lambda0 * h_value
    )
    phi_grad = lambda4 * h_ddddot_grad + lambda3 * h_dddot_grad + lambda2 * h_ddot_grad

    phi_ref = (
        lambda4 * h_ddddot_ref
        + lambda3 * h_dddot_ref
        + lambda2 * h_ddot_ref
        + lambda1 * h_dot
        + lambda0 * h_value
    )
    debug_terms = {
        "rho": rho,
        "h0": h_value,
        "h1": h_dot,
        "h2_ref": h_ddot_ref,
        "h3_ref": h_dddot_ref,
        "h4_ref": h_ddddot_ref,
        "phi_ref": phi_ref,
    }
    return phi_const, phi_grad, debug_terms


def _solve_cbf_qp_single(
    *,
    u_ref_thrust: np.ndarray,
    dynamics,
    swarm_state: SwarmState,
    radii: np.ndarray,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
    debug=False,
) -> np.ndarray:
    """
    Build and solve one affine approximation of the ego-only multi-neighbor ECBF QP.
    """
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    u_min, u_max = thrust_bounds
    u_min = np.asarray(u_min, dtype=np.float64).reshape(-1)
    u_max = np.asarray(u_max, dtype=np.float64).reshape(-1)

    solo_pos = np.asarray(swarm_state.positions[-1], dtype=np.float64)
    solo_vel = np.asarray(swarm_state.velocities[-1], dtype=np.float64)
    num_multi_agents = len(radii)
    ego_terms = _ego_ecbf_terms(dynamics)

    alpha = float(CBF_K1)
    lambda4 = float(CBF_K4)
    lambda3 = 4.0 * alpha
    lambda2 = 6.0 * (alpha ** 2)
    lambda1 = 4.0 * (alpha ** 3)
    lambda0 = float(CBF_K0) * (alpha ** 4)

    constraints: List[cp.Constraint] = []
    u_var = cp.Variable(4)
    slack = cp.Variable()

    phi_list = []
    h1_list = []
    h_list = []

    for teammate_idx in range(num_multi_agents):
        radius = float(radii[teammate_idx])
        if radius < 0.0:
            continue
        teammate_pos = swarm_state.positions[teammate_idx]
        teammate_vel = swarm_state.velocities[teammate_idx]
        phi_const, phi_grad, debug_terms = _ecbf_coefficients(
            solo_pos=solo_pos,
            solo_vel=solo_vel,
            teammate_pos=teammate_pos,
            teammate_vel=teammate_vel,
            ego_terms=ego_terms,
            radius=radius,
            u_ref_thrust=u_ref,
            lambda4=lambda4,
            lambda0=lambda0,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
        )
        phi_expr = phi_const + phi_grad @ u_var
        constraints.append(phi_expr >= -slack)
        phi_list.append((teammate_idx, phi_expr))
        h1_list.append((teammate_idx, debug_terms["h1"]))
        h_list.append((teammate_idx, debug_terms["h0"]))

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
        for h_data, h1_data, phi_data in zip(h_list, h1_list, phi_list):
            teammate_idx = h_data[0]
            print("neighbor:", teammate_idx)
            print("h:", h_data[1])
            print("h_dot:", h1_data[1])
            print("phi:", phi_data[1].value)

    solution = np.array(u_var.value, dtype=np.float64)
    if debug:
        return solution, h_list, h1_list, phi_list
    return solution


def _solve_cbf_qp(
    *,
    u_ref_thrust: np.ndarray,
    dynamics,
    swarm_state: SwarmState,
    radii: np.ndarray,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
    relinearization_passes: int = 1,
    debug=False,
) -> np.ndarray:
    """
    Solve the ego-only ECBF QP, optionally re-linearising h⁽⁴⁾ around the latest iterate.
    """
    passes = int(relinearization_passes)
    if passes <= 0:
        raise ValueError("relinearization_passes must be >= 1")

    iterate = np.asarray(u_ref_thrust, dtype=np.float64).reshape(-1)
    final_debug = None

    for _ in range(passes):
        outputs = _solve_cbf_qp_single(
            u_ref_thrust=iterate,
            dynamics=dynamics,
            swarm_state=swarm_state,
            radii=radii,
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
    radii: np.ndarray,
    env_state,
    swarm_state: SwarmState,
    use_repeated_linearization: bool = False,
    debug=False,
) -> np.ndarray:
    """
    Wrap the raw ego-policy action with a 4th-order ECBF safety filter.

    ``swarm_state`` is expected to contain predicted teammate positions and
    velocities in entries ``[:N]`` and the actual ego state at index ``N``.
    """
    quad = env_state.envs[-1]
    dynamics = quad.dynamics

    base_action = np.asarray(base_action, dtype=np.float64)
    normalized = np.clip(0.5 * (base_action + 1.0), 0.0, 1.0)
    u_ref_thrust = _normalized_to_thrust(normalized, dynamics)

    u_min = np.zeros(4, dtype=np.float64)
    u_max = np.asarray(dynamics.thrust_max, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _solve_cbf_qp(
            u_ref_thrust=u_ref_thrust,
            dynamics=dynamics,
            swarm_state=swarm_state,
            radii=radii,
            thrust_bounds=(u_min, u_max),
            relinearization_passes=(CBF_RELINEARIZATION_PASSES if use_repeated_linearization else 1),
            debug=debug,
        )

    if debug:
        safe_thrust, h_list, h1_list, phi_list = outputs
    else:
        safe_thrust = outputs

    safe_normalized = _thrust_to_normalized(safe_thrust, dynamics)
    safe_action = 2.0 * safe_normalized - 1.0
    clipped_action = np.clip(safe_action.astype(np.float32), -1.0, 1.0)
    if debug:
        return clipped_action, h_list, h1_list, phi_list
    return clipped_action


def make_cbf_filter(radii: np.ndarray, use_repeated_linearization: bool = False):
    def filter(base_action: np.ndarray, env_state, swarm_state: SwarmState, debug=False):
        return apply_cbf_filter(
            base_action=base_action,
            radii=radii,
            env_state=env_state,
            swarm_state=swarm_state,
            use_repeated_linearization=use_repeated_linearization,
            debug=debug,
        )

    return filter
