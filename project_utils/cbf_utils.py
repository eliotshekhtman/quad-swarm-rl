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
    dt = dt / 2.0 # control_dt is 2*dt since there are 2 env steps every control step
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
            # u = omega_vec / omega_norm <-- Euler vector, below skew-sym K
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            rot_angle = omega_norm * dt
            # I + sin(theta) @ u + (1 - cos(theta)) * [u]^2 ?
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
    dt = dt / 2.0 # control_dt is 2*dt since there are 2 env steps every control step
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
# ECBF helpers
# ---------------------------------------------------------------------------

def _ecbf_coefficients(
    *,
    dynamics,
    obs_pos: np.ndarray,
    radius: float,
    u_ref_thrust: np.ndarray,
    lambda4: float,
    lambda0: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> Tuple[float, np.ndarray, dict]:
    """
    Compute the affine ECBF inequality ``phi_const + phi_grad^T u >= -slack``.

    The exact obstacle geometry is an infinite vertical cylinder, so the
    barrier ignores the obstacle z coordinate by projecting all barrier-space
    quantities into the xy plane:

        P      = diag(1, 1, 0)
        z      = P (x - p)
        h(x)   = ||z|| - r.

    Equivalently, the dynamics keep their full 3D form, but every occurrence of
    ``(x - p)``, ``ẋ``, ``x⁽²⁾``, ``x⁽³⁾`` and ``x⁽⁴⁾`` inside the barrier
    derivatives is projected by ``P`` before taking norms or inner products.

    The exact projected barrier is

        h(x) = ||P (x - p)|| - r,

    and the exact 4th-order ECBF expression is

        phi(u) = λ₄ h⁽⁴⁾(u) + λ₃ h⁽³⁾(u) + λ₂ h⁽²⁾(u) + λ₁ h⁽¹⁾ + λ₀ h.

    Under zero-order hold the exact ``h⁽⁴⁾(u)`` is nonlinear in the rotor
    thrusts ``u`` because it contains both ``(1ᵀu)²`` and
    ``(1ᵀu) (qᵀu)``.  To keep the safety filter a QP, we linearise only the
    4th-derivative term around the reference thrust ``u_ref_thrust``:

        h⁽⁴⁾(u) ≈ h⁽⁴⁾(u_ref) + ∇h⁽⁴⁾(u_ref)ᵀ (u - u_ref).

    The lower-order terms ``h``, ``h⁽¹⁾``, ``h⁽²⁾`` and ``h⁽³⁾`` are kept exact.
    """
    # x := solo quad position in world coordinates.
    solo_pos = np.asarray(dynamics.pos, dtype=np.float64)
    # ẋ := solo quad linear velocity in world coordinates.
    solo_vel = np.asarray(dynamics.vel, dtype=np.float64)
    # ω_b := body-frame angular velocity from the simulator state.
    solo_omg = np.asarray(dynamics.omega, dtype=np.float64)
    # R := body-to-world rotation matrix from the simulator state.
    solo_rot = np.asarray(dynamics.rot, dtype=np.float64)
    # u_ref := reference per-rotor thrusts used for the Taylor linearisation.
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64)

    # m := quadrotor mass.
    mass = float(dynamics.mass)
    # I := diagonal inertia entries so that I^{-1} is elementwise inverse.
    inertia = np.asarray(dynamics.inertia, dtype=np.float64)
    # I^{-1} := inverse inertia used in ω̇ = I^{-1}(τ - ω × (Iω)).
    inv_inertia = 1.0 / inertia
    # 1 := all-ones vector so that 1ᵀu is the total thrust magnitude.
    ones4 = np.ones(4, dtype=np.float64)

    # p_full := raw obstacle centre from the environment.  Its z coordinate is
    # ignored by the cylindrical barrier because the obstacle extends infinitely
    # along the z axis.
    obs_pos_arr = np.asarray(obs_pos, dtype=np.float64)
    # z_full := x - p is the full 3D relative position before projection.
    z_full = solo_pos - obs_pos_arr
    # P := diag(1, 1, 0) projects vectors into the xy plane.
    proj = CYLINDER_PROJECTION
    # z := P (x - p) is the cylindrical barrier-space relative position.
    z = proj @ z_full
    # ρ := ||P (x - p)|| is the horizontal distance to the infinite cylinder.
    rho = max(float(np.linalg.norm(z)), EPSILON)
    # v := ẋ is shorthand for the current linear velocity.
    v = solo_vel
    # v_proj := P ẋ is the velocity projected into the barrier geometry.
    v_proj = proj @ v
    # g := gravity vector in world coordinates.
    g = GRAVITY_VECTOR
    # g_proj := P g is the projected gravity contribution.  For vertical gravity
    # this is identically zero, but we keep the projection explicit to mirror
    # the projected derivative derivation.
    g_proj = proj @ g
    # s := Re₃ is the world-frame thrust axis (the third column of R).
    s = solo_rot[:, 2]
    # s_proj := P Re₃ is the thrust axis projected into the barrier geometry.
    s_proj = proj @ s
    # ω_w := R ω_b is the world-frame angular velocity used in Ṙ = ω_w× R.
    omega_world = solo_rot @ solo_omg

    # G_τ := maps per-rotor thrust magnitudes to body torques from arm geometry only.
    G_tau = np.asarray(dynamics.prop_crossproducts, dtype=np.float64).T
    # C_τ := I^{-1} G_τ maps per-rotor thrust magnitudes to body angular acceleration.
    C_tau = inv_inertia[:, None] * G_tau
    # c₀ := I^{-1}(-ω × (Iω)) is the body angular-acceleration drift term.
    c0 = inv_inertia * _cross(-solo_omg, inertia * solo_omg)

    # j₁ := (ω_w × s) / m so that x⁽³⁾ = (1ᵀu) j₁ under zero-order hold.
    j1 = _cross(omega_world, s) / mass
    # j₁_proj := P j₁ is the projected coefficient in the cylindrical barrier.
    j1_proj = proj @ j1
    # K := -(1/m) [s]_× R C_τ so that x⁽⁴⁾ contains the bilinear term (1ᵀu) K u.
    K = -(1.0 / mass) * (_skew(s) @ solo_rot @ C_tau)
    # k₀ := state-only part of x⁽⁴⁾ / (1ᵀu) from ω̇ drift and ω × (ω × s).
    k0 = (_cross(solo_rot @ c0, s) + _cross(omega_world, _cross(omega_world, s))) / mass
    # k₀_proj := P k₀ is the projected state-only coefficient in x⁽⁴⁾.
    k0_proj = proj @ k0
    # q := Kᵀ P (x - p) so that zᵀ(P K u) = qᵀu in the cylindrical barrier.
    q = K.T @ z

    # A := zᵀ(P ẋ) = (P (x - p))ᵀ (P ẋ) is the scalar used in h⁽¹⁾.
    A = float(z @ v_proj)
    # B₀ := (P ẋ)ᵀ(P ẋ) + zᵀ(P g) is the state-only part of
    # B(u) = (P ẋ)ᵀ(P ẋ) + zᵀ(P x⁽²⁾).
    B0 = float(v_proj @ v_proj + z @ g_proj)
    # β := zᵀ(P Re₃) / m is the coefficient of total thrust T = 1ᵀu in B(u).
    beta = float(z @ s_proj) / mass
    # C₀ := 3 (P ẋ)ᵀ(P g) is the state-only part of
    # C(u) = 3 (P ẋ)ᵀ(P x⁽²⁾) + zᵀ(P x⁽³⁾).
    C0 = float(3.0 * (v_proj @ g_proj))
    # γ := 3 (P ẋ)ᵀ(P Re₃) / m + zᵀ(P j₁) is the coefficient of T in C(u).
    gamma = 3.0 * float(v_proj @ s_proj) / mass + float(z @ j1_proj)
    # D₀ := 3 (P g)ᵀ(P g) is the state-only part of
    # D(u) = 3 (P x⁽²⁾)ᵀ(P x⁽²⁾) + 4 (P ẋ)ᵀ(P x⁽³⁾) + zᵀ(P x⁽⁴⁾).
    D0 = float(3.0 * (g_proj @ g_proj))
    # δ := coefficient of total thrust T in D(u).
    delta = 6.0 * float(g_proj @ s_proj) / mass + 4.0 * float(v_proj @ j1_proj) + float(z @ k0_proj)
    # ε := coefficient of T² in
    #   3 (P x⁽²⁾)ᵀ(P x⁽²⁾)
    #     = 3 (P g + T P Re₃ / m)ᵀ(P g + T P Re₃ / m).
    # Because the cylindrical barrier ignores z, this depends on the projected
    # thrust-axis norm ||P Re₃||² rather than the full 3D unit norm of Re₃.
    epsilon = 3.0 * float(s_proj @ s_proj) / (mass * mass)

    # T_ref := 1ᵀu_ref is the total reference thrust.
    T_ref = float(ones4 @ u_ref)
    # B_ref := B(u_ref) = B₀ + β T_ref.
    B_ref = B0 + beta * T_ref
    # C_ref := C(u_ref) = C₀ + γ T_ref.
    C_ref = C0 + gamma * T_ref
    # D_ref := D(u_ref) = D₀ + δ T_ref + ε T_ref² + T_ref (qᵀu_ref).
    D_ref = D0 + delta * T_ref + epsilon * (T_ref ** 2) + T_ref * float(q @ u_ref)

    # h := ||P (x - p)|| - r is the horizontal clearance to the infinite cylinder.
    h_value = rho - radius
    # ḣ := A / ρ.
    h_dot = A / rho
    # ḧ_ref := B(u_ref) / ρ - A² / ρ³.
    h_ddot_ref = B_ref / rho - (A * A) / (rho ** 3)
    # h⁽³⁾_ref := C(u_ref) / ρ - 3 A B(u_ref) / ρ³ + 3 A³ / ρ⁵.
    h_dddot_ref = C_ref / rho - 3.0 * A * B_ref / (rho ** 3) + 3.0 * (A ** 3) / (rho ** 5)

    # ḧ(u) is already affine, so ḧ(u) = ḧ_const + ḧ_gradᵀ u exactly.
    h_ddot_const = B0 / rho - (A * A) / (rho ** 3)
    # ∇ḧ = (β / ρ) 1 because only the total thrust T = 1ᵀu enters ḧ.
    h_ddot_grad = (beta / rho) * ones4

    # h⁽³⁾(u) is also affine, so h⁽³⁾(u) = h⁽³⁾_const + h⁽³⁾_gradᵀ u exactly.
    h_dddot_const = C0 / rho - 3.0 * A * B0 / (rho ** 3) + 3.0 * (A ** 3) / (rho ** 5)
    # ∇h⁽³⁾ = (γ / ρ - 3 A β / ρ³) 1 because only the total thrust enters h⁽³⁾.
    h_dddot_grad = (gamma / rho - 3.0 * A * beta / (rho ** 3)) * ones4

    # Exact nonlinear h⁽⁴⁾(u):
    #   h⁽⁴⁾(u) = D(u)/ρ - 4 A C(u)/ρ³ - 3 B(u)²/ρ³ + 18 A² B(u)/ρ⁵ - 15 A⁴/ρ⁷.
    h_ddddot_ref = (
        D_ref / rho
        - 4.0 * A * C_ref / (rho ** 3)
        - 3.0 * (B_ref ** 2) / (rho ** 3)
        + 18.0 * (A ** 2) * B_ref / (rho ** 5)
        - 15.0 * (A ** 4) / (rho ** 7)
    )

    # ∇D(u_ref) = (δ + 2 ε T_ref + qᵀu_ref) 1 + T_ref q.
    grad_D_ref = (
        (delta + 2.0 * epsilon * T_ref + float(q @ u_ref)) * ones4
        + T_ref * q
    )
    # ∇h⁽⁴⁾(u_ref) from differentiating the exact nonlinear h⁽⁴⁾(u) above.
    h_ddddot_grad = (
        grad_D_ref / rho
        - 4.0 * A * gamma * ones4 / (rho ** 3)
        - 6.0 * B_ref * beta * ones4 / (rho ** 3)
        + 18.0 * (A ** 2) * beta * ones4 / (rho ** 5)
    )
    # First-order Taylor model:
    #   h⁽⁴⁾(u) ≈ h⁽⁴⁾(u_ref) + ∇h⁽⁴⁾(u_ref)ᵀ (u - u_ref)
    #           = h⁽⁴⁾_lin_const + h⁽⁴⁾_gradᵀ u.
    h_ddddot_lin_const = h_ddddot_ref - float(h_ddddot_grad @ u_ref)

    # phi_const := constant term in
    #   phi(u) = λ₄ h⁽⁴⁾_lin(u) + λ₃ h⁽³⁾(u) + λ₂ h⁽²⁾(u) + λ₁ h⁽¹⁾ + λ₀ h.
    phi_const = (
        lambda4 * h_ddddot_lin_const
        + lambda3 * h_dddot_const
        + lambda2 * h_ddot_const
        + lambda1 * h_dot
        + lambda0 * h_value
    )
    # phi_grad := affine coefficient multiplying the QP variable u.
    phi_grad = lambda4 * h_ddddot_grad + lambda3 * h_dddot_grad + lambda2 * h_ddot_grad

    # phi_ref := exact 4th-order ECBF expression evaluated at u_ref for debugging.
    phi_ref = (
        lambda4 * h_ddddot_ref
        + lambda3 * h_dddot_ref
        + lambda2 * h_ddot_ref
        + lambda1 * h_dot
        + lambda0 * h_value
    )

    debug_terms = {
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
    r_mismatch: float,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
    debug=False
) -> np.ndarray:
    """
    Build and solve one affine approximation of the obstacle ECBF QP.

    Decision variables
    ------------------
    - ``u`` ∈ ℝ⁴ : per-motor thrusts.
    - ``slack`` ≥ 0 : shared softening variable.

    Objective
    ---------
    minimise ‖u - u_ref‖² + CBF_SLACK_WEIGHT · slack²

    Constraints
    -----------
    - One inequality per obstacle: ``phi_const_i + phi_grad_iᵀ u ≥ -slack``.
    - Elementwise thrust bounds ``u_min ≤ u ≤ u_max``.
    - ``slack ≥ 0``.
    """
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64)
    u_min, u_max = thrust_bounds
    num_multi_agents = len(radii)

    # Repeated-pole ECBF gains for a 4th-order barrier.  When ``CBF_K0 == 1``
    # these are the exact coefficients of ``(λ + CBF_K1)^4``.  Retaining
    # ``CBF_K0`` here preserves a separate scaling knob on the zeroth-order term.
    # ``CBF_K4`` independently scales the highest-order term h⁽⁴⁾.
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

    for obs_idx in range(num_multi_agents):
        radius = float(radii[obs_idx])
        if radius < 0.0:
            continue
        obs_pos = swarm_state.positions[obs_idx]
        phi_const, phi_grad, debug_terms = _ecbf_coefficients(
            dynamics=dynamics,
            obs_pos=obs_pos,
            radius=radius,
            u_ref_thrust=u_ref,
            lambda4=lambda4,
            lambda0=lambda0,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
        )
        # Affine 4th-order ECBF constraint:
        #   phi_lin(u) = phi_const + phi_gradᵀ u >= -slack.
        phi_expr = phi_const + phi_grad @ u_var
        phi_list.append(phi_expr)
        h1_list.append(debug_terms["h1"])
        h_list.append(debug_terms["h0"])
        constraints.append(phi_expr >= lambda1 * r_mismatch - slack)

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
            return clipped, h_list, h1_list, phi_list
        return clipped
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print('OTHER ISSUE', problem.status)
        return np.clip(u_ref, u_min, u_max)
    if debug:
        for agent_id in range(len(h_list)):
            print('h:', h_list[agent_id])
            print('h_dot:', h1_list[agent_id])
            print('phi:', phi_list[agent_id].value)
        # print('Slack: ', slack.value, 'u dist:', np.linalg.norm(u_ref - u_var.value), u_var.value)
        # print(np.linalg.norm(u_var.value - u_min), np.linalg.norm(u_var.value - u_max))
    solution = np.array(u_var.value, dtype=np.float64)
    # solution = u_ref / np.sum(u_ref) * np.sum(solution) # 
    if debug:
        return solution, h_list, h1_list, phi_list
    return solution # np.clip(solution, u_min, u_max)


def _solve_cbf_qp(
    *,
    u_ref_thrust: np.ndarray,
    dynamics,
    swarm_state: SwarmState,
    radii: np.ndarray,
    r_mismatch: float,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
    relinearization_passes: int = 1,
    debug=False
) -> np.ndarray:
    """
    Build and solve the obstacle ECBF quadratic program, optionally repeating
    the 4th-order Taylor linearisation around the latest QP solution.

    When ``relinearization_passes > 1`` this performs sequential convexification:

        u⁽⁰⁾ := u_ref
        linearise h⁽⁴⁾ around u⁽k⁾
        solve affine QP to obtain u⁽k+1⁾

    and returns the final iterate.  This reduces the mismatch between the exact
    nonlinear ECBF and the single affine model, at the cost of extra QP solves.
    """
    passes = int(relinearization_passes)
    if passes <= 0:
        raise ValueError("relinearization_passes must be >= 1")

    iterate = np.asarray(u_ref_thrust, dtype=np.float64)
    final_debug = None
    for pass_idx in range(passes):
        outputs = _solve_cbf_qp_single(
            u_ref_thrust=iterate,
            dynamics=dynamics,
            swarm_state=swarm_state,
            radii=radii,
            r_mismatch=r_mismatch,
            thrust_bounds=thrust_bounds,
            debug=debug,
        )
        if debug:
            iterate, h_list, h1_list, phi_list = outputs
            final_debug = (h_list, h1_list, phi_list)
        else:
            iterate = outputs

        # The last iterate becomes the next linearisation point.
        iterate = np.asarray(iterate, dtype=np.float64)

    if debug:
        h_list, h1_list, phi_list = final_debug
        return iterate, h_list, h1_list, phi_list
    return iterate


def apply_cbf_filter(
    base_action: np.ndarray,
    radii: np.ndarray,
    r_mismatch: float,
    env_state,
    swarm_state: SwarmState,
    use_repeated_linearization: bool = False,
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
    r_mismatch : float
        Placeholder conformal mismatch term forwarded to ``_solve_cbf_qp``.
    use_repeated_linearization : bool
        If true, solve the affine ECBF QP multiple times, re-linearising h⁽⁴⁾
        around the latest iterate each time.
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _solve_cbf_qp(
            u_ref_thrust=u_ref_thrust,
            dynamics=dynamics,
            swarm_state=swarm_state,
            radii=radii,
            r_mismatch=float(r_mismatch),
            thrust_bounds=(u_min, u_max),
            relinearization_passes=(CBF_RELINEARIZATION_PASSES if use_repeated_linearization else 1),
            debug=debug
        )
    if debug:
        safe_thrust, h_list, h1_list, phi_list = outputs
    else:
        safe_thrust = outputs

    # Convert Newton thrust back to the environment's action space.
    safe_normalized = _thrust_to_normalized(safe_thrust, dynamics)
    safe_action = 2.0 * safe_normalized - 1.0
    clipped_action = np.clip(safe_action.astype(np.float32), -1.0, 1.0)
    if debug:
        return clipped_action, h_list, h1_list, phi_list
    else:
        return clipped_action

def make_cbf_filter(radii: np.ndarray, use_repeated_linearization: bool = False):
    def filter(base_action: np.ndarray, env_state, swarm_state: SwarmState, debug=False):
        return apply_cbf_filter(
            base_action,
            radii,
            0.0,
            env_state,
            swarm_state,
            use_repeated_linearization=use_repeated_linearization,
            debug=debug,
        )
    return filter
