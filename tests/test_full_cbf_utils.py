import unittest
from types import SimpleNamespace
from unittest import mock

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy.lin_ops import lin_utils

from project_utils.full_cbf_utils import (
    CBF_SLACK_WEIGHT,
    _ParameterizedCbfQp,
    _ecbf_coefficients,
    _ego_ecbf_terms,
    _normalized_to_thrust,
    _skew,
    _solve_cbf_qp,
    _solve_legacy_parameterized_qp,
    _thrust_to_normalized,
    make_cbf_filter,
)
from project_utils.utils import SwarmState


class FakeDynamics:
    def __init__(self):
        self.motor_tau_down = 1.0
        self.motor_tau_up = 1.0
        self.thrust_cmds_damp = np.zeros(4)
        self.thrust_rot_damp = np.zeros(4)
        self.thrust_max = np.full(4, 10.0)
        self.motor_linearity = 1.0
        self.rot = np.eye(3)
        self.omega = np.zeros(3)
        self.mass = 1.0
        self.inertia = np.ones(3)
        self.prop_crossproducts = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

    @staticmethod
    def angvel2thrust(commands, linearity):
        del linearity
        return np.asarray(commands)


class FullEcbfTests(unittest.TestCase):
    def setUp(self):
        self.dynamics = FakeDynamics()

    def test_skew_matrix_matches_cross_product(self):
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([-2.0, 4.0, 1.0])
        np.testing.assert_allclose(_skew(left) @ right, np.cross(left, right))

    def test_motor_conversion_round_trip(self):
        commands = np.array([0.1, 0.25, 0.5, 0.9])
        thrust = _normalized_to_thrust(commands, self.dynamics)
        recovered = _thrust_to_normalized(thrust, self.dynamics)
        np.testing.assert_allclose(recovered, commands, atol=1.0e-8)

    def test_linearized_constraint_matches_reference_value(self):
        ego_terms = _ego_ecbf_terms(self.dynamics)
        u_ref = np.array([1.0, 1.5, 2.0, 2.5])
        phi_const, phi_grad, debug = _ecbf_coefficients(
            solo_pos=np.array([2.0, 1.0, 0.5]),
            solo_vel=np.array([0.2, -0.1, 0.3]),
            teammate_pos=np.zeros(3),
            teammate_vel=np.array([0.1, 0.0, -0.2]),
            ego_terms=ego_terms,
            radius=0.75,
            u_ref_thrust=u_ref,
            lambda4=0.5,
            lambda0=1.0,
            lambda1=2.0,
            lambda2=3.0,
            lambda3=4.0,
        )
        self.assertTrue(np.isfinite(phi_const))
        self.assertEqual(phi_grad.shape, (4,))
        np.testing.assert_allclose(
            phi_const + phi_grad @ u_ref,
            debug["phi_ref"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_teammate_velocity_changes_relative_barrier_derivative(self):
        ego_terms = _ego_ecbf_terms(self.dynamics)
        common = {
            "solo_pos": np.array([2.0, 0.0, 0.0]),
            "solo_vel": np.array([1.0, 0.0, 0.0]),
            "teammate_pos": np.zeros(3),
            "ego_terms": ego_terms,
            "radius": 0.5,
            "u_ref_thrust": np.ones(4),
            "lambda4": 0.5,
            "lambda0": 1.0,
            "lambda1": 2.0,
            "lambda2": 3.0,
            "lambda3": 4.0,
        }
        _, _, stationary = _ecbf_coefficients(
            teammate_vel=np.zeros(3),
            **common,
        )
        _, _, moving = _ecbf_coefficients(
            teammate_vel=np.array([0.5, 0.0, 0.0]),
            **common,
        )
        self.assertNotEqual(stationary["h1"], moving["h1"])

    def test_qp_without_active_neighbors_clips_to_motor_bounds(self):
        swarm_state = SwarmState(
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        actual = _solve_cbf_qp(
            u_ref_thrust=np.array([-1.0, 2.0, 12.0, 5.0]),
            dynamics=self.dynamics,
            swarm_state=swarm_state,
            radii=np.array([-1.0]),
            thrust_bounds=(np.zeros(4), np.full(4, 10.0)),
        )
        np.testing.assert_allclose(actual, [0.0, 2.0, 10.0, 5.0])

    def test_explicit_current_settings_preserve_default_solution(self):
        state = SwarmState(
            positions=np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        common = dict(
            u_ref_thrust=np.array([2.0, 3.0, 4.0, 5.0]),
            dynamics=self.dynamics, swarm_state=state, radii=np.array([0.75]),
            thrust_bounds=(np.zeros(4), np.full(4, 10.0)),
        )
        np.testing.assert_array_equal(
            _solve_cbf_qp(**common),
            _solve_cbf_qp(**common, cbf_k1=2.5, solver_mode="cascade"),
        )

    def test_per_filter_settings_do_not_leak_or_keep_legacy_fallback_state(self):
        quad = SimpleNamespace(dynamics=self.dynamics, tick=0)
        env = SimpleNamespace(envs=[quad])
        legacy = make_cbf_filter(np.array([1.0]), cbf_k1=4.0, solver_mode="legacy")
        current = make_cbf_filter(np.array([1.0]), cbf_k1=2.5)
        with mock.patch(
            "project_utils.full_cbf_utils._solve_cbf_qp", return_value=np.ones(4),
        ) as solve:
            legacy(np.zeros(4), env, None)
            quad.tick = 1
            legacy(np.zeros(4), env, None)
            current(np.zeros(4), env, None)
        for call in solve.call_args_list[:2]:
            self.assertEqual(call.kwargs["cbf_k1"], 4.0)
            self.assertEqual(call.kwargs["solver_mode"], "legacy")
            self.assertIsNone(call.kwargs["fallback_thrust"])
        self.assertEqual(solve.call_args_list[2].kwargs["cbf_k1"], 2.5)
        self.assertEqual(solve.call_args_list[2].kwargs["solver_mode"], "cascade")

    def test_gain_override_reaches_every_barrier_coefficient(self):
        state = SwarmState(
            positions=np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        for gain in (4.0, 2.5):
            with self.subTest(gain=gain), mock.patch(
                "project_utils.full_cbf_utils._ecbf_coefficients",
                wraps=_ecbf_coefficients,
            ) as coefficients:
                _solve_cbf_qp(
                    u_ref_thrust=np.array([2.0, 3.0, 4.0, 5.0]),
                    dynamics=self.dynamics,
                    swarm_state=state,
                    radii=np.array([0.75]),
                    thrust_bounds=(np.zeros(4), np.full(4, 10.0)),
                    cbf_k1=gain,
                    solver_mode="legacy",
                )
                values = coefficients.call_args.kwargs
                self.assertEqual(values["lambda4"], 0.5)
                self.assertEqual(values["lambda3"], 4 * gain)
                self.assertEqual(values["lambda2"], 6 * gain ** 2)
                self.assertEqual(values["lambda1"], 4 * gain ** 3)
                self.assertEqual(values["lambda0"], gain ** 4)

    def test_filter_rejects_invalid_controller_settings(self):
        for kwargs in ({"cbf_k1": 0}, {"cbf_k1": -1}, {"cbf_k1": np.nan},
                       {"cbf_k1": np.inf}, {"solver_mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_cbf_filter(np.array([1.0]), **kwargs)

    @staticmethod
    def _fresh_problem(u_ref, u_min, u_max, coefficients):
        u_var = cp.Variable(4)
        slack = cp.Variable()
        constraints = [
            phi_const + phi_grad @ u_var >= -slack
            for phi_const, phi_grad in coefficients
        ]
        constraints.extend(
            [
                u_var >= u_min,
                u_var <= u_max,
                slack >= 0,
            ]
        )
        objective = (
            cp.sum_squares(u_var - u_ref)
            + CBF_SLACK_WEIGHT * cp.square(slack)
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem, u_var

    def assert_solver_value_equal(self, left, right):
        if left is None or right is None:
            self.assertIs(left, right)
            return
        if sp.issparse(left) or sp.issparse(right):
            left = left.tocsc()
            right = right.tocsc()
            self.assertEqual(left.shape, right.shape)
            np.testing.assert_array_equal(left.indptr, right.indptr)
            np.testing.assert_array_equal(left.indices, right.indices)
            np.testing.assert_array_equal(left.data, right.data)
            return
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_parameterized_qp_has_identical_ecos_data(self):
        rng = np.random.default_rng(20261019)
        active_neighbors = tuple(range(5))
        parameterized = _ParameterizedCbfQp(active_neighbors)

        for _ in range(20):
            u_ref = rng.uniform(0.1, 9.9, size=4)
            u_min = np.zeros(4)
            u_max = np.full(4, 10.0)
            coefficients = [
                (float(rng.normal()), rng.normal(size=4))
                for _ in active_neighbors
            ]
            fresh, _ = self._fresh_problem(
                u_ref,
                u_min,
                u_max,
                coefficients,
            )
            parameterized.set_values(
                u_ref=u_ref,
                u_min=u_min,
                u_max=u_max,
                coefficients=coefficients,
            )

            fresh_data = fresh.get_problem_data(cp.ECOS)[0]
            parameterized_data = parameterized.problem.get_problem_data(cp.ECOS)[0]
            for key in ("c", "A", "b", "G", "h"):
                self.assert_solver_value_equal(
                    fresh_data.get(key),
                    parameterized_data.get(key),
                )
            self.assertEqual(str(fresh_data["dims"]), str(parameterized_data["dims"]))

    def test_parameterized_qp_matches_fresh_solutions_exactly(self):
        rng = np.random.default_rng(271828)
        active_neighbors = tuple(range(5))
        parameterized = _ParameterizedCbfQp(active_neighbors)
        cases = []

        for _ in range(40):
            u_ref = rng.uniform(0.1, 9.9, size=4)
            u_min = np.zeros(4)
            u_max = np.full(4, 10.0)
            coefficients = [
                (float(rng.normal()), rng.normal(size=4))
                for _ in active_neighbors
            ]
            cases.append((u_ref, u_min, u_max, coefficients))

            fresh, fresh_u = self._fresh_problem(*cases[-1])
            fresh.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            parameterized.set_values(
                u_ref=u_ref,
                u_min=u_min,
                u_max=u_max,
                coefficients=coefficients,
            )
            parameterized.problem.solve(
                solver=cp.ECOS,
                warm_start=False,
                verbose=False,
            )

            self.assertEqual(fresh.status, parameterized.problem.status)
            np.testing.assert_array_equal(fresh_u.value, parameterized.u_var.value)
            np.testing.assert_array_equal(
                fresh_u.value.astype(np.float32),
                parameterized.u_var.value.astype(np.float32),
            )

        target = cases[0]
        parameterized.set_values(
            u_ref=target[0],
            u_min=target[1],
            u_max=target[2],
            coefficients=target[3],
        )
        parameterized.problem.solve(
            solver=cp.ECOS,
            warm_start=False,
            verbose=False,
        )
        fresh, fresh_u = self._fresh_problem(*target)
        fresh.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        np.testing.assert_array_equal(fresh_u.value, parameterized.u_var.value)

    def test_cached_qp_does_not_consume_ids_after_first_solve(self):
        swarm_state = SwarmState(
            positions=np.array(
                [
                    [3.0, 0.0, 1.0],
                    [-3.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, -3.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            velocities=np.zeros((6, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 6, axis=0),
        )
        common = {
            "u_ref_thrust": np.array([2.0, 3.0, 4.0, 5.0]),
            "dynamics": self.dynamics,
            "swarm_state": swarm_state,
            "radii": np.full(5, 0.75),
            "thrust_bounds": (np.zeros(4), np.full(4, 10.0)),
        }
        qp_cache = {}
        expected = _solve_cbf_qp(qp_cache=qp_cache, **common)
        counter_after_compile = lin_utils.ID_COUNTER.count

        for _ in range(100):
            actual = _solve_cbf_qp(qp_cache=qp_cache, **common)
            np.testing.assert_array_equal(actual, expected)

        self.assertEqual(lin_utils.ID_COUNTER.count, counter_after_compile)
        self.assertEqual(list(qp_cache), [(0, 1, 2, 3, 4)])

    def test_cached_relinearization_matches_uncached_solves(self):
        swarm_state = SwarmState(
            positions=np.array(
                [
                    [3.0, 0.0, 1.0],
                    [-3.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            velocities=np.zeros((3, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        )
        common = {
            "u_ref_thrust": np.array([2.0, 3.0, 4.0, 5.0]),
            "dynamics": self.dynamics,
            "swarm_state": swarm_state,
            "radii": np.full(2, 0.75),
            "thrust_bounds": (np.zeros(4), np.full(4, 10.0)),
            "relinearization_passes": 2,
        }
        uncached = _solve_cbf_qp(**common)
        cached = _solve_cbf_qp(qp_cache={}, **common)
        np.testing.assert_array_equal(cached, uncached)

    def test_solver_errors_use_explicit_fallback_after_all_attempts(self):
        swarm_state = SwarmState(
            positions=np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        u_ref = np.array([2.0, 3.0, 4.0, 5.0])
        common = {
            "u_ref_thrust": u_ref,
            "dynamics": self.dynamics,
            "swarm_state": swarm_state,
            "radii": np.array([0.75]),
            "thrust_bounds": (np.zeros(4), np.full(4, 10.0)),
        }
        qp_cache = {}
        _solve_cbf_qp(qp_cache=qp_cache, **common)
        qp = qp_cache[(0,)]
        qp.u_var.value = np.full(4, 9.0)
        attempted_solvers = []

        def fail_solve(*args, **kwargs):
            attempted_solvers.append(kwargs["solver"])
            self.assertIsNone(qp.u_var.value)
            self.assertFalse(kwargs["warm_start"])
            raise cp.SolverError("forced failure")

        fallback = np.array([8.0, 7.0, 6.0, 5.0])
        with mock.patch.object(qp.problem, "solve", side_effect=fail_solve):
            with mock.patch("builtins.print"):
                actual = _solve_cbf_qp(
                    qp_cache=qp_cache,
                    fallback_thrust=fallback,
                    **common,
                )
        self.assertEqual(attempted_solvers, [cp.ECOS, cp.CLARABEL, cp.SCS])
        np.testing.assert_array_equal(actual, fallback)

    def test_solver_cascade_uses_clarabel_after_ecos_failure(self):
        swarm_state = SwarmState(
            positions=np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        common = {
            "u_ref_thrust": np.array([2.0, 3.0, 4.0, 5.0]),
            "dynamics": self.dynamics,
            "swarm_state": swarm_state,
            "radii": np.array([0.75]),
            "thrust_bounds": (np.zeros(4), np.full(4, 10.0)),
        }
        qp_cache = {}
        _solve_cbf_qp(qp_cache=qp_cache, **common)
        qp = qp_cache[(0,)]
        original_solve = qp.problem.solve
        attempted_solvers = []

        def solve_after_ecos_failure(*args, **kwargs):
            attempted_solvers.append(kwargs["solver"])
            if kwargs["solver"] == cp.ECOS:
                raise TypeError("forced CVXPY canonicalization failure")
            return original_solve(*args, **kwargs)

        with mock.patch.object(
            qp.problem,
            "solve",
            side_effect=solve_after_ecos_failure,
        ):
            actual = _solve_cbf_qp(qp_cache=qp_cache, **common)

        self.assertEqual(attempted_solvers, [cp.ECOS, cp.CLARABEL])
        self.assertTrue(np.all(np.isfinite(actual)))

    def test_solver_cascade_retries_ecos_optimal_inaccurate(self):
        swarm_state = SwarmState(
            positions=np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        common = {
            "u_ref_thrust": np.array([2.0, 3.0, 4.0, 5.0]),
            "dynamics": self.dynamics,
            "swarm_state": swarm_state,
            "radii": np.array([0.75]),
            "thrust_bounds": (np.zeros(4), np.full(4, 10.0)),
        }
        qp_cache = {}
        _solve_cbf_qp(qp_cache=qp_cache, **common)
        qp = qp_cache[(0,)]
        original_solve = qp.problem.solve
        attempted_solvers = []

        def mark_ecos_inaccurate(*args, **kwargs):
            attempted_solvers.append(kwargs["solver"])
            result = original_solve(*args, **kwargs)
            if kwargs["solver"] == cp.ECOS:
                qp.problem._status = cp.OPTIMAL_INACCURATE
            return result

        with mock.patch.object(qp.problem, "solve", side_effect=mark_ecos_inaccurate):
            actual = _solve_cbf_qp(qp_cache=qp_cache, **common)

        self.assertEqual(attempted_solvers, [cp.ECOS, cp.CLARABEL])
        self.assertTrue(np.all(np.isfinite(actual)))

    def test_solver_cascade_uses_scs_after_ecos_and_clarabel_failures(self):
        swarm_state = SwarmState(
            positions=np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        common = {
            "u_ref_thrust": np.array([2.0, 3.0, 4.0, 5.0]),
            "dynamics": self.dynamics,
            "swarm_state": swarm_state,
            "radii": np.array([0.75]),
            "thrust_bounds": (np.zeros(4), np.full(4, 10.0)),
        }
        qp_cache = {}
        _solve_cbf_qp(qp_cache=qp_cache, **common)
        qp = qp_cache[(0,)]
        original_solve = qp.problem.solve
        attempted_solvers = []

        def solve_with_scs(*args, **kwargs):
            attempted_solvers.append(kwargs["solver"])
            if kwargs["solver"] in (cp.ECOS, cp.CLARABEL):
                raise cp.SolverError("forced solver failure")
            return original_solve(*args, **kwargs)

        with mock.patch.object(qp.problem, "solve", side_effect=solve_with_scs):
            actual = _solve_cbf_qp(qp_cache=qp_cache, **common)

        self.assertEqual(attempted_solvers, [cp.ECOS, cp.CLARABEL, cp.SCS])
        self.assertTrue(np.all(np.isfinite(actual)))

    def test_filter_reuses_previous_action_and_clears_it_on_rollout_reset(self):
        quad = SimpleNamespace(dynamics=self.dynamics, tick=0)
        env_state = SimpleNamespace(envs=[quad])
        swarm_state = SwarmState(
            positions=np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            velocities=np.zeros((2, 3)),
            rotations=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )
        action_filter = make_cbf_filter(np.array([0.75]))
        returned_thrusts = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.5, 2.5, 3.5, 4.5]),
            np.array([2.0, 3.0, 4.0, 5.0]),
        ]

        with mock.patch(
            "project_utils.full_cbf_utils._solve_cbf_qp",
            side_effect=returned_thrusts,
        ) as solve:
            action_filter(np.zeros(4), env_state, swarm_state)
            quad.tick = 1
            action_filter(np.zeros(4), env_state, swarm_state)
            quad.tick = 0
            action_filter(np.zeros(4), env_state, swarm_state)

        self.assertIsNone(solve.call_args_list[0].kwargs["fallback_thrust"])
        np.testing.assert_array_equal(
            solve.call_args_list[1].kwargs["fallback_thrust"],
            returned_thrusts[0],
        )
        self.assertIsNone(solve.call_args_list[2].kwargs["fallback_thrust"])


class LegacySolverTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([2.0, 3.0, 4.0, 5.0])
        self.qp = SimpleNamespace(
            problem=mock.Mock(status=cp.OPTIMAL),
            u_ref=SimpleNamespace(value=self.reference),
            u_var=SimpleNamespace(value=None),
        )
        self.bounds = (np.zeros(4), np.full(4, 10.0))

    def solve(self):
        with mock.patch("builtins.print"):
            result = _solve_legacy_parameterized_qp(self.qp, *self.bounds)
        self.qp.problem.solve.assert_called_once_with(
            solver=cp.ECOS, warm_start=False, verbose=False,
        )
        return result

    def test_accepts_both_optimal_statuses_without_new_clipping_or_validation(self):
        for status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            with self.subTest(status=status):
                self.qp.problem.solve.reset_mock()
                self.qp.problem.status = status
                self.qp.u_var.value = np.array([-1e-6, 3.0, 4.0, 10.000001])
                np.testing.assert_array_equal(self.solve(), self.qp.u_var.value)

    def test_nonoptimal_status_uses_reference_not_available_iterate(self):
        for status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED):
            with self.subTest(status=status):
                self.qp.problem.solve.reset_mock()
                self.qp.problem.status = status
                self.qp.u_var.value = np.ones(4)
                np.testing.assert_array_equal(self.solve(), self.reference)

    def test_solver_error_uses_available_iterate_or_reference(self):
        self.qp.problem.solve.side_effect = cp.SolverError("forced")
        np.testing.assert_array_equal(self.solve(), self.reference)
        self.qp.problem.solve.reset_mock()
        self.qp.u_var.value = np.array([-1.0, 2.0, 12.0, 5.0])
        np.testing.assert_array_equal(self.solve(), [0.0, 2.0, 10.0, 5.0])

    def test_does_not_swallow_additional_exception_types(self):
        for error in (TypeError, ValueError):
            with self.subTest(error=error), self.assertRaises(error):
                self.qp.problem.solve.side_effect = error("forced")
                _solve_legacy_parameterized_qp(self.qp, *self.bounds)


if __name__ == "__main__":
    unittest.main()
