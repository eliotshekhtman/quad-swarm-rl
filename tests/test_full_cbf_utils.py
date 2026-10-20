import unittest
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
    _thrust_to_normalized,
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

    def test_solver_error_does_not_reuse_previous_solution(self):
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

        def fail_solve(*args, **kwargs):
            self.assertIsNone(qp.u_var.value)
            self.assertFalse(kwargs["warm_start"])
            raise cp.SolverError("forced failure")

        with mock.patch.object(qp.problem, "solve", side_effect=fail_solve):
            with mock.patch("builtins.print"):
                actual = _solve_cbf_qp(qp_cache=qp_cache, **common)
        np.testing.assert_array_equal(actual, u_ref)


if __name__ == "__main__":
    unittest.main()
