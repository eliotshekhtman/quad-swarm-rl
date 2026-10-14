import unittest

import numpy as np

from project_utils.full_cbf_utils import (
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


if __name__ == "__main__":
    unittest.main()
