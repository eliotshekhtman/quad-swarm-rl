import contextlib
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np

from collect_rollout_json import EXPLANATION, RolloutRecorder, _json_default, parse_args
from gym_art.quadrotor_multi.quadrotor_control import RawControl


def make_quad():
    dynamics = SimpleNamespace(
        pos=np.array([1., 2., 3.]), vel=np.zeros(3), rot=np.eye(3),
        omega=np.zeros(3), acc=np.zeros(3), accelerometer=np.array([0., 0., 9.81]),
        omega_dot=np.zeros(3), torque=np.zeros(3),
        prop_pos=np.zeros((4, 3)), model_params={"mass": np.float64(0.03)},
        thrust_rot_damp=np.zeros(4), thrust_cmds_damp=np.zeros(4),
        thrust_noise=SimpleNamespace(state=np.zeros(4)), thrust_max=np.ones(4),
        angvel2thrust=lambda x, linearity: x, motor_linearity=1.,
        on_floor=False, crashed_floor=False, crashed_wall=False, crashed_ceiling=False,
    )
    controller = RawControl(dynamics)
    controller.action_space(dynamics)
    return SimpleNamespace(
        dynamics=dynamics, controller=controller, dt=0.005, sim_steps=2,
        goal=np.array([4., 5., 6.]), obs_repr="xyz_vxyz_R_omega",
        neighbor_obs_type="pos_vel",
    )


class RolloutJsonTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_rollout_seed(self):
        args = parse_args(["--seed", "439", "--r", "1.555"])
        self.assertEqual(args.rollout_seed, 439)
        self.assertTrue(args.deterministic)
        self.assertEqual(args.episode_length, 300)
        self.assertEqual(args.num_multi_agents, 5)
        self.assertEqual(args.output_path, "train_dir/rollout_seed439_r1p555_rollout439.json")
        args = parse_args([
            "--seed", "439", "--radius", "1.555", "--rollout_seed", "0", "--no-deterministic",
        ])
        self.assertEqual(args.rollout_seed, 0)
        self.assertFalse(args.deterministic)

    def test_cli_rejects_invalid_inputs(self):
        for extra in (["--r", "nan"], ["--r", "inf"], ["--r", "0"],
                      ["--episode_length", "0"], ["--num_multi_agents", "0"],
                      ["--seed", "-1"], ["--rollout_seed", "-1"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--seed", "439", "--r", "1.555"] + extra)

    def test_state_action_alignment_labels_and_copies(self):
        env = SimpleNamespace(envs=[make_quad(), make_quad()])
        recorder = RolloutRecorder(num_multi_agents=1, solo_obs_dim=18)
        obs = np.arange(48, dtype=np.float32).reshape(2, 24)
        nominal = np.array([[2., 0., -2., 0.], [.5, .5, .5, .5]])
        command = nominal.copy()
        command[1] = [-.4, -.2, .2, .4]
        recorder(env, 0, obs, nominal, command)
        for quad, action in zip(env.envs, command):
            quad.dynamics.pos[:] += 1
            quad.controller.action = .5 * (np.clip(action, -1, 1) + 1)
        obs += 10
        recorder(env, 1, obs, nominal, command)
        for quad in env.envs:
            quad.dynamics.pos[:] += 1
        recorder(env, 2, obs, None, None)
        ego = recorder.agents["ego"]
        other = recorder.agents["agent_0"]
        self.assertTrue(ego["is_ego"])
        self.assertFalse(other["is_ego"])
        self.assertEqual(ego["simulator_index"], 1)
        self.assertEqual(len(ego["states"]), 3)
        self.assertEqual(len(ego["actions"]), 2)
        self.assertEqual([s["time_s"] for s in ego["states"]], [0., .01, .02])
        self.assertEqual(ego["states"][0]["position_world_m"], [1., 2., 3.])
        self.assertEqual(ego["states"][2]["position_world_m"], [3., 4., 5.])
        self.assertEqual(ego["states"][0]["policy_observation"], list(range(24, 42)))
        self.assertEqual(len(other["states"][0]["policy_observation"]), 24)
        self.assertEqual(ego["actions"][0]["nominal"], [.5] * 4)
        self.assertEqual(ego["actions"][0]["command"], [-.4, -.2, .2, .4])
        np.testing.assert_allclose(ego["actions"][1]["motor_throttle"], [.3, .4, .6, .7])
        self.assertEqual(other["actions"][0]["clipped_command"], [1., 0., -1., 0.])
        self.assertIsNone(ego["states"][0]["torque_body_n_m"])
        self.assertEqual(ego["states"][1]["torque_body_n_m"], [0., 0., 0.])
        payload = {"explanation": EXPLANATION, "agents": recorder.agents}
        decoded = json.loads(json.dumps(payload, default=_json_default, allow_nan=False))
        self.assertEqual(next(iter(decoded)), "explanation")
        command[:] = 0
        self.assertEqual(ego["actions"][0]["command"], [-.4, -.2, .2, .4])

    def test_serializer_rejects_nonfinite_data(self):
        with self.assertRaises(ValueError):
            json.dumps({"state": np.array([np.nan])}, default=_json_default, allow_nan=False)
        with self.assertRaises(TypeError):
            json.dumps({"state": object()}, default=_json_default)


if __name__ == "__main__":
    unittest.main()
