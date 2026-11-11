#!/usr/bin/env python3
"""Export a fixed-radius rollout using the same execution path as collect_video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from collect_video import (
    DEVICE,
    _apply_rollout_seed,
    _build_eval_cfg,
    _collect_trajectory,
    _load_or_generate_pred_trajectories,
)
from gym_art.quadrotor_multi.quadrotor_control import RawControl
from project_utils.full_cbf_utils import CBF_K1, make_cbf_filter
from project_utils.prediction_utils import EVALUATION_SCENARIO
from project_utils.restart_utils import deterministic_reset
from project_utils.utils import latest_checkpoint, load_actor, load_cfg
from sample_factory.model.model_utils import get_rnn_size
from swarm_rl.env_snapshot import safe_capture_env_snapshot
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import register_swarm_components


EXPLANATION = (
    "One joint go_to_goal rollout. agents is keyed by agent_0, agent_1, ..., ego; "
    "ego is the last simulator agent and uses the solo policy plus the full fourth-order ECBF. "
    "Other agents use the multi-agent policy with deterministic mean actions. "
    "Each agent has T+1 states and T actions. states[t] is the state BEFORE actions[t]; "
    "actions[t] produces states[t+1]. State 0 is the initial condition and state T has no outgoing action. "
    "time_s uses the control timestep, including all physics substeps. "
    "position_world_m, velocity_world_m_s, and goal_world_m are world-frame xyz values, with z upward. "
    "rotation_body_to_world is a row-major 3x3 matrix: v_world = R @ v_body. "
    "angular_velocity_body_rad_s is body-frame xyz angular velocity, not Euler angles. "
    "These position, velocity, rotation and angular-velocity fields encode the full rigid-body state. "
    "linear_acceleration_world_m_s2 and accelerometer_body_m_s2 are simulator acceleration and "
    "body-frame specific force; angular_acceleration_body_rad_s2 and torque_body_n_m are the "
    "last physics-substep values (null initially). motor_speed_filtered and motor_command_filtered "
    "are the four dimensionless motor-lag states; the latter includes clipped thrust noise. "
    "thrust_noise_state is the four-component Ornstein-Uhlenbeck noise state. "
    "rotor_thrusts_n is computed from the current filtered motor commands and motor parameters. "
    "on_floor and crashed_floor/wall/ceiling are simulator flags. "
    "policy_observation is the exact unnormalized input supplied to the actor, which may contain "
    "sensor noise and differ from the physical state. For xyz_vxyz_R_omega, entries 0:3 are "
    "position minus goal, 3:6 velocity, 6:15 flattened rotation, 15:18 angular velocity; "
    "pos_vel neighbor observations append relative position/velocity blocks of six. "
    "The ego receives only its solo observation prefix. "
    "Each action is a length-four vector in simulator rotor order 0,1,2,3, NOT a desired "
    "xyz acceleration or waypoint. nominal is the policy output; command is what is passed to "
    "env.step, after ECBF filtering for ego. clipped_command respects the controller limits. "
    "motor_throttle is the actual controller output into the dynamics before motor lag/noise: "
    "scale * (clipped_command + bias), normally (clipped_command + 1)/2. Thus -1 means off, "
    "+1 maximum throttle, and 0 half throttle, not necessarily hover. Rotor body positions "
    "and model parameters are recorded per agent. "
    "Each non-ego agent also has predictions_used_by_ecbf[t] containing the position and velocity "
    "reference used for actions[t]. These preserve the existing video/conformal indexing: "
    "prediction index t is a post-step homogeneous prediction, while the controller reads it "
    "at pre-step t. radius_m is the total controller radius, including physical separation. "
    "scenario_seed chooses the layout; rollout_seed chooses the rollout random stream and defaults "
    "to scenario_seed. Deterministic policy actions do not remove simulator noise. "
    "This is a physical trajectory/action record, not a resumable simulator/RNG checkpoint."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--r", "--radius", dest="r", type=float, required=True)
    parser.add_argument("--rollout_seed", type=int, default=None)
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--num_multi_agents", type=int, default=5)
    parser.add_argument("--multi_train_dir", default="train_dir")
    parser.add_argument("--solo_train_dir", default="train_dir")
    parser.add_argument("--multi_experiment", default="multi_ddg_6")
    parser.add_argument("--solo_experiment", default="solo_ddg")
    parser.add_argument("--predictions_dir", default="train_dir")
    parser.add_argument("--init_predictions", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args(argv)
    if not np.isfinite(args.r) or args.r <= 0:
        parser.error("--r must be finite and positive")
    if args.episode_length <= 0 or args.num_multi_agents <= 0:
        parser.error("--episode_length and --num_multi_agents must be positive")
    if args.seed < 0 or (args.rollout_seed is not None and args.rollout_seed < 0):
        parser.error("seeds must be non-negative")
    if args.rollout_seed is None:
        args.rollout_seed = args.seed
    if args.init_predictions is None:
        candidate = f"homog_gtg_pred_traj_data_seed{args.seed}"
        if (Path(args.predictions_dir) / candidate / "pred_trajectories.npz").is_file():
            args.init_predictions = candidate
    if args.output_path is None:
        radius_tag = f"{args.r:g}".replace(".", "p")
        args.output_path = (
            f"train_dir/rollout_seed{args.seed}_r{radius_tag}"
            f"_rollout{args.rollout_seed}.json"
        )
    return args


def _array(value):
    return np.asarray(value).tolist()


class RolloutRecorder:
    def __init__(self, num_multi_agents, solo_obs_dim):
        self.ego_index = num_multi_agents
        self.solo_obs_dim = solo_obs_dim
        self.agents = {}
        self.control_dt = None

    def __call__(self, env, step, obs, nominal_actions, actions):
        for i, quad in enumerate(env.envs):
            d = quad.dynamics
            controller = quad.controller
            if not isinstance(controller, RawControl):
                raise ValueError("JSON action descriptions require the RawControl controller")
            key = "ego" if i == self.ego_index else f"agent_{i}"
            if step == 0:
                self.control_dt = float(quad.dt * quad.sim_steps)
                self.agents[key] = {
                    "simulator_index": i,
                    "is_ego": i == self.ego_index,
                    "controller": "solo_policy_with_full_ecbf" if i == self.ego_index else "multi_policy",
                    "rotor_positions_body_m": _array(d.prop_pos),
                    "model_parameters": d.model_params,
                    "action_low": _array(controller.low),
                    "action_high": _array(controller.high),
                    "action_scale": float(controller.scale),
                    "action_bias": float(controller.bias),
                    "observation_representation": quad.obs_repr,
                    "neighbor_observation_type": quad.neighbor_obs_type if i != self.ego_index else "none",
                    "states": [],
                    "actions": [],
                }
            agent = self.agents[key]
            # The controller has now executed the preceding action; capture its actual output.
            if step > 0:
                agent["actions"][-1]["motor_throttle"] = _array(controller.action)
            agent["states"].append({
                "timestep": step,
                "time_s": step * self.control_dt,
                "position_world_m": _array(d.pos),
                "velocity_world_m_s": _array(d.vel),
                "rotation_body_to_world": _array(d.rot),
                "angular_velocity_body_rad_s": _array(d.omega),
                "goal_world_m": _array(quad.goal),
                "linear_acceleration_world_m_s2": _array(d.acc),
                "accelerometer_body_m_s2": _array(d.accelerometer),
                "angular_acceleration_body_rad_s2": _array(d.omega_dot) if step else None,
                "torque_body_n_m": _array(d.torque) if step else None,
                "motor_speed_filtered": _array(d.thrust_rot_damp),
                "motor_command_filtered": _array(d.thrust_cmds_damp),
                "thrust_noise_state": _array(d.thrust_noise.state),
                "rotor_thrusts_n": _array(d.thrust_max * d.angvel2thrust(
                    d.thrust_cmds_damp, linearity=d.motor_linearity)),
                "on_floor": bool(d.on_floor),
                "crashed_floor": bool(d.crashed_floor),
                "crashed_wall": bool(d.crashed_wall),
                "crashed_ceiling": bool(d.crashed_ceiling),
                "policy_observation": _array(obs[i, :self.solo_obs_dim] if i == self.ego_index else obs[i]),
            })
            if actions is not None:
                agent["actions"].append({
                    "timestep": step,
                    "nominal": _array(nominal_actions[i]),
                    "command": _array(actions[i]),
                    "clipped_command": _array(np.clip(actions[i], controller.low, controller.high)),
                })


def collect_rollout(args):
    register_swarm_components()
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    eval_cfg = _build_eval_cfg(cfg_multi, args.num_multi_agents + 1, False, "topdownfollow")
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)
    try:
        if args.episode_length > min(q.ep_len for q in env.unwrapped.envs):
            raise ValueError("--episode_length exceeds the simulator horizon; auto-reset would corrupt the final state")
        multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment)
        multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, DEVICE)
        multi_rnn = torch.zeros((args.num_multi_agents, get_rnn_size(cfg_multi)), device=DEVICE)
        solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
        try:
            solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment)
            solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, DEVICE)
            solo_obs_dim = int(solo_env.observation_space.shape[0])
            solo_rnn = torch.zeros((1, get_rnn_size(cfg_solo)), device=DEVICE)
        finally:
            solo_env.close()
        obs, stored_states = deterministic_reset(env, args.seed, None)
        predictions = _load_or_generate_pred_trajectories(
            args, env, stored_states, args.num_multi_agents, multi_actor, multi_rnn, multi_ckpt)
        obs, _ = deterministic_reset(env, args.seed, stored_states)
        _apply_rollout_seed(env, args.rollout_seed)
        recorder = RolloutRecorder(args.num_multi_agents, solo_obs_dim)
        trajectory = _collect_trajectory(
            safe_capture_env_snapshot(env), obs,
            num_multi_agents=args.num_multi_agents,
            multi_actor=multi_actor, multi_rnn_states=multi_rnn,
            solo_actor=solo_actor, solo_rnn_states=solo_rnn, solo_obs_dim=solo_obs_dim,
            pred_trajectories=predictions,
            solo_action_fn=make_cbf_filter(np.full(args.num_multi_agents, args.r)),
            max_steps=args.episode_length, deterministic=args.deterministic,
            step_callback=recorder,
        )
    finally:
        env.close()
    steps = int(trajectory["trajectory_length"])
    for i in range(args.num_multi_agents):
        recorder.agents[f"agent_{i}"]["predictions_used_by_ecbf"] = [
            {"timestep": t, "position_world_m": _array(p[:3]), "velocity_world_m_s": _array(p[3:])}
            for t, p in enumerate(predictions[i][:steps])
        ]
    prediction_path = (
        str(Path(args.predictions_dir) / args.init_predictions / "pred_trajectories.npz")
        if args.init_predictions else None
    )
    return {
        "explanation": EXPLANATION,
        "schema_version": 1,
        "metadata": {
            "scenario": EVALUATION_SCENARIO,
            "scenario_seed": args.seed,
            "rollout_seed": args.rollout_seed,
            "radius_m": args.r,
            "cbf_k1": CBF_K1,
            "deterministic_ego_policy": args.deterministic,
            "device": str(DEVICE),
            "num_agents": args.num_multi_agents + 1,
            "ego_index": args.num_multi_agents,
            "num_steps": steps,
            "control_timestep_s": recorder.control_dt,
            "multi_checkpoint": multi_ckpt,
            "solo_checkpoint": solo_ckpt,
            "prediction_artifact": prediction_path,
            "prediction_source": "artifact" if prediction_path else "generated_homogeneous_rollout",
        },
        "agents": recorder.agents,
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    args = parse_args()
    payload = collect_rollout(args)
    text = json.dumps(payload, indent=2, allow_nan=False, default=_json_default)
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"Saved {payload['metadata']['num_steps']} steps for {len(payload['agents'])} agents to {path}")


if __name__ == "__main__":
    main()
