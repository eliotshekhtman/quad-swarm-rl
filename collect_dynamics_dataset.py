#!/usr/bin/env python3
"""
Collect state-action-next-state tuples for a configurable number of quadrotors flying the
dynamic_diff_goal scenario.

Each sample records (position, velocity, angular velocity, rotation, thrusts, delta position,
delta velocity, delta angular velocity, delta rotation) so the dataset can train a forward
dynamics model that maps the current quad state and planner output to the state change.

Place this file at: quad-swarm-rl/scripts/collect_dynamics_dataset.py
"""

import argparse
import glob
import json
import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from gymnasium import spaces

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env


OBS_KEY = "obs"


def load_cfg(train_dir: str, experiment: str) -> AttrDict:
    """Load the saved config.json from a training run into an AttrDict."""
    cfg_path = os.path.join(train_dir, experiment, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _to_attr(obj):
        if isinstance(obj, dict):
            return AttrDict({k: _to_attr(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_attr(v) for v in obj]
        return obj

    cfg = _to_attr(data)
    cfg.train_dir = train_dir
    cfg.experiment = experiment
    cfg.device = "cpu"
    return cfg


def latest_checkpoint(train_dir: str, experiment: str, policy_index: int = 0) -> str:
    """Return the newest checkpoint path for the specified policy."""
    ckpt_dir = Learner.checkpoint_dir(
        AttrDict(train_dir=train_dir, experiment=experiment), policy_index
    )
    pattern = os.path.join(ckpt_dir, "checkpoint_*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {pattern}")
    return candidates[-1]


def _dict_obs_space(space):
    if isinstance(space, spaces.Dict):
        return space
    return spaces.Dict({OBS_KEY: space})


def load_actor(cfg: AttrDict, obs_space, act_space, checkpoint_path: str, device: torch.device):
    """Instantiate an ActorCritic and load weights from a checkpoint."""
    dict_obs_space = _dict_obs_space(obs_space)
    actor = create_actor_critic(cfg, dict_obs_space, act_space)
    actor.model_to_device(device)
    state = Learner.load_checkpoint([checkpoint_path], device)
    actor.load_state_dict(state["model"])
    actor.eval()
    return actor


def extract_full_state(env_state) -> Dict[str, np.ndarray]:
    """Fetch position, velocity, angular velocity, and rotation for every quad."""
    positions = []
    velocities = []
    angular_velocities = []
    rotations = []
    for quad in env_state.envs:
        dynamics = quad.dynamics
        positions.append(np.asarray(dynamics.pos, dtype=np.float32))
        velocities.append(np.asarray(dynamics.vel, dtype=np.float32))
        angular_velocities.append(np.asarray(dynamics.omega, dtype=np.float32))
        rotations.append(np.asarray(dynamics.rot, dtype=np.float32).reshape(-1))

    return {
        "position": np.stack(positions, axis=0),
        "velocity": np.stack(velocities, axis=0),
        "angular_velocity": np.stack(angular_velocities, axis=0),
        "rotation": np.stack(rotations, axis=0),
    }


def ensure_output_dir(path: str) -> None:
    """Create the parent directory for an output path if needed."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Collect quadrotor dynamics dataset")
    parser.add_argument("--train_dir", required=True,
                        help="Train dir that holds the policy checkpoints")
    parser.add_argument("--experiment", required=True,
                        help="Experiment name for the pre-trained policy")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional explicit checkpoint path (defaults to latest)")
    parser.add_argument("--num_steps", type=int, default=10000,
                        help="How many environment steps to simulate")
    parser.add_argument("--output_path", default="dynamic_diff_goal_dataset.npz",
                        help="Where to store the collected dataset (npz format)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic (mean) actions instead of sampling from the policy")
    parser.add_argument("--num_agents", type=int, default=10,
                        help="Number of quadrotors to load in the environment")
    parser.add_argument("--neighbor_visible_num", type=int, default=None,
                        help="Number of neighbors exposed in observations (defaults to num_agents - 1, capped at 9)")
    args = parser.parse_args()

    if args.num_agents <= 0:
        raise ValueError("num_agents must be positive.")

    if args.neighbor_visible_num is not None and args.neighbor_visible_num < 0:
        raise ValueError("neighbor_visible_num must be non-negative when provided.")

    torch.set_grad_enabled(False)
    register_swarm_components()

    neighbor_visible = args.neighbor_visible_num
    if neighbor_visible is None:
        neighbor_visible = min(9, max(0, args.num_agents - 1))
    else:
        neighbor_visible = min(args.neighbor_visible_num, max(0, args.num_agents - 1))

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=dynamic_diff_goal",
        f"--quads_num_agents={args.num_agents}",
        f"--quads_neighbor_visible_num={neighbor_visible}",
        "--quads_neighbor_obs_type=pos_vel",
        "--quads_collision_reward=8.0",
        "--quads_collision_hitbox_radius=2.5",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=False",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    device = torch.device(eval_cfg.device)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)
    if env.num_agents != args.num_agents:
        raise RuntimeError(
            f"Environment instantiated with {env.num_agents} agents, but num_agents={args.num_agents} was requested."
        )

    cfg_multi = load_cfg(args.train_dir, args.experiment)
    checkpoint_path = args.checkpoint or latest_checkpoint(args.train_dir, args.experiment, policy_index=0)
    actor = load_actor(cfg_multi, env.observation_space, env.action_space, checkpoint_path, device)

    rnn_states = torch.zeros((env.num_agents, get_rnn_size(cfg_multi)), dtype=torch.float32, device=device)

    data_buffers: Dict[str, list] = {
        "position": [],
        "velocity": [],
        "angular_velocity": [],
        "rotation": [],
        "thrusts": [],
        "delta_position": [],
        "delta_velocity": [],
        "delta_angular_velocity": [],
        "delta_rotation": [],
    }

    obs, _ = env.reset()
    for _ in tqdm(range(args.num_steps), desc="Collecting rollouts"):
        env_state = env.unwrapped
        current_state = extract_full_state(env_state)
        obs_np = np.asarray(obs)

        obs_dict = {OBS_KEY: obs_np}
        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(actor, obs_dict)
            policy_out = actor(normalized_obs, rnn_states)

        actions = policy_out["actions"]
        rnn_states = policy_out["new_rnn_states"]
        if args.deterministic:
            actions = argmax_actions(actor.action_distribution())
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        actions_np = actions.detach().cpu().numpy()

        obs, _, terminated, truncated, _ = env.step(actions_np)
        next_state = extract_full_state(env_state)

        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        done_mask = np.logical_or(terminated, truncated)
        valid_mask = ~done_mask

        if np.any(valid_mask):
            data_buffers["position"].append(current_state["position"][valid_mask])
            data_buffers["velocity"].append(current_state["velocity"][valid_mask])
            data_buffers["angular_velocity"].append(current_state["angular_velocity"][valid_mask])
            data_buffers["rotation"].append(current_state["rotation"][valid_mask])
            data_buffers["thrusts"].append(actions_np[valid_mask].astype(np.float32))

            data_buffers["delta_position"].append(next_state["position"][valid_mask] - current_state["position"][valid_mask])
            data_buffers["delta_velocity"].append(next_state["velocity"][valid_mask] - current_state["velocity"][valid_mask])
            data_buffers["delta_angular_velocity"].append(next_state["angular_velocity"][valid_mask] - current_state["angular_velocity"][valid_mask])
            data_buffers["delta_rotation"].append(next_state["rotation"][valid_mask] - current_state["rotation"][valid_mask])

        if np.any(done_mask):
            done_tensor = torch.as_tensor(done_mask, device=device, dtype=torch.bool)
            rnn_states[done_tensor] = 0.0

        if np.all(done_mask):
            obs, _ = env.reset()

    env.close()

    if not data_buffers["position"]:
        print("[collect_dynamics_dataset] No samples gathered; nothing to save.")
        return

    ensure_output_dir(args.output_path)

    dataset = {
        key: np.concatenate(values, axis=0)
        for key, values in data_buffers.items()
        if values
    }

    np.savez_compressed(args.output_path, **dataset)
    total_samples = dataset["position"].shape[0]
    print(f"[collect_dynamics_dataset] Saved {total_samples} samples to {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
