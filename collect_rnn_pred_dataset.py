#!/usr/bin/env python3
"""
Collect position and velocity sequences from a solo quadrotor controlled by a pre-trained policy.

The script runs the simulator for a specified number of steps, storing consecutive sequences of
`sequence_length` timesteps (default 1,000). When executed for 1,000,000 steps this produces
exactly 1,000 sequences suitable for training recurrent predictors.

Dataset contents (all float32):
    - positions:  [num_sequences, sequence_length, 3]
    - velocities: [num_sequences, sequence_length, 3]

Place this file at: quad-swarm-rl/scripts/collect_solo_sequence_dataset.py
"""

import argparse
import glob
import json
import os
from typing import Tuple

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


def extract_position_velocity(env_state) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch position and velocity for the solo quadrotor."""
    quad = env_state.envs[0]
    dynamics = quad.dynamics
    position = np.asarray(dynamics.pos, dtype=np.float32)
    velocity = np.asarray(dynamics.vel, dtype=np.float32)
    return position, velocity


def ensure_output_dir(path: str) -> None:
    """Create the parent directory for an output path if needed."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Collect solo quadrotor position/velocity sequences"
    )
    parser.add_argument(
        "--train_dir",
        required=True,
        help="Training directory containing saved policy checkpoints",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name for the pre-trained policy",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path (defaults to latest available)",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=1_000_000,
        help="Total number of environment steps to simulate",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1_000,
        help="Number of steps per stored sequence",
    )
    parser.add_argument(
        "--output_path",
        default="rnn_dataset.npz",
        help="Path where the NPZ dataset will be written",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic (argmax) actions instead of sampling",
    )
    args = parser.parse_args()

    if args.sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if args.total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if args.total_steps % args.sequence_length != 0:
        raise ValueError(
            f"total_steps ({args.total_steps}) must be divisible by sequence_length ({args.sequence_length})."
        )

    torch.set_grad_enabled(False)
    register_swarm_components()

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=dynamic_diff_goal",
        "--quads_num_agents=1",
        "--quads_neighbor_visible_num=0",
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
    if env.num_agents != 1:
        raise RuntimeError(
            f"Environment instantiated with {env.num_agents} agents, but 1 agent was requested."
        )

    cfg_solo = load_cfg(args.train_dir, args.experiment)
    checkpoint_path = args.checkpoint or latest_checkpoint(
        args.train_dir, args.experiment, policy_index=0
    )
    actor = load_actor(cfg_solo, env.observation_space, env.action_space, checkpoint_path, device)
    rnn_states = torch.zeros(
        (env.num_agents, get_rnn_size(cfg_solo)), dtype=torch.float32, device=device
    )

    num_sequences = args.total_steps // args.sequence_length
    positions = np.zeros((num_sequences, args.sequence_length, 3), dtype=np.float32)
    velocities = np.zeros_like(positions)

    obs, _ = env.reset()

    seq_idx = 0
    step_in_seq = 0

    progress = tqdm(range(args.total_steps), desc="Collecting sequences")
    for _ in progress:
        env_state = env.unwrapped
        position, velocity = extract_position_velocity(env_state)

        positions[seq_idx, step_in_seq, :] = position
        velocities[seq_idx, step_in_seq, :] = velocity

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
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        done_mask = np.logical_or(terminated, truncated)

        if np.any(done_mask):
            done_tensor = torch.as_tensor(done_mask, device=device, dtype=torch.bool)
            rnn_states[done_tensor] = 0.0
            if np.all(done_mask):
                obs, _ = env.reset()

        step_in_seq += 1
        if step_in_seq == args.sequence_length:
            seq_idx += 1
            step_in_seq = 0

    env.close()

    ensure_output_dir(args.output_path)
    np.savez_compressed(
        args.output_path,
        positions=positions,
        velocities=velocities,
        sequence_length=np.int32(args.sequence_length),
        total_steps=np.int64(args.total_steps),
    )
    message = (
        f"[collect_solo_sequence_dataset] Saved {num_sequences} sequences "
        f"of length {args.sequence_length} to {os.path.abspath(args.output_path)}"
    )
    print(message)


if __name__ == "__main__":
    main()

