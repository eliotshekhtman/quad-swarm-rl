#!/usr/bin/env python3
"""
Collect quadrotor trajectories in the patrol_dual_goal scenario with deterministic resets.

This script mirrors the evaluation flow of ``mixed_enjoy.py`` while enforcing:
    * A user-specified random seed that is re-applied before every environment reset so the
      patrol scenario regenerates the same goals each episode.
    * Manual restoration of the initial quad states (position, velocity, rotation, body rates)
      after every reset, ensuring agents respawn exactly where they started on the first episode.
    * Per-timestep logging of each quad's position and velocity to an ``NPZ`` archive.
    * Loading of a pretrained recurrent predictor (unused for control, but instantiated and ready).

Initial states detected on the first episode are persisted to ``initial_states.json`` inside the
output experiment directory so subsequent runs can reproduce the exact spawn configuration.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from tqdm import tqdm

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env

from pretrain_rnn_predictor import RNNPredictor


OBS_KEY = "obs"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuadState:
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3)
    omega: np.ndarray     # (3,)
    goal: np.ndarray      # (3,)


# ---------------------------------------------------------------------------
# Helpers for Sample Factory actors
# ---------------------------------------------------------------------------

def load_cfg(train_dir: str, experiment: str) -> AttrDict:
    """Load the stored ``config.json`` for a Sample Factory experiment."""
    cfg_path = os.path.join(train_dir, experiment, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    def to_attr(obj):
        if isinstance(obj, dict):
            return AttrDict({k: to_attr(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [to_attr(v) for v in obj]
        return obj

    cfg = to_attr(data)
    cfg.train_dir = train_dir
    cfg.experiment = experiment
    cfg.device = "cpu"
    return cfg


def latest_checkpoint(train_dir: str, experiment: str, policy_index: int = 0) -> str:
    ckpt_dir = Learner.checkpoint_dir(
        AttrDict(train_dir=train_dir, experiment=experiment),
        policy_index,
    )
    pattern = os.path.join(ckpt_dir, "checkpoint_*")
    candidates = sorted([path for path in glob.glob(pattern)])
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return candidates[-1]


def load_actor(cfg: AttrDict, obs_space, act_space, checkpoint_path: str, device: torch.device):
    if isinstance(obs_space, spaces.Dict):
        dict_obs_space = obs_space
    else:
        dict_obs_space = spaces.Dict({OBS_KEY: obs_space})
    actor = create_actor_critic(cfg, dict_obs_space, act_space)
    actor.model_to_device(device)
    state = Learner.load_checkpoint([checkpoint_path], device)
    actor.load_state_dict(state["model"])
    actor.eval()
    return actor


# ---------------------------------------------------------------------------
# Environment state utilities
# ---------------------------------------------------------------------------

def capture_initial_states(env) -> List[QuadState]:
    """Snapshot the current state (pos, vel, rot, omega, goal) of each quad."""
    states = []
    for quad in env.envs:
        dynamics = quad.dynamics
        state = QuadState(
            position=np.asarray(dynamics.pos, dtype=np.float64),
            velocity=np.asarray(dynamics.vel, dtype=np.float64),
            rotation=np.asarray(dynamics.rot, dtype=np.float64),
            omega=np.asarray(dynamics.omega, dtype=np.float64),
            goal=np.asarray(quad.goal, dtype=np.float64),
        )
        states.append(state)
    return states


def quad_state_to_serialisable(state: QuadState) -> Dict[str, list]:
    """Convert a QuadState into JSON-friendly lists."""
    return {
        "position": state.position.tolist(),
        "velocity": state.velocity.tolist(),
        "rotation": state.rotation.tolist(),
        "omega": state.omega.tolist(),
        "goal": state.goal.tolist(),
    }


def quad_state_from_dict(data: Dict[str, Sequence[float]]) -> QuadState:
    """Rehydrate a QuadState from ``initial_states.json``."""
    return QuadState(
        position=np.asarray(data["position"], dtype=np.float64),
        velocity=np.asarray(data["velocity"], dtype=np.float64),
        rotation=np.asarray(data["rotation"], dtype=np.float64),
        omega=np.asarray(data["omega"], dtype=np.float64),
        goal=np.asarray(data["goal"], dtype=np.float64),
    )


def apply_initial_states(env, states: List[QuadState]) -> List[np.ndarray]:
    """
    Force each quadrotor to the provided state and rebuild the observation list
    returned to the control loop.
    """
    if len(states) != len(env.envs):
        raise ValueError("Number of stored initial states does not match number of agents.")

    for idx, (quad, state) in enumerate(zip(env.envs, states)):
        quad.dynamics.set_state(state.position, state.velocity, state.rotation, state.omega)
        quad.dynamics.reset()
        quad.dynamics.on_floor = False
        quad.dynamics.crashed_floor = quad.dynamics.crashed_wall = quad.dynamics.crashed_ceiling = False
        quad.tick = 0
        quad.actions = [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)]
        quad.goal = state.goal.copy()
        env.pos[idx, :] = quad.dynamics.pos
        env.vel[idx, :] = quad.dynamics.vel

    obs = [quad.state_vector(quad) for quad in env.envs]
    if env.num_use_neighbor_obs > 0:
        obs = env.add_neighborhood_obs(obs)
    return obs


def set_global_seed(seed: int) -> None:
    """Synchronise Python, NumPy, and Torch RNGs to the chosen seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reseed_quads(env, base_seed: int) -> None:
    """Reseed each QuadrotorSingle for deterministic spawn reproduction."""
    for idx, quad in enumerate(env.envs):
        quad._seed(base_seed + idx)


def deterministic_reset(env, seed: int, stored_states: Optional[List[QuadState]]) -> Tuple[np.ndarray, List[QuadState]]:
    """
    Reset the environment while enforcing deterministic scenario generation and
    optionally reapplying saved initial states.
    """
    set_global_seed(seed)
    reseed_quads(env, seed)
    obs, _ = env.reset()
    if stored_states is None:
        states = capture_initial_states(env)
        return np.asarray(obs), states
    obs = apply_initial_states(env.unwrapped, stored_states)
    return np.asarray(obs), stored_states


def extract_positions_velocities(env) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked positions and velocities (num_agents, 3)."""
    positions, velocities = [], []
    for quad in env.envs:
        dynamics = quad.dynamics
        positions.append(np.asarray(dynamics.pos, dtype=np.float64))
        velocities.append(np.asarray(dynamics.vel, dtype=np.float64))
    return np.stack(positions, axis=0), np.stack(velocities, axis=0)


# ---------------------------------------------------------------------------
# RNN predictor loading
# ---------------------------------------------------------------------------

def load_rnn_checkpoint(path: str, device: torch.device) -> RNNPredictor:
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("config")
    if cfg is None:
        raise KeyError("Checkpoint missing 'config' field required to rebuild the model.")
    input_dim = cfg["hidden_size"] if False else None  # placeholder to satisfy linter
    input_dim = checkpoint["model_state_dict"]["head.bias"].shape[0]
    model = RNNPredictor(
        input_dim=input_dim,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        rnn_type=cfg["rnn_type"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model.config = cfg
    return model


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic patrol_dual_goal data collection.")
    parser.add_argument("--multi_train_dir", required=True, help="Directory containing the trained multi-agent policy.")
    parser.add_argument("--multi_experiment", required=True, help="Experiment name for the multi-agent policy.")
    parser.add_argument("--train_dir", required=True, help="Base directory to store the new conformal experiment.")
    parser.add_argument("--experiment_name", required=True, help="Subdirectory under train_dir for outputs.")
    parser.add_argument("--predictor_checkpoint", required=True, help="Path to a pretrained RNN predictor checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Seed applied before every reset to reproduce goals.")
    parser.add_argument("--num_runs", type=int, default=20, help="Total number of resets to collect.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Desired probability of error")
    return parser.parse_args()


def ensure_experiment_dir(base_dir: str, name: str) -> str:
    experiment_dir = os.path.join(base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_initial_states(path: str, states: List[QuadState]) -> None:
    payload = [quad_state_to_serialisable(s) for s in states]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_initial_states(path: str) -> Optional[List[QuadState]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [quad_state_from_dict(entry) for entry in payload]


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    register_swarm_components()

    experiment_dir = ensure_experiment_dir(args.train_dir, args.experiment_name)
    initial_state_path = os.path.join(experiment_dir, "initial_states.json")

    device = torch.device("cpu")

    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=patrol_dual_goal",
        f"--quads_num_agents={cfg_multi.quads_num_agents}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_use_numba=False",
        "--quads_render=False",
        "--max_num_episodes=1",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)
    num_agents = int(eval_cfg.quads_num_agents)

    multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
    multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, device)
    multi_rnn_states = torch.zeros(
        (num_agents, get_rnn_size(cfg_multi)),
        dtype=torch.float32,
        device=device,
    )

    stored_states = load_initial_states(initial_state_path)
    obs, stored_states = deterministic_reset(env, args.seed, stored_states)

    if not os.path.exists(initial_state_path):
        save_initial_states(initial_state_path, stored_states)

    pos, vel = extract_positions_velocities(env.unwrapped)
    logs = {}
    for i, state in enumerate(stored_states):
        logs[i] = []

    progress = tqdm(range(args.num_runs + 1))
    for run in progress:
        for i, state in enumerate(stored_states):
            logs[i].append({
                "position": [state.position],
                "velocity": [state.velocity]
            })
        done = False
        step_num = 0
        while not done:
            progress.set_postfix_str(f"step={step_num}")
            obs_dict = {OBS_KEY: np.asarray(obs, dtype=np.float32)}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(multi_actor, obs_dict)
                policy_output = multi_actor(normalized_obs, multi_rnn_states)
            actions = policy_output["actions"]
            multi_rnn_states = policy_output["new_rnn_states"]
            if run == args.num_runs: # For the last run, deterministic
                actions = argmax_actions(multi_actor.action_distribution())
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
            actions_np = actions.detach().cpu().numpy()

            obs, rewards, terminated, truncated, infos = env.step(actions_np)

            pos, vel = extract_positions_velocities(env.unwrapped)
            for i in range(num_agents):
                logs[i][-1]["position"].append(pos[i])
                logs[i][-1]["velocity"].append(vel[i])

            terminated = np.asarray(terminated)
            truncated = np.asarray(truncated)
            done = np.all(np.logical_or(terminated, truncated))
            step_num += 1
        # Reset in between runs
        obs, stored_states = deterministic_reset(env, args.seed, stored_states)

    env.close()

    for agent_id in range(num_agents):
        # Fine-tune it just a little for the actual path
        predictor = load_rnn_checkpoint(args.predictor_checkpoint, device)
        deterministic_run = np.concatenate(
            [logs[agent_id][-1]["position"], logs[agent_id][-1]["velocity"]],
            axis=-1,
        ).astype(np.float32)

        predictor.train()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=5e-4)
        criterion = nn.MSELoss()
        print(f'Training agent {agent_id}')
        with torch.enable_grad():
            for _ in tqdm(range(3)):
                sequence_tensor = torch.from_numpy(deterministic_run).to(
                    device=device, dtype=torch.float32
                )
                inputs = sequence_tensor[:-1].unsqueeze(0)
                targets = sequence_tensor[1:].unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                preds = predictor(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                optimizer.step()
        predictor.eval()

        print('Collecting trajectory-level nonconformity scores')
        # Collect trajectory-level nonconformity scores
        scores = []
        progress = tqdm(logs[agent_id][:-1])
        for run_log in progress:
            score = 0  # Lowerbound on possible nonconformity score
            # Warm it up with the first 50 steps, and collect predictions for the rest
            run = np.concatenate(
                [run_log["position"], run_log["velocity"]],
                axis=-1,
            ).astype(np.float32)
            warmup = min(50, len(run))
            predictions = [run[idx].copy() for idx in range(warmup)]
            for i in range(warmup, len(run)):
                history_arr = np.asarray(predictions, dtype=np.float32)
                history = torch.from_numpy(history_arr).unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                pred = predictor(history)[0, -1].detach().cpu().numpy()
                predictions.append(pred.astype(np.float32))
                # L2 norm difference between prediction and what really happened
                score = max(score, np.linalg.norm(pred - run[i]))
            scores.append(score)
        scores.sort() # Make sure the index isn't equal to len
        # Just want to visually check that this makes sense
        print(f'Scores for agent {agent_id}: ', scores)
        conformal_radius = scores[np.ceil(len(scores) * (1 - args.alpha)) - 1]

        finetuned_path = os.path.join(experiment_dir, f"predictor_{agent_id}.pt")
        payload = {
            "model_state_dict": {k: v.detach().cpu() for k, v in predictor.state_dict().items()},
            "conformal_radius": conformal_radius
        }
        if hasattr(predictor, "config"):
            payload["config"] = predictor.config
        torch.save(payload, finetuned_path)

    metadata = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "multi_train_dir": args.multi_train_dir,
        "multi_experiment": args.multi_experiment,
        "predictor_checkpoint": args.predictor_checkpoint,
        "deterministic_actions": bool(args.deterministic),
        "num_agents": int(eval_cfg.quads_num_agents),
    }
    with open(os.path.join(experiment_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
