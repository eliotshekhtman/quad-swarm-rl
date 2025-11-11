"""
Shared utilities for CBF-based evaluation scripts.

Provides data containers and helper routines for loading policies as well as
extracting swarm state snapshots from the quadrotor simulator.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.attr_dict import AttrDict

OBS_KEY = "obs"


@dataclass
class SwarmState:
    """Snapshot of the swarm used when assembling the CBF constraints."""

    positions: np.ndarray  # shape (N, 3)
    velocities: np.ndarray  # shape (N, 3)
    rotations: np.ndarray  # shape (N, 3, 3)


def load_cfg(train_dir: str, experiment: str) -> AttrDict:
    """
    Retrieve the Sample Factory configuration saved alongside a trained policy.

    The returned object mimics the AttrDict passed to the RL training loop and
    carries enough metadata (train_dir, experiment, device) for evaluation.
    """
    cfg_path = os.path.join(train_dir, experiment, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    def to_attr(obj: Any):
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
    """Return the newest checkpoint path saved for ``policy_index``."""
    ckpt_dir = Learner.checkpoint_dir(
        AttrDict(train_dir=train_dir, experiment=experiment), policy_index
    )
    pattern = os.path.join(ckpt_dir, "checkpoint_*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {pattern}")
    return candidates[-1]


def _as_dict_space(space):
    """Sample Factory expects Dict observations; wrap plain Box spaces on-the-fly."""
    if isinstance(space, spaces.Dict):
        return space
    return spaces.Dict({OBS_KEY: space})


def load_actor(cfg: AttrDict, obs_space, act_space, checkpoint_path: str, device: torch.device):
    """Instantiate an ActorCritic network and load weights from ``checkpoint_path``."""
    dict_obs_space = _as_dict_space(obs_space)
    actor = create_actor_critic(cfg, dict_obs_space, act_space)
    actor.model_to_device(device)
    state = Learner.load_checkpoint([checkpoint_path], device)
    actor.load_state_dict(state["model"])
    actor.eval()
    return actor


def get_swarm_state(env) -> SwarmState:
    """
    Pull a consistent snapshot (positions, velocities, attitudes) from each agent.

    The simulator stores the state per-agent in ``env.envs`` where each entry is a
    ``QuadrotorSingle`` with a ``dynamics`` member holding the physical data.
    """
    positions = []
    velocities = []
    rotations = []
    for quad in env.envs:
        dynamics = quad.dynamics
        positions.append(np.asarray(dynamics.pos, dtype=np.float64))
        velocities.append(np.asarray(dynamics.vel, dtype=np.float64))
        rotations.append(np.asarray(dynamics.rot, dtype=np.float64))
    return SwarmState(
        positions=np.stack(positions, axis=0),
        velocities=np.stack(velocities, axis=0),
        rotations=np.stack(rotations, axis=0),
    )


__all__ = [
    "OBS_KEY",
    "SwarmState",
    "get_swarm_state",
    "latest_checkpoint",
    "load_actor",
    "load_cfg",
]
