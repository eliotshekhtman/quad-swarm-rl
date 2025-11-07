"""
Utilities for capturing and restoring quadrotor simulator state so rollouts can be
replayed deterministically from an identical starting point.

Typical usage::

    snapshot = capture_env_snapshot(env)
    # ... advance the environment ...
    restore_env_snapshot(env, snapshot)
    # env is now exactly where it was when the snapshot was taken

    clone = clone_env_from_snapshot(snapshot)
    # run alternative rollouts starting from the same state without mutating ``env``
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class RNGSnapshot:
    """Container for Python, NumPy, and Torch RNG states."""

    python_state: Any
    numpy_state: tuple
    torch_state: torch.Tensor
    torch_cuda_state: Optional[list]


@dataclass
class EnvSnapshot:
    """
    Captured simulator state consisting of a deep copy of the unwrapped environment
    and matching RNG states.
    """

    env_state: Any
    rng: RNGSnapshot


def _snapshot_rng() -> RNGSnapshot:
    """Record current RNG states for Python, NumPy, and Torch."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.get_rng_state_all()
    else:
        torch_cuda_state = None
    return RNGSnapshot(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_state=torch_state,
        torch_cuda_state=torch_cuda_state,
    )


def _restore_rng(snapshot: RNGSnapshot) -> None:
    """Restore saved RNG states."""
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_state)
    if snapshot.torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_state)


def capture_env_snapshot(env) -> EnvSnapshot:
    """
    Deep-copy the unwrapped simulator along with RNG states.

    Parameters
    ----------
    env :
        Gymnasium environment (possibly wrapped). ``env.unwrapped`` is snapshotted.
    """
    base_env = env.unwrapped
    env_copy = copy.deepcopy(base_env)
    rng_snapshot = _snapshot_rng()
    return EnvSnapshot(env_state=env_copy, rng=rng_snapshot)


def restore_env_snapshot(env, snapshot: EnvSnapshot, restore_rng: bool) -> None:
    """
    Mutate ``env.unwrapped`` in-place to match the captured snapshot.

    Notes
    -----
    - The method replaces the ``__dict__`` of the unwrapped env with a deep copy of the stored state.
    - RNG streams (Python, NumPy, Torch, CUDA) are rewound to the snapshot.
    """
    base_env = env.unwrapped
    restored = copy.deepcopy(snapshot.env_state.__dict__)
    base_env.__dict__.clear()
    base_env.__dict__.update(restored)
    if restore_rng:
        _restore_rng(snapshot.rng)


def clone_env_from_snapshot(snapshot: EnvSnapshot, restore_rng: bool):
    """
    Create a new environment instance initialised with the captured state.

    Returns
    -------
    Env :
        Deep copy of the stored ``QuadrotorEnvMulti`` that can be stepped independently.
    """
    clone = copy.deepcopy(snapshot.env_state)
    if restore_rng:
        _restore_rng(snapshot.rng)
    return clone

