#!/usr/bin/env python3
"""
Evaluate a swarm of eleven quadrotors where the first ten share a trained policy
and the eleventh vehicle is shielded by a quadratic-program-based exponential
control barrier function (ECBF).

The ECBF is derived exactly as in the user-provided formulation: it constrains
the motor-space thrust vector so that the protected vehicle remains outside a
ball of radius ``CBF_RADII[i]`` centred on the i-th teammate.  Each control
cycle the script:

    1. Runs the multi-agent APPO policy for the first ten vehicles.
    2. Runs the solo policy to obtain a reference action for the eleventh.
    3. Converts that reference action into physical thrusts (Newtons).
    4. Builds one linear ECBF inequality per teammate using the current
       positions, velocities, attitude, and gravity.
    5. Solves a convex QP that keeps the thrusts close to the reference while
       respecting actuator bounds and the softened ECBF constraints.
    6. Maps the safe thrust profile back to the environment's native action
       space and steps the simulator.

Place this script under ``quad-swarm-rl/scripts/`` to match the layout of
``mixed_enjoy.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
import torch
from gymnasium import spaces
from tqdm import tqdm

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.huggingface.huggingface_utils import generate_replay_video
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

# ---------------------------------------------------------------------------
# Configuration shared across helper functions
# ---------------------------------------------------------------------------

OBS_KEY = "obs"
NUM_MULTI_AGENTS = 20
TOTAL_AGENTS = NUM_MULTI_AGENTS + 1
SOLO_AGENT_INDEX = NUM_MULTI_AGENTS
NUM_NEIGHBORS = 9  # matches the default Sample Factory quadrotor config

# Barrier parameters (can be adjusted at runtime through ``set_cbf_radii``).
CBF_RADII = np.full(NUM_MULTI_AGENTS, 2, dtype=np.float64)
CBF_K1 = 8.0
CBF_K0 = 2.0
CBF_SLACK_WEIGHT = 1.0e4
CBF_QP_DIAGONAL = np.ones(4, dtype=np.float64)
CBF_QP_SOLVER = "OSQP"

GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81], dtype=np.float64)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SwarmState:
    """Snapshot of the swarm used when assembling the CBF constraints."""

    positions: np.ndarray  # shape (TOTAL_AGENTS, 3)
    velocities: np.ndarray  # shape (TOTAL_AGENTS, 3)
    rotations: np.ndarray  # shape (TOTAL_AGENTS, 3, 3)


# ---------------------------------------------------------------------------
# Utility functions for loading policies and translating actions
# ---------------------------------------------------------------------------

def load_cfg(train_dir: str, experiment: str) -> AttrDict:
    """
    Retrieve the Sample Factory configuration saved alongside a trained policy.

    The returned object mimics the AttrDict passed to the RL training loop and
    carries enough metadata (train_dir, experiment, device) for evaluation.
    """
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


def set_cbf_radii(radii: Sequence[float]) -> None:
    """
    Overwrite the per-teammate exclusion radii used by the barrier.

    ``radii`` must supply exactly ``NUM_MULTI_AGENTS`` non-negative values.
    """
    if len(radii) != NUM_MULTI_AGENTS:
        raise ValueError(f"Expected {NUM_MULTI_AGENTS} radii, received {len(radii)}")
    radii_array = np.asarray(radii, dtype=np.float64)
    if np.any(radii_array < 0.0):
        raise ValueError("CBF radii must be non-negative")
    np.copyto(CBF_RADII, radii_array)


# ---------------------------------------------------------------------------
# Motor command conversions
# ---------------------------------------------------------------------------

def _normalized_to_thrust(norm_cmds: np.ndarray, dynamics) -> np.ndarray:
    """
    Convert environment actions in [0, 1] into per-rotor thrust magnitudes (Newtons).

    The quadrotor dynamics use a convex combination of linear and quadratic curves
    to map the high-level command to thrust.  We delegate the core conversion to
    ``QuadrotorDynamics.angvel2thrust`` so the QP shares the exact actuator model.
    """
    commands = np.clip(np.asarray(norm_cmds, dtype=np.float64), 0.0, 1.0)
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    linearity = np.broadcast_to(np.atleast_1d(linearity), commands.shape).astype(np.float64)
    return thrust_max * dynamics.angvel2thrust(commands, linearity=linearity)


def _thrust_to_normalized(thrusts: np.ndarray, dynamics) -> np.ndarray:
    """
    Invert ``_normalized_to_thrust`` by numerically recovering the [0, 1] motor
    commands whose actuator model yields the requested thrust magnitudes.
    """
    thrusts = np.asarray(thrusts, dtype=np.float64)
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    max_safe = np.clip(thrust_max, 1e-6, None)
    ratio = np.clip(thrusts / max_safe, 0.0, 1.0)
    linearity = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    linearity = np.broadcast_to(np.atleast_1d(linearity), ratio.shape).astype(np.float64)

    def _invert_single(r: float, lin: float) -> float:
        """
        Use a monotone bisection driven entirely by ``angvel2thrust`` so the
        inverse stays coupled to the simulator's actuator curves.
        """
        if abs(lin - 1.0) < 1e-9:
            return r
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            val = float(dynamics.angvel2thrust(np.array([mid]), linearity=np.array([lin]))[0])
            if val < r:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    vectorized = np.vectorize(_invert_single, otypes=[np.float64])
    normalized = vectorized(ratio, linearity)
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ECBF helpers
# ---------------------------------------------------------------------------

def _ecbf_coefficients(
    *,
    solo_pos: np.ndarray,
    solo_vel: np.ndarray,
    solo_rot: np.ndarray,
    teammate_pos: np.ndarray,
    teammate_vel: np.ndarray,
    radius: float,
    mass: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute the ``(a, b, h)`` triple for the ECBF constraint ``a^T u ≥ -b - slack``.

    The derivation follows the supplied formulation:

        h(x) = ||x - p||² - r²
        ḣ(x) = 2 zᵀ v_rel
        ḧ(x) = 2 ||v_rel||² + 2 zᵀ g + (2/m) (zᵀ R e₃) 1ᵀ u

    where:

        z      := solo_pos - teammate_pos
        v_rel  := solo_vel - teammate_vel   (moving obstacle extension)
        g      := GRAVITY_VECTOR
        R e₃   := third column of the body-to-world rotation

    The moving-obstacle extension treats the teammate as a point translating with
    velocity ``teammate_vel``.  This keeps the barrier conservative: if the
    teammate is stationary it reduces to the exact textbook form.
    """
    z_vec = solo_pos - teammate_pos # (x-p)
    v_rel = solo_vel - teammate_vel
    h_value = float(np.dot(z_vec, z_vec) - radius**2) # h
    z_dot_gravity = float(np.dot(z_vec, GRAVITY_VECTOR)) # (x-p)ᵀg
    z_dot_v_rel = float(np.dot(z_vec, solo_vel)) # (x-p)ᵀv
    v_rel_sq = float(np.dot(solo_vel, solo_vel)) # vᵀv
    thrust_axis_world = solo_rot[:, 2] # R e₃
    thrust_alignment = float(np.dot(z_vec, thrust_axis_world)) # (x-p)ᵀRe₃
    c_scale = (2.0 / mass) * thrust_alignment # (2/m) (x-p)ᵀRe₃
    a_vec = c_scale * np.ones(4, dtype=np.float64) # (2/m) (x-p)ᵀRe₃ 1ᵀ
    b_scalar = ( # 2vᵀv + 2vᵀg + a1(2(x-p)ᵀv + a0(h))
        2.0 * v_rel_sq 
        + 2.0 * z_dot_gravity
        + 2.0 * CBF_K1 * z_dot_v_rel
        + CBF_K1 * CBF_K0 * h_value
    )
    return a_vec, b_scalar, h_value


def _solve_cbf_qp(
    *,
    u_ref_thrust: np.ndarray,
    swarm_state: SwarmState,
    solo_index: int,
    mass: float,
    thrust_bounds: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Build and solve the ECBF quadratic program described in the task statement.

    Decision variables
    ------------------
    - ``u`` ∈ ℝ⁴ : per-motor thrusts.
    - ``slack`` ≥ 0 : shared softening variable.

    Objective
    ---------
    minimise ‖u - u_ref‖²_W + CBF_SLACK_WEIGHT · slack

    Constraints
    -----------
    - One inequality per teammate: ``a_iᵀ u ≥ -b_i - slack``.
    - Elementwise thrust bounds ``u_min ≤ u ≤ u_max``.
    - ``slack ≥ 0``.
    """
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64)
    u_min, u_max = thrust_bounds
    solo_pos = swarm_state.positions[solo_index]
    solo_vel = swarm_state.velocities[solo_index]
    solo_rot = swarm_state.rotations[solo_index]

    constraints: List[cp.Constraint] = []
    u_var = cp.Variable(4)
    slack = cp.Variable()

    for teammate_idx in range(NUM_MULTI_AGENTS):
        if teammate_idx == solo_index:
            continue
        radius = float(CBF_RADII[teammate_idx])
        if radius <= 0.0:
            continue
        teammate_pos = swarm_state.positions[teammate_idx]
        teammate_vel = swarm_state.velocities[teammate_idx]
        a_vec, b_scalar, _ = _ecbf_coefficients(
            solo_pos=solo_pos,
            solo_vel=solo_vel,
            solo_rot=solo_rot,
            teammate_pos=teammate_pos,
            teammate_vel=teammate_vel,
            radius=radius,
            mass=mass,
        )
        constraints.append(a_vec @ u_var >= -b_scalar - slack)

    if len(constraints) == 0:
        return np.clip(u_ref, u_min, u_max)

    weight_matrix = np.diag(CBF_QP_DIAGONAL)
    objective = cp.quad_form(u_var - u_ref, weight_matrix) + CBF_SLACK_WEIGHT * slack
    constraints.extend(
        [
            u_var >= u_min,
            u_var <= u_max,
            slack >= 0,
        ]
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver = getattr(cp, CBF_QP_SOLVER, cp.OSQP)
    try:
        problem.solve(solver=solver, warm_start=True, verbose=False)
    except cp.SolverError:
        return np.clip(u_ref, u_min, u_max)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return np.clip(u_ref, u_min, u_max)
    solution = np.array(u_var.value, dtype=np.float64)
    return np.clip(solution, u_min, u_max)


def apply_cbf_filter(
    base_action: np.ndarray,
    *,
    solo_index: int,
    env_state,
    swarm_state: SwarmState,
) -> np.ndarray:
    """
    Wrap the raw solo-policy action with the ECBF safety filter.

    Parameters
    ----------
    base_action : np.ndarray
        Motor command in [-1, 1]⁴ produced by the solo policy.
    solo_index : int
        Index of the protected agent (``NUM_MULTI_AGENTS``).
    env_state :
        Vectorised environment used to access the simulator dynamics.
    swarm_state : SwarmState
        Pre-computed positions, velocities, and orientations for the current step.
    """
    quad = env_state.envs[solo_index]
    dynamics = quad.dynamics

    # Action space conversions: [-1, 1] → [0, 1] → Newton thrusts.
    base_action = np.asarray(base_action, dtype=np.float64)
    normalized = np.clip(0.5 * (base_action + 1.0), 0.0, 1.0)
    u_ref_thrust = _normalized_to_thrust(normalized, dynamics)

    u_min = np.zeros(4, dtype=np.float64)
    u_max = np.asarray(dynamics.thrust_max, dtype=np.float64)
    safe_thrust = _solve_cbf_qp(
        u_ref_thrust=u_ref_thrust,
        swarm_state=swarm_state,
        solo_index=solo_index,
        mass=float(dynamics.mass),
        thrust_bounds=(u_min, u_max),
    )

    # Convert Newton thrust back to the environment's action space.
    safe_normalized = _thrust_to_normalized(safe_thrust, dynamics)
    safe_action = 2.0 * safe_normalized - 1.0
    return np.clip(safe_action.astype(np.float32), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a 10+1 quadrotor swarm where the solo vehicle is shielded by a "
            "control-barrier-function QP acting directly in motor thrust space."
        )
    )
    parser.add_argument("--multi_train_dir", default='train_dir')
    parser.add_argument("--multi_experiment", required=True)
    parser.add_argument("--solo_train_dir", default='train_dir')
    parser.add_argument("--solo_experiment", required=True)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--video_name", default="cbf_guard_replay.mp4")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--disable_cbf", action="store_true", help="Bypass the QP for debugging.")
    args = parser.parse_args()

    register_swarm_components()
    torch.set_grad_enabled(False)

    if os.path.isabs(args.video_name):
        video_dir = os.path.dirname(args.video_name) or "."
        video_file = os.path.basename(args.video_name)
    else:
        video_dir = os.path.join(args.multi_train_dir, args.multi_experiment)
        video_file = args.video_name
    os.makedirs(video_dir, exist_ok=True)
    video_frames: List[np.ndarray] = []

    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        "--quads_mode=dynamic_diff_goal",
        f"--quads_num_agents={TOTAL_AGENTS}",
        f"--quads_neighbor_visible_num={NUM_NEIGHBORS}",
        "--quads_neighbor_obs_type=pos_vel",
        "--quads_collision_reward=8.0",
        "--quads_collision_hitbox_radius=2.5",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_render=True",
        "--quads_view_mode=topdown",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    device = torch.device(eval_cfg.device)
    render_mode = "rgb_array"
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=render_mode)

    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
    multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, device)

    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, device)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    multi_rnn_states = torch.zeros(
        (NUM_MULTI_AGENTS, get_rnn_size(cfg_multi)), dtype=torch.float32, device=device
    )
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=device)

    obs, _ = env.reset()
    dones = np.zeros(env.num_agents, dtype=bool)
    solo_collision_count = 0

    # First frame
    frame = env.render()
    if frame is not None:
        video_frames.append(frame.copy())

    progress_bar = tqdm(range(args.max_steps))
    for step in progress_bar:
        obs_np = np.asarray(obs)

        obs_multi_dict = {OBS_KEY: obs_np[:NUM_MULTI_AGENTS]}
        with torch.no_grad():
            normalized_multi = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
            policy_multi = multi_actor(normalized_multi, multi_rnn_states)
        actions_multi = policy_multi["actions"]
        multi_rnn_states = policy_multi["new_rnn_states"]
        if args.deterministic:
            actions_multi = argmax_actions(multi_actor.action_distribution())
        if actions_multi.dim() == 1:
            actions_multi = actions_multi.unsqueeze(-1)
        actions_multi = actions_multi.detach().cpu().numpy()

        obs_solo_self = obs_np[SOLO_AGENT_INDEX, :solo_obs_dim]
        obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
        with torch.no_grad():
            normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
            policy_solo = solo_actor(normalized_solo, solo_rnn_states)
        action_solo = policy_solo["actions"]
        solo_rnn_states = policy_solo["new_rnn_states"]
        if args.deterministic:
            action_solo = argmax_actions(solo_actor.action_distribution())
        if action_solo.dim() == 1:
            action_solo = action_solo.unsqueeze(0)
        action_solo = action_solo.detach().cpu().numpy()[0]

        swarm_state = get_swarm_state(env.unwrapped)
        if not args.disable_cbf:
            action_solo = apply_cbf_filter(
                base_action=action_solo,
                solo_index=SOLO_AGENT_INDEX,
                env_state=env.unwrapped,
                swarm_state=swarm_state,
            )

        actions = np.vstack([actions_multi, action_solo[None, :]])
        obs, rewards, terminated, truncated, infos = env.step(actions)

        solo_info_rewards = infos[SOLO_AGENT_INDEX].get("rewards", {})
        if solo_info_rewards.get("rewraw_quadcol", 0.0) < 0.0:
            solo_collision_count += 1
            progress_bar.set_postfix_str(f"crashes={solo_collision_count}")

        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        dones = np.logical_or(terminated, truncated)

        frame = env.render()
        if frame is not None:
            video_frames.append(frame.copy())

        if np.all(dones):
            obs, _ = env.reset()
            dones = np.zeros(env.num_agents, dtype=bool)

    env.close()

    if len(video_frames) > 0:
        video_cfg = AttrDict(video_name=video_file)
        generate_replay_video(video_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_dir, video_file))
        print(f"[cbf_guard_enjoy] Video saved to {final_path}")

    print(f"[cbf_guard_enjoy] Solo drone collisions: {solo_collision_count}")


if __name__ == "__main__":
    main()
