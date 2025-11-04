#!/usr/bin/env python3
"""
Evaluate 11 quadrotors where the first 10 use a shared multi-agent policy
and the 11th runs a separate policy plus custom logic.

Place this file at: quad-swarm-rl/scripts/mixed_enjoy.py
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Sequence, Tuple
from tqdm import tqdm

import cvxpy as cp
import numpy as np
import torch
from gymnasium import spaces

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.huggingface.huggingface_utils import generate_replay_video

from swarm_rl.train import register_swarm_components, parse_swarm_cfg
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env

OBS_KEY = "obs"
NUM_MULTI_AGENTS = 20
NUM_NEIGHBORS = 9
TOTAL_AGENTS = NUM_MULTI_AGENTS + 1
SOLO_AGENT_INDEX = NUM_MULTI_AGENTS

# CBF configuration shared across evaluation runs.  The radii can be overridden at runtime
# via `set_cbf_radii`, enabling the user to dial in different personal-space envelopes
# for each of the first ten agents without editing the rest of the script.

CBF_RADII = np.full(NUM_MULTI_AGENTS, 2, dtype=np.float32) # at least 2.5
CBF_K1 = 8.0
CBF_K0 = 12.0
CBF_SLACK_WEIGHT = 1e4
CBF_QP_DIAGONAL = np.ones(4, dtype=np.float64)
CBF_QP_SOLVER = "OSQP"
VEL_HISTORY_LEN = 4


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_cfg(train_dir: str, experiment: str) -> AttrDict:
    """Load the saved config.json and convert it into an AttrDict."""
    cfg_path = os.path.join(train_dir, experiment, "config.json")
    with open(cfg_path, "r") as f:
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
    """Return the newest checkpoint path for the given policy."""
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


def get_drone_states(env) -> Tuple[np.ndarray, np.ndarray]:
    """Extract positions and velocities from each underlying QuadrotorSingle."""
    positions = np.stack([quad.dynamics.pos.copy() for quad in env.envs], axis=0)
    velocities = np.stack([quad.dynamics.vel.copy() for quad in env.envs], axis=0)
    return positions, velocities


def set_cbf_radii(radii: Sequence[float]) -> None:
    """Update the safety radii assigned to each of the first 10 drones."""
    # Ensure the caller supplied exactly one radius per multi-agent teammate.
    if len(radii) != NUM_MULTI_AGENTS:
        raise ValueError(f"Expected {NUM_MULTI_AGENTS} radii, received {len(radii)}")
    # Convert to a numpy array for efficient vectorised checks/updates.
    radii_arr = np.asarray(radii, dtype=np.float32)
    # Guard against negative radii which would invalidate the barrier.
    if np.any(radii_arr < 0):
        raise ValueError("CBF radii must be non-negative")
    # Copy the validated radii into the global configuration vector.
    np.copyto(CBF_RADII, radii_arr)


def _normalized_to_thrust(norm_cmds: np.ndarray, dynamics) -> np.ndarray:
    """
    Map normalized motor commands in [0, 1] to per-rotor thrust values (Newtons).

    The simulator's raw controller applies the elementwise mapping
    `T_i = thrust_max_i * ((1 - linearity) * u_i^2 + linearity * u_i)` once the high-level
    policy has produced normalized commands `u`.  We invert only the static portion of that
    pipeline here so that the quadratic program operates in the physical thrust space that the
    paper's derivation assumes.
    """
    # Clamp and cast the incoming commands so downstream algebra stays well-defined.
    norm_cmds = np.clip(np.asarray(norm_cmds, dtype=np.float64), 0.0, 1.0)
    # Fetch the per-motor thrust scaling factors from the simulator dynamics.
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    # Retrieve the CrazyFlie-style linearity parameter controlling the static map.
    linearity_raw = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    # Broadcast linearity so it matches the elementwise shape of the commands.
    linearity = np.broadcast_to(np.atleast_1d(linearity_raw), norm_cmds.shape).astype(np.float64)

    # Apply the quadratic component of the motor curve.
    quadratic_term = (1.0 - linearity) * np.square(norm_cmds)
    # Apply the linear component of the motor curve.
    linear_term = linearity * norm_cmds
    # Scale by thrust_max to obtain physical thrust magnitudes (Newtons).
    return thrust_max * (quadratic_term + linear_term)


def _thrust_to_normalized(thrusts: np.ndarray, dynamics) -> np.ndarray:
    """
    Invert `_normalized_to_thrust`: recover normalized commands that would yield the requested
    per-rotor thrust magnitudes when passed through the simulator's motor model.
    """
    # Work with float64 arrays to keep the inverse map numerically stable.
    thrusts = np.asarray(thrusts, dtype=np.float64)
    # Read back the motor scaling factors used by the forward map.
    thrust_max = np.asarray(getattr(dynamics, "thrust_max"), dtype=np.float64)
    # Normalise the desired thrusts into the [0, 1] range before inversion.
    ratio = np.clip(thrusts / np.clip(thrust_max, 1e-6, None), 0.0, 1.0)

    # Retrieve the linearity parameter and broadcast it to match vector shapes.
    linearity_raw = np.asarray(getattr(dynamics, "motor_linearity", 1.0), dtype=np.float64)
    linearity = np.broadcast_to(np.atleast_1d(linearity_raw), ratio.shape).astype(np.float64)
    # Identify the quadratic (a) and linear (b) coefficients of the actuator map.
    a = 1.0 - linearity
    b = linearity

    # Prepare the output array that will hold the inverted commands.
    norm_cmds = np.empty_like(ratio)
    # Handle motors that effectively follow a linear mapping.
    linear_mask = np.abs(a) < 1e-6
    if np.any(linear_mask):
        # Solve the simple linear relation ratio = b * u.
        norm_cmds[linear_mask] = ratio[linear_mask] / np.clip(b[linear_mask], 1e-6, None)

    # Handle the general quadratic case when linearity != 1.
    quad_mask = ~linear_mask
    if np.any(quad_mask):
        a_q = a[quad_mask]
        b_q = b[quad_mask]
        ratio_q = ratio[quad_mask]
        # Solve the quadratic equation using the positive root that maps to [0, 1].
        discriminant = np.maximum(b_q ** 2 + 4.0 * a_q * ratio_q, 0.0)
        norm_cmds[quad_mask] = (-b_q + np.sqrt(discriminant)) / (2.0 * a_q)

    # Enforce the actuator bounds after inversion.
    return np.clip(norm_cmds, 0.0, 1.0)


def _cbf_filter(
    u_ref_thrust: np.ndarray,
    *,
    solo_index: int,
    all_positions: np.ndarray,
    all_velocities: np.ndarray,
    all_accelerations: np.ndarray,
    env_state,
) -> np.ndarray:
    """
    Solve the exponential control barrier function (ECBF) quadratic program for the solo agent.

    Parameters
    ----------
    u_ref_thrust:
        4-vector of reference per-motor thrusts (Newtons) supplied by the nominal solo policy.
    solo_index:
        Index of the controlled agent within the vectorised environment.
    all_positions / all_velocities:
        Arrays of shape (N, 3) in world coordinates describing the swarm state at the current step.
    env_state:
        Handle to the underlying multi-agent environment (needed to access the simulator dynamics
        object for mass, gravity, thrust limits, and current attitude).

    Returns
    -------
    np.ndarray
        The thrust vector (Newtons) that minimises ‖u − u_ref‖² subject to the ECBF safety
            hyperplanes, actuator limits, and a shared slack variable.
    """
    # Cache the reference thrust as a double precision array for the optimisation.
    u_ref = np.asarray(u_ref_thrust, dtype=np.float64)
    # Pull out the simulator's per-agent dynamics object for physical parameters.
    solo_quad = env_state.envs[solo_index]
    dynamics = solo_quad.dynamics

    # Retrieve the current world-frame position and velocity of the solo quad.
    solo_pos = np.asarray(all_positions[solo_index], dtype=np.float64)
    solo_vel = np.asarray(all_velocities[solo_index], dtype=np.float64)

    # Extract mass and gravity to parameterise the translational dynamics.
    mass = float(dynamics.mass)
    gravity_vec = np.array([0.0, 0.0, -float(dynamics.gravity)], dtype=np.float64)
    # The third column of the rotation matrix gives the body thrust axis in world coordinates.
    thrust_axis_world = np.asarray(dynamics.rot, dtype=np.float64)[:, 2]

    # Accumulate one (a, b) pair per teammate for the ECBF inequality.
    constraints_data = []
    h_values = []
    for teammate_idx in range(NUM_MULTI_AGENTS):
        if teammate_idx == solo_index:
            continue

        # Skip inactive barriers (radius <= 0).
        radius = float(CBF_RADII[teammate_idx])
        if radius <= 0.0:
            continue

        # Compute the relative displacement and safety radius.
        teammate_pos = np.asarray(all_positions[teammate_idx], dtype=np.float64)
        z_vec = solo_pos - teammate_pos # Distance to teammate

        # Evaluate the barrier function h(z) = ||z||^2 - R^2.
        h_val = float(np.dot(z_vec, z_vec) - radius ** 2)
        h_values.append(h_val)
        # Pre-compute velocity magnitude used by the ECBF drift term.
        speed_sq = float(np.dot(solo_vel, solo_vel))
        # Dot the displacement with gravity to capture how thrust must counteract sag.
        z_dot_g = float(np.dot(z_vec, gravity_vec))
        # Dot with velocity for the first-order barrier derivative.
        z_dot_v = float(np.dot(z_vec, solo_vel))

        # Measure how aligned the thrust axis is with the line-of-sight vector.
        thrust_alignment = float(np.dot(z_vec, thrust_axis_world))
        # Scale factor c(x, R) in the ECBF inequality.
        c_scale = (2.0 / mass) * thrust_alignment
        # Each motor enters symmetrically, yielding `a^T u = c * sum(u_i)`.
        a_vec = c_scale * np.ones(4, dtype=np.float64)

        # Collect the drift and exponential terms into b(x, v).
        b_scalar = (
            2.0 * speed_sq
            + 2.0 * z_dot_g
            + 2.0 * CBF_K1 * z_dot_v
            + CBF_K0 * h_val
        )
        # Append the linear constraint coefficients for this teammate.
        constraints_data.append((a_vec, b_scalar))

    # If no constraints are active we can safely return the reference thrust.
    if not constraints_data:
        return u_ref

    # Lower bounds (grounded rotor thrust cannot be negative).
    u_min = np.zeros(4, dtype=np.float64)
    # Upper bounds match the simulator's maximum thrust per motor.
    u_max = np.asarray(dynamics.thrust_max, dtype=np.float64)
    # Optional weighting matrix shaping how deviations from u_ref are penalised.
    weight_matrix = np.diag(CBF_QP_DIAGONAL)

    # Decision variables: per-motor thrust vector and shared slack.
    u_var = cp.Variable(4)
    slack = cp.Variable()

    # Quadratic objective ‖u − u_ref‖²_W plus soft constraint penalty on slack.
    objective = cp.quad_form(u_var - u_ref, weight_matrix) + CBF_SLACK_WEIGHT * slack

    # Build the core box constraints and non-negativity of the slack.
    constraints = [
        u_var >= u_min,
        u_var <= u_max,
        slack >= 0,
    ]
    # Add one ECBF inequality per teammate, softened by the shared slack.
    for a_vec, b_scalar in constraints_data:
        constraints.append(a_vec @ u_var >= -b_scalar - slack)

    # Formulate the full optimisation problem for the chosen solver.
    problem = cp.Problem(cp.Minimize(objective), constraints)

    # Select and invoke the solver (defaults to OSQP) with warm-starts enabled.
    solver = getattr(cp, CBF_QP_SOLVER, cp.OSQP)
    try:
        problem.solve(solver=solver, warm_start=True, verbose=False)
    except cp.SolverError:
        # On solver failure, fall back to a box-clipped reference thrust.
        return np.clip(u_ref, u_min, u_max)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        # If the solver did not converge to a solution, use the safe fallback.
        return np.clip(u_ref, u_min, u_max)

    # Extract the optimal thrust vector, enforce actuator limits, and double-check constraints.
    solution = np.array(u_var.value, dtype=np.float64)
    solution = np.clip(solution, u_min, u_max)

    # slack_val = float(slack.value)
    # for teammate_idx, (a_vec, b_scalar) in enumerate(constraints_data):
    #     residual = float(np.dot(a_vec, solution) + b_scalar)
    #     if residual < -slack_val - 1e-4:
    #         print(
    #             "[cbf] Constraint violation detected "
    #             f"(teammate={teammate_idx}, residual={residual:.6f}, "
    #             f"slack={slack_val:.6f}, status={problem.status})"
    #         )

    return solution

def go_forward_for_half(
    base_action: np.ndarray,
    obs_last: np.ndarray,
    *,
    drone_index: int,
    step_index: int,
    total_steps: int,
    all_positions: np.ndarray,
    all_velocities: np.ndarray,
    policies: Dict[str, torch.nn.Module],
) -> np.ndarray:
    """
    Modify the 11th drone's action.  `base_action` and the return value are 4-element
    motor commands in [-1, 1].  `obs_last` is the full observation vector for drone_index.
    `all_positions`/`all_velocities` are (N, 3).  `policies` provides the live policy objects.
    """
    if total_steps <= 0:
        return base_action

    halfway = total_steps // 2
    if step_index <= halfway:
        return base_action

    forward_action = np.array([-1, -1, 1, 1], dtype=np.float32) # Overriding
    return forward_action

def custom_logic(
    base_action: np.ndarray,
    obs_last: np.ndarray,
    *,
    drone_index: int,
    step_index: int,
    total_steps: int,
    all_positions: np.ndarray,
    all_velocities: np.ndarray,
    policies: Dict[str, torch.nn.Module],
    env_state,
) -> np.ndarray:
    """
    Apply the motor-space ECBF quadratic program to keep the 11th quad within the safe set.

    Steps:
        1. Convert the policy's raw action ([-1, 1] per motor) into the physical thrust
           commands that drive the simulator's translational dynamics.
        2. Assemble one linear ECBF inequality per teammate using the current position,
           velocity, attitude, and the requested safety radius.
        3. Solve the QP to find the closest thrust vector that satisfies every inequality
           while respecting the actuator box constraints (and sharing one slack variable).
        4. Map that thrust back to the environment's action space for execution.
    """
    if total_steps <= 0 or env_state is None:
        return base_action

    # The observation, policy dictionary, and step index are accepted for extensibility, but
    # the current optimisation relies purely on the dynamics state pulled from the simulator.
    _ = (obs_last, policies, step_index)

    # Access the simulated dynamics so we can interpret motor actions in thrust units.
    solo_quad = env_state.envs[drone_index]
    dynamics = solo_quad.dynamics

    # Convert the policy action into the [0, 1] normalized space used inside the simulator.
    base_action = np.asarray(base_action, dtype=np.float64)
    norm_ref = np.clip(0.5 * (base_action + 1.0), 0.0, 1.0)
    # Map normalized actions to physical thrusts, matching the ECBF derivation.
    u_ref_thrust = _normalized_to_thrust(norm_ref, dynamics)

    safe_thrust = _cbf_filter(
        u_ref_thrust,
        solo_index=drone_index,
        all_positions=all_positions,
        all_velocities=all_velocities,
        all_accelerations=None,
        env_state=env_state,
    )

    # Convert the safe thrust profile back to normalized commands.
    safe_norm = _thrust_to_normalized(safe_thrust, dynamics)
    # Finally, remap to the environment's [-1, 1] action space and clip for robustness.
    safe_action = 2.0 * safe_norm - 1.0
    return np.clip(safe_action.astype(np.float32), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mixed-policy quadrotor evaluation")
    parser.add_argument("--multi_train_dir", required=True,
                        help="Train dir that holds the multi-agent policy (collision-aware)")
    parser.add_argument("--multi_experiment", required=True,
                        help="Experiment name for the multi-agent run")
    parser.add_argument("--solo_train_dir", required=True,
                        help="Train dir that holds the single-agent policy")
    parser.add_argument("--solo_experiment", required=True,
                        help="Experiment name for the single-agent run")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Simulation steps to run before exiting")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic (mean) actions instead of sampling")
    parser.add_argument("--save_video", action="store_true",
                        help="Capture frames and save an MP4 once evaluation finishes")
    parser.add_argument("--cbf", action="store_true",
                        help="Whether to activate the CBF or not")
    parser.add_argument("--video_name", default="mixed_enjoy_replay.mp4",
                        help="Filename (or path) for the saved video when --save_video is set")
    parser.add_argument("--video_fps", type=int, default=30,
                        help="Target FPS for the saved video (only used with --save_video)")
    args = parser.parse_args()

    if args.save_video:
        if os.path.isabs(args.video_name):
            video_output_dir = os.path.dirname(args.video_name) or "."
            video_filename = os.path.basename(args.video_name)
        else:
            video_output_dir = os.path.join(args.multi_train_dir, args.multi_experiment)
            video_filename = args.video_name
        os.makedirs(video_output_dir, exist_ok=True)
        video_frames = []
    else:
        video_output_dir = None
        video_filename = None
        video_frames = []

    torch.set_grad_enabled(False)
    register_swarm_components()

    # Build the evaluation environment config (11 agents, render on).
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
    render_mode = "rgb_array" if args.save_video else "human"
    env = make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=render_mode)

    # Load policies.
    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    multi_ckpt = latest_checkpoint(args.multi_train_dir, args.multi_experiment, policy_index=0)
    multi_actor = load_actor(cfg_multi, env.observation_space, env.action_space, multi_ckpt, device)

    cfg_solo = load_cfg(args.solo_train_dir, args.solo_experiment)
    solo_env = make_quadrotor_env("quadrotor_multi", cfg=cfg_solo, render_mode=None)
    solo_ckpt = latest_checkpoint(args.solo_train_dir, args.solo_experiment, policy_index=0)
    solo_actor = load_actor(cfg_solo, solo_env.observation_space, solo_env.action_space, solo_ckpt, device)
    solo_obs_dim = solo_env.observation_space.shape[0]
    solo_env.close()

    multi_rnn_states = torch.zeros((NUM_MULTI_AGENTS, get_rnn_size(cfg_multi)), dtype=torch.float32, device=device)
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=device)

    obs, _ = env.reset()
    dones = np.zeros(env.num_agents, dtype=bool)
    solo_collision_count = 0

    if args.save_video:
        initial_frame = env.render()
        if initial_frame is not None:
            video_frames.append(initial_frame.copy())

    for step in tqdm(range(args.max_steps)):
        obs_np = np.asarray(obs)

        # First 10 drones use the shared multi-agent policy.
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

        # 11th drone (index 10) uses the solo policy + custom logic.
        obs_last_self = obs_np[NUM_MULTI_AGENTS, :solo_obs_dim]
        obs_last_dict = {OBS_KEY: obs_last_self[None, :]}
        with torch.no_grad():
            normalized_last = prepare_and_normalize_obs(solo_actor, obs_last_dict)
            policy_last = solo_actor(normalized_last, solo_rnn_states)
        action_last = policy_last["actions"]
        solo_rnn_states = policy_last["new_rnn_states"]
        if args.deterministic:
            action_last = argmax_actions(solo_actor.action_distribution())
        if action_last.dim() == 1:
            action_last = action_last.unsqueeze(0)
        action_last = action_last.detach().cpu().numpy()[0]

        positions, velocities = get_drone_states(env.unwrapped)
        policies = {"multi": multi_actor, "solo": solo_actor}
        if args.cbf:
            action_last = custom_logic(
                base_action=action_last,
                obs_last=obs_np[NUM_MULTI_AGENTS],
                drone_index=NUM_MULTI_AGENTS,
                step_index=step,
                total_steps=args.max_steps,
                all_positions=positions,
                all_velocities=velocities,
                policies=policies,
                env_state=env.unwrapped,
            )

        actions = np.vstack([actions_multi, action_last[None, :]])
        obs, rewards, terminated, truncated, infos = env.step(actions)
        solo_info_rewards = infos[SOLO_AGENT_INDEX].get("rewards", {})
        if solo_info_rewards.get("rewraw_quadcol", 0.0) < 0.0:
            solo_collision_count += 1
            print('Crash', solo_collision_count)
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        dones = np.logical_or(terminated, truncated)
        if args.save_video:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
        else:
            env.render()

        if np.all(dones):
            obs, _ = env.reset()
            dones = np.zeros(env.num_agents, dtype=bool)
            if args.save_video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame.copy())

    env.close()

    if args.save_video and video_frames:
        video_cfg = AttrDict(video_name=video_filename)
        generate_replay_video(video_output_dir, video_frames, args.video_fps, video_cfg)
        final_path = os.path.abspath(os.path.join(video_output_dir, video_filename))
        print(f"[mixed_enjoy] Video saved to {final_path}")

    print(f"[mixed_enjoy] Solo drone collisions: {solo_collision_count}")


if __name__ == "__main__":
    main()
