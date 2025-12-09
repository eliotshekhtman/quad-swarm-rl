#!/usr/bin/env python3
"""
Scan a range of seeds and report the minimum initial distance between the
non-ego agents and the ego agent at timestep 0.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from project_utils.utils import load_cfg
from project_utils.restart_utils import deterministic_reset, extract_positions_velocities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find seeds with well-spaced initial spawns.")
    parser.add_argument("--multi_train_dir", default="train_dir", help="Directory containing the trained multi-agent policy config.")
    parser.add_argument("--multi_experiment", required=True, help="Experiment name for the multi-agent policy.")
    parser.add_argument("--num_multi_agents", type=int, default=-1, help="Number of non-ego agents; defaults to config value.")
    parser.add_argument("--seed_start", type=int, default=0, help="First seed (inclusive).")
    parser.add_argument("--seed_end", type=int, default=99, help="Last seed (inclusive).")
    parser.add_argument("--seed_step", type=int, default=1, help="Seed step size (positive).")
    return parser.parse_args()


def build_env(num_multi_agents: int, cfg_multi) -> object:
    register_swarm_components()
    quads_mode = getattr(cfg_multi, "quads_mode", "patrol_dual_goal")
    eval_cli = [
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--device=cpu",
        f"--quads_mode={quads_mode}",
        f"--quads_num_agents={num_multi_agents + 1}",
        f"--quads_neighbor_visible_num={cfg_multi.quads_neighbor_visible_num}",
        f"--quads_neighbor_obs_type={cfg_multi.quads_neighbor_obs_type}",
        "--quads_use_numba=False",
        "--max_num_episodes=1",
        "--quads_collision_reward=8.0",
        "--quads_collision_hitbox_radius=2.5",
        "--quads_collision_falloff_radius=5.0",
        "--quads_collision_smooth_max_penalty=12.0",
    ]
    eval_cfg = parse_swarm_cfg(eval_cli, evaluation=True)
    return make_quadrotor_env("quadrotor_multi", cfg=eval_cfg, render_mode=None)


def min_distance_for_seed(env, seed: int, num_multi_agents: int) -> float:
    deterministic_reset(env, seed, None)
    positions, _ = extract_positions_velocities(env.unwrapped)
    ego_idx = num_multi_agents
    ego_pos = positions[ego_idx]
    other_positions = positions[:ego_idx]
    if other_positions.size == 0:
        return float("inf")
    distances = np.linalg.norm(other_positions - ego_pos, axis=1)
    return float(np.min(distances))


def main() -> None:
    args = parse_args()
    if args.seed_step <= 0:
        sys.exit("seed_step must be positive")

    cfg_multi = load_cfg(args.multi_train_dir, args.multi_experiment)
    num_multi_agents = args.num_multi_agents if args.num_multi_agents > 0 else cfg_multi.quads_num_agents

    env = build_env(num_multi_agents, cfg_multi)

    for i in range(2):
        dyn = env.unwrapped.envs[i].dynamics
        print('mass:', dyn.mass)
        print('intertia:', dyn.inertia)
        print('prop_pos:', dyn.prop_pos)
        print('prop_crossproducts:', dyn.prop_crossproducts)
        print('prop_ccw_mx:', dyn.prop_ccw_mx)
        print('thrust_max:', dyn.thrust_max)
        print('torque_max:', dyn.torque_max)

    results: List[Tuple[int, float]] = []
    for seed in range(args.seed_start, args.seed_end + 1, args.seed_step):
        min_dist = min_distance_for_seed(env, seed, num_multi_agents)
        results.append((seed, min_dist))

    env.close()

    print(f"Scanned seeds {args.seed_start} to {args.seed_end} (step {args.seed_step}).")
    print("seed,min_distance")
    for seed, dist in results:
        print(f"{seed},{dist:.6f}")

    best_seed, best_dist = max(results, key=lambda item: item[1])
    print(f"Largest min-distance seed: {best_seed} (min distance {best_dist:.6f})")


if __name__ == "__main__":
    main()
