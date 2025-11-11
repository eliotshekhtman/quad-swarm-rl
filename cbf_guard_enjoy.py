from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.huggingface.huggingface_utils import generate_replay_video
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

from project_utils.utils import *
from project_utils.cbf_utils import make_cbf_filter


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
    parser.add_argument("--num_multi_agents", type=int, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
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
        f"--quads_num_agents={args.num_multi_agents + 1}",
        f"--quads_neighbor_visible_num={args.num_neighbors}",
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
        (args.num_multi_agents, get_rnn_size(cfg_multi)), dtype=torch.float32, device=device
    )
    solo_rnn_states = torch.zeros((1, get_rnn_size(cfg_solo)), dtype=torch.float32, device=device)

    obs, _ = env.reset()
    dones = np.zeros(env.num_agents, dtype=bool)
    solo_collision_count = 0

    # First frame
    frame = env.render()
    scene = env.unwrapped.scenes[0] # first entry matches the first quads_view_mode
    scene.camera_drone_index = -1

    radii = np.full(args.num_multi_agents, 1, dtype=np.float64)
    filter = make_cbf_filter(radii)

    progress_bar = tqdm(range(args.max_steps))
    for step in progress_bar:
        obs_np = np.asarray(obs)

        obs_multi_dict = {OBS_KEY: obs_np[:args.num_multi_agents]}
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

        obs_solo_self = obs_np[-1, :solo_obs_dim]
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
            action_solo = filter(
                base_action=action_solo,
                env_state=env.unwrapped,
                swarm_state=swarm_state,
            )
            closest_dist = 100 # arbitrarily big
            solo_pos = swarm_state.positions[-1]
            for teammate_pos in swarm_state.positions[:-1]:
                dist = np.linalg.norm(solo_pos - teammate_pos)
                closest_dist = min(dist, closest_dist)
            progress_bar.set_postfix_str(f"dist>={closest_dist}")

        actions = np.vstack([actions_multi, action_solo[None, :]])
        obs, rewards, terminated, truncated, infos = env.step(actions)

        solo_info_rewards = infos[-1].get("rewards", {})
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
