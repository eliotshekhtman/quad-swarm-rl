import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from swarm_rl.env_snapshot import (
    clone_env_from_snapshot,
    restore_rng_state,
    safe_capture_env_snapshot,
    snapshot_rng_state,
)

from project_utils.utils import OBS_KEY, get_swarm_state
from project_utils.restart_utils import extract_positions_velocities

def get_alpha_bar(alpha, delta, num_trajectories):
    return alpha - np.sqrt(np.log(1 / delta) / (2 * num_trajectories))

def conformal_radii(logs, num_multi_agents, pred_trajectories, alpha, episode_length):
    radii = np.full(num_multi_agents, 0, dtype=np.float64) # Probs set to arm len
    # Need a radius for each agent
    for agent_id in range(num_multi_agents):
        predictions = pred_trajectories[agent_id]
        # Collect trajectory-level nonconformity scores
        scores = []
        for run_log in logs[agent_id]:
            score = 0  # Lowerbound on possible nonconformity score
            run = np.concatenate(
                [run_log["position"], run_log["velocity"]],
                axis=-1,
            ).astype(np.float32)
            for i in range(episode_length):
                # Largest distance across timesteps in the episode
                score = max(score, np.linalg.norm(predictions[i][:3] - run[i][:3]))
            scores.append(score)
        scores.sort()
        # Just want to visually check that this makes sense
        # print(f'Scores for agent {agent_id}: ', scores)
        conformal_radius = scores[int(np.ceil(len(scores + 1) * (1 - alpha)) - 1)]
        radii[agent_id] = conformal_radius
    return radii

def joint_conformal_radii(logs, num_multi_agents, pred_trajectories, alpha, episode_length, num_trajectories):
    # pred_trajectories: num_agents x episode_length x 6
    pred_positions = []
    for agent_id in range(num_multi_agents):
        pred_traj = np.stack(pred_trajectories[agent_id], axis=0)
        pred_positions.append(pred_traj[:,:3])
    predictions = np.concatenate(pred_positions, axis=1) # epsiode_length x (num_multi_agents * 3)
    scores = []
    for i in range(num_trajectories):
        # logs: num_agents x num_trajectories x [] x episode_length x 3
        run = np.concatenate([logs[agent_id][i]["position"] for agent_id in range(num_multi_agents)], axis=-1)
        score = 0
        for t in range(episode_length):
            score = max(score, np.linalg.norm(predictions[t] - run[t]))
        scores.append(score)
    scores.sort()
    conformal_radius = scores[int(np.ceil(len(scores) * (1 - alpha)) - 1)]
    return conformal_radius

def explicit_radius_update(prev_radius, conf_radius, kappa, MIN_RADIUS, MAX_RADIUS):
    if conf_radius <= prev_radius:
        radius = (conf_radius + kappa * prev_radius) / (1 + kappa)
    else:
        radius = (conf_radius - kappa * prev_radius) / (1 - kappa)
    return np.clip(radius, MIN_RADIUS, MAX_RADIUS)






def fall_down(base_action, env_state, swarm_state):
    return np.array([-1., -1., -1., -1.], dtype=np.float64)

def run_multi_agents(env, obs, num_multi_agents, 
                    multi_actor, multi_rnn_states, 
                    solo_actor, solo_rnn_states, solo_obs_dim, 
                    pred_trajectories, solo_action_fn,
                    max_steps=1500, num_runs=1, deterministic=False,
                    num_threads=1):
    '''
    Run the environment for [max_steps] steps, where the multi agents act like normal
    but the solo agent plays a fixed action, and return positions and velocities.
    Does not log the initial state.
    '''
    snapshot = safe_capture_env_snapshot(env)
    # logs: num_multi_agents + 1 x num_runs x [pos or vel] x max_steps x 3
    logs = {i: [None] * num_runs for i in range(num_multi_agents + 1)}
    max_workers = max(1, min(num_threads, num_runs))
    
    def _run_single_episode(run_idx: int):
        run_logs = {agent_id: {"position": [], "velocity": [], "goal_dist": [], "goal_swap": []} for agent_id in range(num_multi_agents + 1)}
        env_run = clone_env_from_snapshot(snapshot)
        obs_run = np.array(obs, copy=True, dtype=np.float32)
        done = False
        step_num = 0
        run_multi_rnn_states = multi_rnn_states.clone()
        run_solo_rnn_states = solo_rnn_states.clone()
        run_max_dist = 0.0

        scenario = env_run.unwrapped.scenario
        goals = []
        for agent_id in range(num_multi_agents + 1):
            active = scenario.active_goal_index[agent_id]
            target = scenario.goal_pairs[agent_id, active]
            goals.append(target)
        # print("Active starting goals:", goals)
        # print(env_run.envs[0].sense_noise.bypass, env_run.envs[0].dynamics.thrust_noise_ratio)
        # print(env_run.envs[-1].sense_noise.bypass, env_run.envs[-1].dynamics.thrust_noise_ratio)

        while not done and step_num < max_steps:
            obs_multi_dict = {OBS_KEY: obs_run[:num_multi_agents]}
            with torch.no_grad():
                normalized_obs = prepare_and_normalize_obs(multi_actor, obs_multi_dict)
                policy_output = multi_actor(normalized_obs, run_multi_rnn_states)
            actions_multi = policy_output["actions"]
            run_multi_rnn_states = policy_output["new_rnn_states"]
            if deterministic or run_idx == 0: # First run always deterministic
                actions_multi = argmax_actions(multi_actor.action_distribution())
            if actions_multi.dim() == 1:
                actions_multi = actions_multi.unsqueeze(-1)
            actions_multi = actions_multi.detach().cpu().numpy()

            obs_solo_self = obs_run[-1, :solo_obs_dim]
            obs_solo_dict = {OBS_KEY: obs_solo_self[None, :]}
            with torch.no_grad():
                normalized_solo = prepare_and_normalize_obs(solo_actor, obs_solo_dict)
                policy_solo = solo_actor(normalized_solo, run_solo_rnn_states)
            run_solo_rnn_states = policy_solo["new_rnn_states"]
            # Running the ego agent as deterministic, as would happen in practice
            action_solo = argmax_actions(solo_actor.action_distribution())
            if action_solo.dim() == 1:
                action_solo = action_solo.unsqueeze(0)
            action_solo = action_solo.detach().cpu().numpy()[0]
            swarm_state = get_swarm_state(env_run.unwrapped)
            # Conformal radius uses predicted next timestep states
            for agent_id in range(num_multi_agents):
                swarm_state.positions[agent_id, :] = pred_trajectories[agent_id][step_num][:3]
                swarm_state.velocities[agent_id, :] = pred_trajectories[agent_id][step_num][3:]
            # Apply CBF to the ego agent based on radius determined earlier
            action_solo = solo_action_fn(
                base_action=action_solo,
                env_state=env_run.unwrapped,
                swarm_state=swarm_state
            )

            actions = np.vstack([actions_multi, action_solo[None, :]])
            obs_run, rewards, dones, infos = env_run.step(actions)
            obs_run = np.array(obs_run, dtype=np.float32)

            pos, vel = extract_positions_velocities(env_run.unwrapped)
            for agent_id in range(num_multi_agents + 1):
                active = scenario.active_goal_index[agent_id]
                target = scenario.goal_pairs[agent_id, active]
                dist = np.linalg.norm(pos[agent_id] - target)
                run_logs[agent_id]["position"].append(pos[agent_id])
                run_logs[agent_id]["velocity"].append(vel[agent_id])
                run_logs[agent_id]["goal_dist"].append(dist)
                if not np.allclose(target, goals[agent_id]):
                    run_logs[agent_id]["goal_swap"].append(True)
                    goals[agent_id] = target 
                else:
                    run_logs[agent_id]["goal_swap"].append(False)
                if agent_id < num_multi_agents: # Don't care how far solo quad deviates
                    run_max_dist = max(run_max_dist, np.linalg.norm(pos[agent_id] - swarm_state.positions[agent_id, :]))
            done = np.all(dones)
            step_num += 1
        env_run.close()
        return run_idx, run_logs, run_max_dist

    max_dist = 0.0
    progress_bar = tqdm(total=num_runs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_episode, run_idx) for run_idx in range(num_runs)]
        for future in as_completed(futures):
            run_idx, run_logs, run_max_dist = future.result()
            for agent_id in range(num_multi_agents + 1):
                logs[agent_id][run_idx] = run_logs[agent_id]
            max_dist = max(max_dist, run_max_dist)
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"max dist={max_dist:.3f}")
    progress_bar.close()

    return logs
