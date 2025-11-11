import numpy as np

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
        conformal_radius = scores[int(np.ceil(len(scores) * (1 - alpha)) - 1)]
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

def explicit_radius_update(prev_radius, conf_radius, kappa):
    if conf_radius <= prev_radius:
        radius = (conf_radius + kappa * prev_radius) / (1 + kappa)
    else:
        radius = (conf_radius - kappa * prev_radius) / (1 - kappa)
    return np.clip(radius, MIN_RADIUS, MAX_RADIUS)