import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rollout_seed_sweeper import (
    _build_agent_rows,
    _causal_metrics,
    _initial_geometry,
    _paired_metrics,
    _pair_metrics,
    _proximity,
    _recording_filter,
    _radius_is_interesting,
    _radius_rank_key,
    _rank_summaries,
    parse_args,
)


class SeedSweeperMetricTests(unittest.TestCase):
    def test_proximity_is_time_aligned_and_per_agent(self):
        ego = np.zeros((3, 3))
        others = np.array(
            [
                [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[1.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
            ]
        )

        result = _proximity(ego, others, threshold=0.75)

        self.assertEqual(result["min_distance"], 0.25)
        self.assertEqual(result["closest_agent"], 1)
        self.assertEqual(result["closest_step"], 2)
        self.assertAlmostEqual(result["close_fraction"], 2.0 / 3.0)
        self.assertEqual(result["unique_close_agents"], 2)
        np.testing.assert_allclose(result["per_agent_min_distance"], [0.5, 0.25])

    def test_initial_geometry_ignores_non_ego_pair_distance(self):
        def state(position, goal=(0.0, 0.0, 0.0)):
            return SimpleNamespace(
                position=np.asarray(position, dtype=np.float64),
                goal=np.asarray(goal, dtype=np.float64),
            )

        first = [
            state((1.0, 0.0, 0.0)),
            state((-1.0, 0.0, 0.0)),
            state((0.0, 0.0, 0.0), goal=(3.0, 0.0, 0.0)),
        ]
        second = [
            state((1.0, 0.0, 0.0)),
            state((0.0, 1.0, 0.0)),
            state((0.0, 0.0, 0.0), goal=(3.0, 0.0, 0.0)),
        ]

        first_result = _initial_geometry(first, ego_index=2)
        second_result = _initial_geometry(second, ego_index=2)

        self.assertEqual(first_result["ego_initial_min_distance"], 1.0)
        self.assertEqual(second_result["ego_initial_min_distance"], 1.0)
        np.testing.assert_array_equal(
            first_result["ego_initial_distances"],
            second_result["ego_initial_distances"],
        )
        self.assertEqual(first_result["ego_goal_distance"], 3.0)

    def test_paired_metrics_keep_other_response_descriptive(self):
        baseline_positions = np.array(
            [
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        filtered_positions = baseline_positions.copy()
        filtered_positions[:, 2, 1] = [0.1, 0.2, 0.3]
        filtered_positions[1, 0, 0] += 0.4
        filtered_positions[2, 1, 0] -= 0.2
        baseline = {"trajectory_length": 3, "positions": baseline_positions}
        filtered = {"trajectory_length": 3, "positions": filtered_positions}
        predictions = np.zeros((2, 3, 6))
        predictions[:, :, :3] = np.transpose(
            baseline_positions[:, :2], (1, 0, 2)
        )

        result = _paired_metrics(
            baseline,
            filtered,
            predictions,
            [0.0, 0.02, 0.04],
            num_multi_agents=2,
            proximity_threshold=0.75,
            action_delta_threshold=0.01,
        )

        self.assertAlmostEqual(result["baseline_min_distance"], 0.4)
        self.assertAlmostEqual(result["action_delta_mean"], 0.02)
        self.assertAlmostEqual(result["action_delta_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(result["ego_response_max"], 0.3)
        self.assertAlmostEqual(result["other_response_max"], 0.4)
        self.assertEqual(result["other_response_max_step"], 1)
        self.assertEqual(result["other_response_max_agent"], 0)
        self.assertAlmostEqual(result["other_response_mean"], 0.1)
        self.assertAlmostEqual(result["other_response_final_max"], 0.2)

    def test_paired_metrics_require_one_delta_per_filtered_step(self):
        trajectory = {"trajectory_length": 2, "positions": np.zeros((2, 2, 3))}
        with self.assertRaisesRegex(ValueError, "one action delta"):
            _paired_metrics(
                trajectory,
                trajectory,
                np.zeros((1, 2, 6)),
                [0.0],
                num_multi_agents=1,
                proximity_threshold=1.0,
                action_delta_threshold=0.01,
            )


class SeedSweeperCausalTests(unittest.TestCase):
    @staticmethod
    def _turning_trajectory(num_steps=100):
        points = np.zeros((num_steps + 1, 3), dtype=np.float64)
        for step in range(num_steps + 1):
            if step <= 40:
                points[step] = [0.1 * step, 0.0, 0.0]
            else:
                points[step] = [4.0, 0.1 * (step - 40), 0.0]
        positions = np.zeros((num_steps, 3, 3), dtype=np.float64)
        positions[:, 2] = points[1:]
        initial_positions = np.zeros((3, 3), dtype=np.float64)
        initial_positions[2] = points[0]
        return {
            "trajectory_length": num_steps,
            "initial_positions": initial_positions,
            "positions": positions,
        }

    @staticmethod
    def _event_args():
        return SimpleNamespace(
            num_multi_agents=2,
            min_agent_action_delta=0.01,
            max_active_gap=2,
            min_active_frames=5,
            heading_window=5,
            min_heading_displacement=0.1,
            min_event_start=5,
            event_end_margin=10,
            event_proximity_margin=0.5,
            min_visible_turn_deg=20.0,
        )

    def test_agent_event_requires_ego_turn_and_ego_agent_proximity(self):
        trajectory = self._turning_trajectory()
        agent_action_delta = np.zeros((100, 2), dtype=np.float64)
        agent_action_delta[30:41, 0] = 0.05
        agent_action_delta[30:41, 1] = 0.05
        records = {
            "agent_action_delta": list(agent_action_delta),
            "actual_distance": list(np.full((100, 2), [0.8, 3.0])),
            "predicted_distance": list(np.full((100, 2), [0.7, 2.5])),
        }

        rows = _build_agent_rows(
            7,
            0.5,
            trajectory,
            records,
            np.array([2.0, 2.0]),
            self._event_args(),
        )

        self.assertEqual(rows[0]["direct_agent"], 1)
        self.assertEqual(rows[0]["visible_agent"], 1)
        self.assertGreater(rows[0]["visible_turn_deg"], 80.0)
        self.assertEqual(rows[1]["direct_agent"], 1)
        self.assertEqual(rows[1]["visible_agent"], 0)

    def test_causal_metrics_require_leaveout_heading_and_path_change(self):
        filtered = self._turning_trajectory()
        leaveout = self._turning_trajectory()
        straight = np.zeros((100, 3, 3), dtype=np.float64)
        straight[:, 2, 0] = np.arange(1, 101) * 0.1
        leaveout["positions"] = straight
        args = SimpleNamespace(
            num_multi_agents=2,
            causal_post_frames=30,
            causal_heading_half_window=3,
            min_heading_displacement=0.1,
            min_causal_heading_deg=10.0,
            min_causal_heading_frames=3,
            min_causal_deviation=0.15,
            episode_length=100,
        )
        row = {"direct_run_start": 30, "direct_run_end": 40}

        result = _causal_metrics(filtered, leaveout, row, args)

        self.assertEqual(result["leaveout_completed"], 1)
        self.assertEqual(result["causal_agent"], 1)
        self.assertGreater(result["causal_heading_max_deg"], 80.0)
        self.assertGreater(result["causal_deviation_max"], 0.15)

    def test_pair_metrics_distinguish_sequential_and_overlapping_events(self):
        sequential = [
            {"causal_agent": 1, "direct_run_start": 30, "direct_run_end": 40},
            {"causal_agent": 1, "direct_run_start": 70, "direct_run_end": 80},
        ]
        overlapping = [
            {"causal_agent": 1, "direct_run_start": 30, "direct_run_end": 50},
            {"causal_agent": 1, "direct_run_start": 45, "direct_run_end": 60},
        ]

        sequential_result = _pair_metrics(sequential, min_event_gap=15)
        overlapping_result = _pair_metrics(overlapping, min_event_gap=15)

        self.assertEqual(sequential_result["has_sequential_pair"], 1)
        self.assertEqual(sequential_result["has_overlapping_pair"], 0)
        self.assertEqual(overlapping_result["has_sequential_pair"], 0)
        self.assertEqual(overlapping_result["has_overlapping_pair"], 1)

    def test_radius_qualification_requires_two_causal_ego_agents(self):
        one = [{"causal_agent": 1}, {"causal_agent": 0}]
        two = [{"causal_agent": 1}, {"causal_agent": 1}]

        self.assertEqual(_radius_is_interesting(1, one), 0)
        self.assertEqual(_radius_is_interesting(1, two), 1)
        self.assertEqual(_radius_is_interesting(0, two), 0)


class SeedSweeperFilterTests(unittest.TestCase):
    @staticmethod
    def _filter_state():
        env_state = SimpleNamespace(
            envs=[
                SimpleNamespace(
                    dynamics=SimpleNamespace(pos=np.array([1.0, 0.0, 0.0]))
                ),
                SimpleNamespace(
                    dynamics=SimpleNamespace(pos=np.array([0.0, 2.0, 0.0]))
                ),
                SimpleNamespace(dynamics=SimpleNamespace(pos=np.zeros(3))),
            ]
        )
        swarm_state = SimpleNamespace(
            positions=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        )
        return env_state, swarm_state

    def test_recording_filter_measures_each_ego_neighbor_action_effect(self):
        weights = np.array([0.1, 0.2], dtype=np.float32)
        constructed_radii = []

        def fake_make_cbf_filter(radii):
            radii = np.asarray(radii, dtype=np.float64).copy()
            constructed_radii.append(radii)

            def fake_filter(base_action, env_state, swarm_state):
                del env_state, swarm_state
                result = np.asarray(base_action, dtype=np.float32).copy()
                result[0] += np.sum(weights[radii >= 0.0])
                return result

            return fake_filter

        env_state, swarm_state = self._filter_state()

        with patch(
            "rollout_seed_sweeper.make_cbf_filter",
            side_effect=fake_make_cbf_filter,
        ):
            action_fn, records = _recording_filter(0.5, 2)
            result = action_fn(np.zeros(4, dtype=np.float32), env_state, swarm_state)

        np.testing.assert_allclose(result, [0.3, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(records["action_delta"], [0.3])
        np.testing.assert_allclose(records["agent_action_delta"], [[0.1, 0.2]])
        np.testing.assert_allclose(records["actual_distance"], [[1.0, 2.0]])
        np.testing.assert_allclose(records["predicted_distance"], [[1.0, 2.0]])
        np.testing.assert_allclose(constructed_radii[0], [0.5, 0.5])
        np.testing.assert_allclose(constructed_radii[1], [-1.0, 0.5])
        np.testing.assert_allclose(constructed_radii[2], [0.5, -1.0])

    def test_recording_filter_does_not_hide_canceling_neighbor_effects(self):
        def fake_make_cbf_filter(radii):
            radii = np.asarray(radii, dtype=np.float64)

            def fake_filter(base_action, env_state, swarm_state):
                del env_state, swarm_state
                result = np.asarray(base_action, dtype=np.float32).copy()
                if radii[0] < 0.0:
                    result[0] = 0.2
                elif radii[1] < 0.0:
                    result[0] = -0.2
                return result

            return fake_filter

        env_state, swarm_state = self._filter_state()
        with patch(
            "rollout_seed_sweeper.make_cbf_filter",
            side_effect=fake_make_cbf_filter,
        ):
            action_fn, records = _recording_filter(0.5, 2)
            result = action_fn(np.zeros(4, dtype=np.float32), env_state, swarm_state)

        np.testing.assert_allclose(result, np.zeros(4))
        np.testing.assert_allclose(records["action_delta"], [0.0])
        np.testing.assert_allclose(records["agent_action_delta"], [[0.2, 0.2]])


class SeedSweeperRankingTests(unittest.TestCase):
    @staticmethod
    def _candidate(seed, *, goal_distance, other_response):
        return {
            "scenario_seed": seed,
            "interesting": 1,
            "qualifying_radius_count": 2,
            "has_sequential_pair": 1,
            "num_visible_agents": 2,
            "num_direct_agents": 2,
            "second_causal_heading_deg": 25.0,
            "second_causal_deviation": 0.3,
            "radius_span_position_max": 0.4,
            "ego_goal_distance": goal_distance,
            "other_response_max": other_response,
        }

    def test_goal_and_other_response_do_not_affect_ranking(self):
        first = self._candidate(1, goal_distance=0.01, other_response=1000.0)
        second = self._candidate(2, goal_distance=1000.0, other_response=0.0)

        ranked = _rank_summaries([second, first])

        self.assertEqual([row["scenario_seed"] for row in ranked], [1, 2])

    def test_goal_and_other_response_do_not_select_best_radius(self):
        common = {
            "interesting": 0,
            "has_sequential_pair": 0,
            "num_causal_agents": 1,
            "second_causal_heading_deg": 0.0,
            "second_causal_deviation": 0.0,
            "num_visible_agents": 1,
            "num_direct_agents": 1,
            "filtered_min_distance": 0.8,
        }
        first = {**common, "ego_goal_distance": 0.01, "other_response_max": 1000.0}
        second = {**common, "ego_goal_distance": 1000.0, "other_response_max": 0.0}

        self.assertEqual(_radius_rank_key(first), _radius_rank_key(second))


class SeedSweeperCliTests(unittest.TestCase):
    def test_defaults_target_low_radius_paired_rollouts(self):
        args = parse_args([])
        self.assertEqual((args.seed_start, args.seed_stop), (0, 100))
        self.assertEqual(args.rollout_seed, 0)
        self.assertEqual(args.episode_length, 300)
        self.assertEqual(args.num_multi_agents, 5)
        self.assertEqual(args.radii, [0.4, 0.5, 0.6])
        self.assertEqual(args.min_agent_action_delta, 0.01)

    def test_rejects_nonpositive_radius(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--radii", "0"])

    def test_event_context_must_cover_heading_window(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--min_event_start", "5", "--heading_window", "10"])


if __name__ == "__main__":
    unittest.main()
