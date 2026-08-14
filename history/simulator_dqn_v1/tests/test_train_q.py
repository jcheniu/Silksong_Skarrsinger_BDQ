"""Tests for the Double DQN training components."""

import random
from pathlib import Path
import unittest

import torch

from hk_rl_DQN.boss_env import BossDodgeEnv
from hk_rl_DQN.train_dqn import (
    ACTION_REPEAT,
    DQN,
    EPSILON_END,
    EPSILON_START,
    FRAME_GAMMA,
    STATE_DIMENSIONS,
    ReplayBuffer,
    Transition,
    action_mask,
    available_action_indices,
    build_network_from_checkpoint,
    encode_state,
    epsilon_for_step,
    load_training_checkpoint,
    optimize_model,
    select_greedy_action,
    step_with_action_repeat,
)


class StateEncodingTests(unittest.TestCase):
    def test_observation_is_normalized_to_fixed_width(self) -> None:
        env = BossDodgeEnv(seed=7)
        observation, _ = env.reset(seed=7)
        state = encode_state(observation)

        self.assertEqual(len(state), STATE_DIMENSIONS)
        self.assertAlmostEqual(state[0], env.player_x / env.ARENA_WIDTH)
        self.assertAlmostEqual(state[8], env.boss_x / env.ARENA_WIDTH)

    def test_wrong_observation_width_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_state((1.0, 2.0))


class NetworkTests(unittest.TestCase):
    def test_network_outputs_one_value_per_action(self) -> None:
        network = DQN()
        values = network(torch.zeros(4, STATE_DIMENSIONS))
        self.assertEqual(values.shape, (4, len(BossDodgeEnv.ACTIONS)))

    def test_optimizer_step_changes_online_weights(self) -> None:
        torch.manual_seed(7)
        online = DQN()
        target = DQN()
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-2)
        transition = Transition(
            state=(0.0,) * STATE_DIMENSIONS,
            action=1,
            reward=2.0,
            next_state=(0.1,) * STATE_DIMENSIONS,
            done=False,
            next_action_mask=(True,) * len(BossDodgeEnv.ACTIONS),
        )
        before = [parameter.detach().clone() for parameter in online.parameters()]

        loss = optimize_model(online, target, optimizer, [transition] * 4, torch.device("cpu"))

        self.assertGreater(loss, 0.0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, online.parameters())))

    def test_epsilon_schedule_reaches_configured_floor(self) -> None:
        self.assertEqual(epsilon_for_step(0), EPSILON_START)
        self.assertEqual(epsilon_for_step(10**9), EPSILON_END)


class ReplayBufferTests(unittest.TestCase):
    @staticmethod
    def transition(action: int) -> Transition:
        return Transition(
            state=(0.0,) * STATE_DIMENSIONS,
            action=action,
            reward=0.0,
            next_state=(0.0,) * STATE_DIMENSIONS,
            done=False,
            next_action_mask=(True,) * len(BossDodgeEnv.ACTIONS),
        )

    def test_capacity_evicts_oldest_transition(self) -> None:
        replay = ReplayBuffer(capacity=2)
        replay.append(self.transition(0))
        replay.append(self.transition(1))
        replay.append(self.transition(2))
        sampled = replay.sample(2, random.Random(7))

        self.assertEqual(len(replay), 2)
        self.assertEqual({item.action for item in sampled}, {1, 2})


class ActionSelectionTests(unittest.TestCase):
    def test_greedy_selection_respects_action_mask(self) -> None:
        selected = select_greedy_action([100.0, 3.0, 1.0], random.Random(7), (1, 2))
        self.assertEqual(selected, 1)

    def test_recovery_actions_are_masked(self) -> None:
        env = BossDodgeEnv(seed=7)
        env.player_attack_recovery_timer = 10
        env.player_dash_recovery_timer = 10
        available = available_action_indices(env)
        mask = action_mask(env)

        self.assertNotIn(env.ACTIONS.index("attack"), available)
        self.assertFalse(mask[env.ACTIONS.index("dash")])
        self.assertTrue(mask[env.ACTIONS.index("wait")])


class ActionRepeatTests(unittest.TestCase):
    def test_one_decision_advances_two_frames_and_aggregates_rewards(self) -> None:
        env = BossDodgeEnv(seed=7, max_steps=1200, initial_boss_hp=1)
        env.attack_phase = env.ATTACK_RECOVERY
        env.attack_timer = 10_000
        env._attack_hitbox = None
        env.boss_y = 200
        start_x = env.player_x

        _, reward, terminated, truncated, info = step_with_action_repeat(
            env, env.ACTIONS.index("right")
        )

        self.assertEqual(env.steps, ACTION_REPEAT)
        self.assertEqual(env.player_x, start_x + ACTION_REPEAT * env.PLAYER_SPEED)
        self.assertAlmostEqual(reward, env.STEP_PENALTY + FRAME_GAMMA * env.STEP_PENALTY)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["frames_advanced"], ACTION_REPEAT)

    def test_repeat_stops_on_terminal_frame(self) -> None:
        env = BossDodgeEnv(seed=7, max_steps=1200, initial_boss_hp=1)
        env.attack_phase = env.ATTACK_RECOVERY
        env.attack_timer = 10_000
        env._attack_hitbox = None
        env.player_x = 100
        env.player_y = env.boss_y
        env.player_facing = 1
        env.boss_x = 112

        _, _, terminated, _, info = step_with_action_repeat(env, env.ACTIONS.index("attack"))

        self.assertTrue(terminated)
        self.assertEqual(info["frames_advanced"], 1)


class CheckpointTests(unittest.TestCase):
    def checkpoint(self) -> dict[str, object]:
        network = DQN()
        return {
            "algorithm": "double-dqn",
            "actions": list(BossDodgeEnv.ACTIONS),
            "state_encoding": "normalized-observation-v1",
            "state_dimensions": STATE_DIMENSIONS,
            "hidden_dimensions": [128, 128],
            "online_state_dict": network.state_dict(),
            "boss_hp": 2,
        }

    def test_checkpoint_round_trip_builds_inference_network(self) -> None:
        path = Path("tests/.checkpoint_round_trip.pt")
        try:
            torch.save(self.checkpoint(), path)
            loaded = load_training_checkpoint(path)
            network = build_network_from_checkpoint(loaded)
        finally:
            path.unlink(missing_ok=True)

        self.assertFalse(network.training)
        self.assertEqual(network(torch.zeros(1, STATE_DIMENSIONS)).shape[1], len(BossDodgeEnv.ACTIONS))

    def test_missing_checkpoint_requires_reset(self) -> None:
        path = Path("does-not-exist.pt")
        with self.assertRaises(FileNotFoundError):
            load_training_checkpoint(path)
        self.assertIsNone(load_training_checkpoint(path, reset=True))


if __name__ == "__main__":
    unittest.main()
