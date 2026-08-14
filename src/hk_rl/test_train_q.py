"""Tests for the tabular state representation."""

import argparse
# 导入本模块依赖的类型与运行时工具。
import json
from pathlib import Path
# 导入本模块依赖的类型与运行时工具。
import random
import unittest
# 导入本模块依赖的类型与运行时工具。
from unittest.mock import MagicMock

from .boss_env import BossDodgeEnv, Rect
# 导入本模块依赖的类型与运行时工具。
from .train_q import (
    ACTION_REPEAT,
    FRAME_GAMMA,
    GAMMA,
    LAMBDA,
    STATE_DIMENSIONS,
    available_action_indices,
    encode_state,
    expected_epsilon_greedy_value,
    load_training_checkpoint,
    parse_bool,
    select_greedy_action,
    step_with_action_repeat,
)


# 定义 StateEncodingTests，组织相关状态和操作接口。
class StateEncodingTests(unittest.TestCase):
    def test_player_and_boss_hp_are_not_encoded(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7)
        before = encode_state(env._observation())
        # 计算并保存 env.player_hp，供后续逻辑直接复用。
        env.player_hp = 2
        env.boss_hp = 4

        # 计算并保存 after，供后续逻辑直接复用。
        after = encode_state(env._observation())

        self.assertEqual(after, before)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(len(after), STATE_DIMENSIONS)

    def test_nearby_boss_positions_share_one_quantized_state(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7)
        env.boss_x = 190
        # 计算并保存 before，供后续逻辑直接复用。
        before = encode_state(env._observation())
        env.boss_x = 194
        # 计算并保存 nearby，供后续逻辑直接复用。
        nearby = encode_state(env._observation())

        self.assertEqual(nearby, before)

    # 覆盖 left and right walls are encoded 场景，防止对应行为发生回归。
    def test_left_and_right_walls_are_encoded(self) -> None:
        env = BossDodgeEnv(seed=7)

        # 计算并保存 env.player_x，供后续逻辑直接复用。
        env.player_x = 0
        at_left_wall = encode_state(env._observation())
        # 计算并保存 env.player_x，供后续逻辑直接复用。
        env.player_x = 100
        away_from_walls = encode_state(env._observation())
        # 计算并保存 env.player_x，供后续逻辑直接复用。
        env.player_x = env.ARENA_WIDTH - env.PLAYER_WIDTH
        at_right_wall = encode_state(env._observation())

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(at_left_wall[6], -1)
        self.assertEqual(away_from_walls[6], 0)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(at_right_wall[6], 1)

    def test_attack_timer_noise_is_merged_into_urgency_bands(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7)
        env.attack_phase = env.ATTACK_WARNING
        # 计算并保存 env._attack_hitbox，供后续逻辑直接复用。
        env._attack_hitbox = Rect(70, 0, 36, 12)
        env.attack_timer = 18
        # 计算并保存 early_warning，供后续逻辑直接复用。
        early_warning = encode_state(env._observation())
        env.attack_timer = 13
        # 计算并保存 nearby_timer，供后续逻辑直接复用。
        nearby_timer = encode_state(env._observation())

        self.assertEqual(early_warning, nearby_timer)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(early_warning[10], 1)

    def test_spike_overlap_encodes_shortest_escape_direction(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7)
        env.attack_phase = env.ATTACK_WARNING
        # 计算并保存 env.attack_timer，供后续逻辑直接复用。
        env.attack_timer = 12
        env._attack_hitbox = Rect(env.player_x - 20, 0, 36, 12)
        # 计算并保存 escape_right，供后续逻辑直接复用。
        escape_right = encode_state(env._observation())
        env._attack_hitbox = Rect(env.player_x - 6, 0, 36, 12)
        # 计算并保存 escape_left，供后续逻辑直接复用。
        escape_left = encode_state(env._observation())

        self.assertEqual(escape_right[9], 1)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(escape_left[9], -1)

    def test_exact_attackability_survives_coarse_distance_binning(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7)
        env.player_x = 100
        # 计算并保存 env.player_y，供后续逻辑直接复用。
        env.player_y = env.boss_y
        env.player_facing = 1
        # 计算并保存 env.boss_x，供后续逻辑直接复用。
        env.boss_x = 112
        attackable = encode_state(env._observation())
        # 计算并保存 env.boss_x，供后续逻辑直接复用。
        env.boss_x = 160
        too_far = encode_state(env._observation())

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(attackable[8], 1)
        self.assertEqual(too_far[8], 0)


# 定义 ExpectedSarsaTests，组织相关状态和操作接口。
class ExpectedSarsaTests(unittest.TestCase):
    def test_epsilon_greedy_expected_value_includes_exploration(self) -> None:
        # 计算并保存 expected，供后续逻辑直接复用。
        expected = expected_epsilon_greedy_value([1.0, 3.0], epsilon=0.2)

        self.assertAlmostEqual(expected, 2.8)

    # 覆盖 trace retains reward credit across sixty frames 场景，防止对应行为发生回归。
    def test_trace_retains_reward_credit_across_sixty_frames(self) -> None:
        sixty_frame_weight = (GAMMA * LAMBDA) ** (60 // ACTION_REPEAT)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertGreater(sixty_frame_weight, 0.4)

    def test_all_zero_q_values_do_not_always_select_left(self) -> None:
        # 计算并保存 rng，供后续逻辑直接复用。
        rng = random.Random(7)
        selected = {
            select_greedy_action([0.0] * 6, rng)
            for _ in range(30)
        }

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertGreater(len(selected), 1)
        self.assertNotEqual(selected, {0})

    # 覆盖 greedy tie breaking only uses maximum actions 场景，防止对应行为发生回归。
    def test_greedy_tie_breaking_only_uses_maximum_actions(self) -> None:
        rng = random.Random(7)
        # 计算并保存 selected，供后续逻辑直接复用。
        selected = {
            select_greedy_action([1.0, 3.0, 3.0, 2.0], rng)
            for _ in range(30)
        }

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(selected, {1, 2})

    def test_unavailable_actions_are_excluded_from_policy_and_expectation(self) -> None:
        # 计算并保存 rng，供后续逻辑直接复用。
        rng = random.Random(7)
        values = [100.0, 3.0, 1.0]

        # 计算并保存 selected，供后续逻辑直接复用。
        selected = select_greedy_action(values, rng, (1, 2))
        expected = expected_epsilon_greedy_value(values, 0.2, (1, 2))

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(selected, 1)
        self.assertAlmostEqual(expected, 2.8)

    # 覆盖 attack is masked during recovery but movement is available 场景，防止对应行为发生回归。
    def test_attack_is_masked_during_recovery_but_movement_is_available(self) -> None:
        env = BossDodgeEnv(seed=7)
        # 计算并保存 env.player_attack_recovery_timer，供后续逻辑直接复用。
        env.player_attack_recovery_timer = 30

        available = available_action_indices(env)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertNotIn(env.ACTIONS.index("attack"), available)
        self.assertIn(env.ACTIONS.index("jump"), available)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertIn(env.ACTIONS.index("dash"), available)


class ActionRepeatTests(unittest.TestCase):
    # 覆盖 one decision advances two frames and aggregates rewards 场景，防止对应行为发生回归。
    def test_one_decision_advances_two_frames_and_aggregates_rewards(self) -> None:
        env = BossDodgeEnv(seed=7, max_steps=1200, initial_boss_hp=1)
        # 计算并保存 env.attack_phase，供后续逻辑直接复用。
        env.attack_phase = env.ATTACK_RECOVERY
        env.attack_timer = 10_000
        # 计算并保存 env._attack_hitbox，供后续逻辑直接复用。
        env._attack_hitbox = None
        env.boss_y = 200
        # 计算并保存 start_x，供后续逻辑直接复用。
        start_x = env.player_x

        _, reward, terminated, truncated, info = step_with_action_repeat(
            env,
            env.ACTIONS.index("right"),
        )

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(env.steps, ACTION_REPEAT)
        self.assertEqual(env.player_x, start_x + ACTION_REPEAT * env.PLAYER_SPEED)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertAlmostEqual(
            reward,
            env.STEP_PENALTY + FRAME_GAMMA * env.STEP_PENALTY,
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(info["frames_advanced"], ACTION_REPEAT)

    def test_action_repeat_stops_immediately_after_victory(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7, max_steps=1200, initial_boss_hp=1)
        env.attack_phase = env.ATTACK_RECOVERY
        # 计算并保存 env.attack_timer，供后续逻辑直接复用。
        env.attack_timer = 10_000
        env._attack_hitbox = None
        # 计算并保存 env.player_x，供后续逻辑直接复用。
        env.player_x = 100
        env.player_y = env.boss_y
        # 计算并保存 env.player_velocity_y，供后续逻辑直接复用。
        env.player_velocity_y = 0
        env.player_facing = 1
        # 计算并保存 env.boss_x，供后续逻辑直接复用。
        env.boss_x = 112

        _, _, terminated, truncated, info = step_with_action_repeat(
            env,
            env.ACTIONS.index("attack"),
        )

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(env.steps, 1)
        self.assertEqual(info["frames_advanced"], 1)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(info["boss_hit"])

    def test_first_frame_progress_penalty_is_retained_in_combined_info(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7, max_steps=1200, initial_boss_hp=3)
        env.attack_phase = env.ATTACK_RECOVERY
        # 计算并保存 env.attack_timer，供后续逻辑直接复用。
        env.attack_timer = 10_000
        env._attack_hitbox = None
        # 计算并保存 env.boss_y，供后续逻辑直接复用。
        env.boss_y = 200
        env.steps = env.PROGRESS_PENALTY_INTERVAL - 1

        # 计算并保存 _、_、_、_、info，供后续逻辑直接复用。
        _, _, _, _, info = step_with_action_repeat(
            env,
            env.ACTIONS.index("wait"),
        )

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(info["progress_penalty"], 3 / env.INITIAL_BOSS_HP)


class CheckpointLoadingTests(unittest.TestCase):
    # 覆盖 existing checkpoint is loaded by default 场景，防止对应行为发生回归。
    def test_existing_checkpoint_is_loaded_by_default(self) -> None:
        path = MagicMock(spec=Path)
        # 计算并保存 expected，供后续逻辑直接复用。
        expected = {"q_values": {"state": [1.0]}}
        path.exists.return_value = True
        # 计算并保存 path.read_text.return_value，供后续逻辑直接复用。
        path.read_text.return_value = json.dumps(expected)

        self.assertEqual(load_training_checkpoint(path), expected)
        # 调用 path.read_text.assert_called_once_with 构造或推进测试场景。
        path.read_text.assert_called_once_with(encoding="utf-8")

    def test_missing_checkpoint_requires_explicit_reset(self) -> None:
        # 计算并保存 path，供后续逻辑直接复用。
        path = MagicMock(spec=Path)
        path.exists.return_value = False

        # 在受控上下文中执行操作，确保资源和异常得到正确处理。
        with self.assertRaises(FileNotFoundError):
            load_training_checkpoint(path)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertIsNone(load_training_checkpoint(path, reset=True))

    def test_reset_boolean_must_be_explicit_true_or_false(self) -> None:
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(parse_bool("true"))
        self.assertFalse(parse_bool("FALSE"))
        # 在受控上下文中执行操作，确保资源和异常得到正确处理。
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_bool("yes")


# 根据当前条件选择对应分支，保持状态转换符合规则。
if __name__ == "__main__":
    unittest.main()
