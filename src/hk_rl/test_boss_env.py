"""Regression tests for the two-dimensional boss environment."""

from collections import deque
# 导入本模块依赖的类型与运行时工具。
import unittest

from .boss_env import BossDodgeEnv, Rect


# 定义 RectTests，组织相关状态和操作接口。
class RectTests(unittest.TestCase):
    def test_intersection_requires_positive_overlap(self) -> None:
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(Rect(0, 0, 10, 20).intersects(Rect(5, 10, 30, 50)))
        self.assertFalse(Rect(0, 0, 10, 20).intersects(Rect(10, 0, 30, 50)))


# 定义 BossDodgeEnvTests，组织相关状态和操作接口。
class BossDodgeEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        # 更新 self.env，使实例状态与当前帧保持一致。
        self.env = BossDodgeEnv(seed=7)

    def _disable_spikes(self) -> None:
        # 更新 self.env.attack_phase，使实例状态与当前帧保持一致。
        self.env.attack_phase = self.env.ATTACK_RECOVERY
        self.env.attack_timer = 10_000
        # 更新 self.env._attack_hitbox，使实例状态与当前帧保持一致。
        self.env._attack_hitbox = None

    def test_arena_and_entity_hitbox_dimensions(self) -> None:
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual((self.env.ARENA_WIDTH, self.env.ARENA_HEIGHT), (300, 300))
        self.assertEqual(
            (self.env.player_hitbox.width, self.env.player_hitbox.height),
            (10, 20),
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            (self.env.boss_hitbox.width, self.env.boss_hitbox.height),
            (30, 50),
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertIsNone(self.env.max_steps)

    def test_default_episode_has_no_time_limit(self) -> None:
        # 调用 self._disable_spikes 构造或推进测试场景。
        self._disable_spikes()
        self.env.steps = 10_000
        # 计算并保存 _、_、_、truncated、_，供后续逻辑直接复用。
        _, _, _, truncated, _ = self.env.step(self.env.ACTIONS.index("wait"))
        self.assertFalse(truncated)

    # 覆盖 initial boss hp is configurable and restored on reset 场景，防止对应行为发生回归。
    def test_initial_boss_hp_is_configurable_and_restored_on_reset(self) -> None:
        env = BossDodgeEnv(seed=7, initial_boss_hp=2)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(env.boss_hp, 2)
        env.boss_hp = 1
        # 调用 env.reset 构造或推进测试场景。
        env.reset()
        self.assertEqual(env.boss_hp, 2)

    # 覆盖 initial boss hp must be within supported range 场景，防止对应行为发生回归。
    def test_initial_boss_hp_must_be_within_supported_range(self) -> None:
        for invalid_hp in (0, self.env.INITIAL_BOSS_HP + 1):
            # 在受控上下文中执行操作，确保资源和异常得到正确处理。
            with self.subTest(initial_boss_hp=invalid_hp):
                with self.assertRaises(ValueError):
                    # 调用 BossDodgeEnv 构造或推进测试场景。
                    BossDodgeEnv(initial_boss_hp=invalid_hp)

    def test_progress_penalty_repeats_every_two_hundred_frames_without_ending(self) -> None:
        # 计算并保存 env，供后续逻辑直接复用。
        env = BossDodgeEnv(seed=7, max_steps=1200, initial_boss_hp=3)
        env.attack_phase = env.ATTACK_RECOVERY
        # 计算并保存 env.attack_timer，供后续逻辑直接复用。
        env.attack_timer = 10_000
        env._attack_hitbox = None
        # 计算并保存 env.boss_y，供后续逻辑直接复用。
        env.boss_y = 200
        wait = env.ACTIONS.index("wait")
        # 计算并保存 expected_penalty，供后续逻辑直接复用。
        expected_penalty = 3 / env.INITIAL_BOSS_HP

        for _ in range(199):
            # 调用 env.step 构造或推进测试场景。
            env.step(wait)
        _, reward_200, terminated, truncated, info_200 = env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(reward_200, env.STEP_PENALTY - expected_penalty)
        self.assertFalse(terminated)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(truncated)
        self.assertEqual(info_200["progress_penalty"], expected_penalty)

        # 计算并保存 _、reward_201、terminated、truncated、info_201，供后续逻辑直接复用。
        _, reward_201, terminated, truncated, info_201 = env.step(wait)
        self.assertEqual(reward_201, env.STEP_PENALTY)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(info_201["progress_penalty"], 0.0)

        for _ in range(198):
            # 调用 env.step 构造或推进测试场景。
            env.step(wait)
        _, reward_400, terminated, truncated, info_400 = env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(reward_400, env.STEP_PENALTY - expected_penalty)
        self.assertFalse(terminated)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(truncated)
        self.assertEqual(info_400["progress_penalty"], expected_penalty)

    # 覆盖 training time limit remains twelve hundred frames 场景，防止对应行为发生回归。
    def test_training_time_limit_remains_twelve_hundred_frames(self) -> None:
        env = BossDodgeEnv(seed=7, max_steps=1200, initial_boss_hp=1)
        # 计算并保存 env.attack_phase，供后续逻辑直接复用。
        env.attack_phase = env.ATTACK_RECOVERY
        env.attack_timer = 10_000
        # 计算并保存 env._attack_hitbox，供后续逻辑直接复用。
        env._attack_hitbox = None
        env.boss_y = 200
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = env.ACTIONS.index("wait")

        for _ in range(1199):
            # 计算并保存 _、_、terminated、truncated、_，供后续逻辑直接复用。
            _, _, terminated, truncated, _ = env.step(wait)
            self.assertFalse(terminated)
            # 核对关键输出，确认环境行为满足测试约束。
            self.assertFalse(truncated)
        _, _, terminated, truncated, _ = env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    # 覆盖 victory on penalty boundary has no progress penalty 场景，防止对应行为发生回归。
    def test_victory_on_penalty_boundary_has_no_progress_penalty(self) -> None:
        self._disable_spikes()
        # 更新 self.env.steps，使实例状态与当前帧保持一致。
        self.env.steps = self.env.PROGRESS_PENALTY_INTERVAL - 1
        self.env.player_x = 100
        # 更新 self.env.player_y，使实例状态与当前帧保持一致。
        self.env.player_y = self.env.boss_y
        self.env.player_velocity_y = 0
        # 更新 self.env.player_facing，使实例状态与当前帧保持一致。
        self.env.player_facing = 1
        self.env.boss_x = 112
        # 更新 self.env.boss_hp，使实例状态与当前帧保持一致。
        self.env.boss_hp = 1

        _, _, terminated, truncated, info = self.env.step(
            self.env.ACTIONS.index("attack")
        )

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(info["progress_penalty"], 0.0)

    def test_jump_requires_landing_before_second_jump(self) -> None:
        # 调用 self._disable_spikes 构造或推进测试场景。
        self._disable_spikes()
        jump = self.env.ACTIONS.index("jump")
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")

        self.env.step(jump)
        # 计算并保存 first_frame_velocity，供后续逻辑直接复用。
        first_frame_velocity = self.env.player_velocity_y
        self.env.step(jump)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertLess(self.env.player_velocity_y, first_frame_velocity)
        self.assertFalse(self.env.is_grounded)

        # 逐项处理当前序列，并累积这一轮所需的结果。
        for _ in range(100):
            if self.env.is_grounded:
                # 准备测试所需状态，并隔离无关的随机因素。
                break
            self.env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(self.env.is_grounded)

        self.env.step(jump)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertGreater(self.env.player_y, 0)

    def test_dash_locks_direction_for_five_frames_then_recovers_for_sixty(self) -> None:
        # 调用 self._disable_spikes 构造或推进测试场景。
        self._disable_spikes()
        left = self.env.ACTIONS.index("left")
        # 计算并保存 right，供后续逻辑直接复用。
        right = self.env.ACTIONS.index("right")
        dash = self.env.ACTIONS.index("dash")
        # 计算并保存 attack，供后续逻辑直接复用。
        attack = self.env.ACTIONS.index("attack")
        jump = self.env.ACTIONS.index("jump")
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        start = self.env.player_x
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(left)
        self.assertEqual(abs(self.env.player_x - start), 3)

        # 计算并保存 start，供后续逻辑直接复用。
        start = self.env.player_x
        self.env.step(dash)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.player_x - start, -self.env.DASH_SPEED)
        self.assertEqual(
            self.env.invulnerable_timer,
            self.env.DASH_INVULNERABILITY_FRAMES,
        )
        # 逐项处理当前序列，并累积这一轮所需的结果。
        for _ in range(self.env.DASH_FRAMES - 1):
            start = self.env.player_x
            # 调用 self.env.step 构造或推进测试场景。
            self.env.step(right)
            self.assertEqual(self.env.player_x - start, -self.env.DASH_SPEED)
            # 核对关键输出，确认环境行为满足测试约束。
            self.assertGreater(self.env.invulnerable_timer, 0)

        self.assertEqual(self.env.player_dash_timer, 0)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            self.env.player_dash_recovery_timer,
            self.env.PLAYER_DASH_RECOVERY_FRAMES,
        )
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(jump)
        self.assertGreater(self.env.player_y, 0)
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(attack)
        self.assertIsNotNone(self.env.sword_hitbox)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            self.env.player_attack_recovery_timer,
            self.env.PLAYER_ATTACK_RECOVERY_FRAMES,
        )

        # 逐项处理当前序列，并累积这一轮所需的结果。
        for _ in range(self.env.PLAYER_ATTACK_RECOVERY_FRAMES):
            self.env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.player_dash_recovery_timer, 0)
        self.env.step(jump)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertGreater(self.env.player_y, 0)

    def test_movement_keeps_player_box_inside_arena(self) -> None:
        # 调用 self._disable_spikes 构造或推进测试场景。
        self._disable_spikes()
        left = self.env.ACTIONS.index("left")
        # 计算并保存 right，供后续逻辑直接复用。
        right = self.env.ACTIONS.index("right")
        for _ in range(400):
            # 调用 self.env.step 构造或推进测试场景。
            self.env.step(left)
        self.assertEqual(self.env.player_x, 0)
        # 逐项处理当前序列，并累积这一轮所需的结果。
        for _ in range(400):
            self.env.step(right)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.player_x, 290)

    def test_approaching_boss_has_no_extra_reward(self) -> None:
        # 调用 self._disable_spikes 构造或推进测试场景。
        self._disable_spikes()
        _, reward, _, _, _ = self.env.step(self.env.ACTIONS.index("right"))
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(reward, self.env.STEP_PENALTY)

    def test_entering_attack_range_rewards_only_once(self) -> None:
        # 调用 self._disable_spikes 构造或推进测试场景。
        self._disable_spikes()
        self.env.player_x = 100
        # 更新 self.env.player_y，使实例状态与当前帧保持一致。
        self.env.player_y = self.env.boss_y
        self.env.player_velocity_y = 0
        # 更新 self.env.boss_x，使实例状态与当前帧保持一致。
        self.env.boss_x = 115
        self.env.boss_velocity_x = -1

        # 计算并保存 _、first_reward、_、_、first_info，供后续逻辑直接复用。
        _, first_reward, _, _, first_info = self.env.step(
            self.env.ACTIONS.index("wait")
        )
        # 计算并保存 _、second_reward、_、_、second_info，供后续逻辑直接复用。
        _, second_reward, _, _, second_info = self.env.step(
            self.env.ACTIONS.index("wait")
        )

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            first_reward,
            self.env.STEP_PENALTY + self.env.ATTACK_RANGE_REWARD,
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(first_info["entered_attack_range"])
        self.assertEqual(second_reward, self.env.STEP_PENALTY)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(second_info["entered_attack_range"])

    def test_boss_reward_zone_extends_one_sword_width_on_both_sides(self) -> None:
        # 计算并保存 zone，供后续逻辑直接复用。
        zone = self.env.boss_reward_hitbox

        self.assertEqual(zone.x, self.env.boss_x - self.env.SWORD_WIDTH)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            zone.width,
            self.env.BOSS_WIDTH + 2 * self.env.SWORD_WIDTH,
        )

    # 覆盖 attack range reward does not require facing the boss 场景，防止对应行为发生回归。
    def test_attack_range_reward_does_not_require_facing_the_boss(self) -> None:
        self._disable_spikes()
        # 更新 self.env.player_x，使实例状态与当前帧保持一致。
        self.env.player_x = 120
        self.env.player_y = self.env.boss_y
        # 更新 self.env.player_velocity_y，使实例状态与当前帧保持一致。
        self.env.player_velocity_y = 0
        self.env.player_facing = -1
        # 更新 self.env.boss_x，使实例状态与当前帧保持一致。
        self.env.boss_x = 150
        self.env.boss_velocity_x = -1

        # 计算并保存 _、reward、_、_、info，供后续逻辑直接复用。
        _, reward, _, _, info = self.env.step(self.env.ACTIONS.index("wait"))

        self.assertEqual(
            reward,
            self.env.STEP_PENALTY + self.env.ATTACK_RANGE_REWARD,
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(info["entered_attack_range"])
        self.assertFalse(self.env._make_sword_hitbox().intersects(self.env.boss_hitbox))

    # 覆盖 boss moves one grid each frame 场景，防止对应行为发生回归。
    def test_boss_moves_one_grid_each_frame(self) -> None:
        self._disable_spikes()
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        before = self.env.boss_x
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(wait)
        self.assertEqual(abs(self.env.boss_x - before), 1)

    # 覆盖 sword box follows facing and lasts one frame 场景，防止对应行为发生回归。
    def test_sword_box_follows_facing_and_lasts_one_frame(self) -> None:
        self._disable_spikes()
        # 计算并保存 attack，供后续逻辑直接复用。
        attack = self.env.ACTIONS.index("attack")
        left = self.env.ACTIONS.index("left")
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")

        self.env.step(attack)
        # 计算并保存 sword，供后续逻辑直接复用。
        sword = self.env.sword_hitbox
        self.assertEqual(
            (sword.width, sword.height),
            (self.env.PLAYER_WIDTH * 2, self.env.PLAYER_HEIGHT),
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(sword.x, self.env.player_x + self.env.PLAYER_WIDTH)
        self.assertEqual(sword.y, self.env.player_y)

        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(left)
        for _ in range(self.env.FRAMES_PER_SECOND - 1):
            # 调用 self.env.step 构造或推进测试场景。
            self.env.step(wait)
        self.env.step(attack)
        # 计算并保存 sword，供后续逻辑直接复用。
        sword = self.env.sword_hitbox
        self.assertEqual(sword.x + sword.width, self.env.player_x)
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(wait)
        self.assertIsNone(self.env.sword_hitbox)

    # 覆盖 sword collision damages boss 场景，防止对应行为发生回归。
    def test_sword_collision_damages_boss(self) -> None:
        self._disable_spikes()
        # 更新 self.env.player_x，使实例状态与当前帧保持一致。
        self.env.player_x = 100
        self.env.player_y = self.env.boss_y
        # 更新 self.env.player_velocity_y，使实例状态与当前帧保持一致。
        self.env.player_velocity_y = 0
        self.env.player_facing = 1
        # 更新 self.env.boss_x，使实例状态与当前帧保持一致。
        self.env.boss_x = 112
        self.env.boss_hp = 1
        # 计算并保存 hp_before，供后续逻辑直接复用。
        hp_before = self.env.boss_hp
        distance_before = abs(self.env.boss_x - self.env.player_x)

        # 计算并保存 _、reward、_、_、info，供后续逻辑直接复用。
        _, reward, _, _, info = self.env.step(self.env.ACTIONS.index("attack"))

        self.assertEqual(self.env.boss_hp, hp_before - 1)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(info["boss_hit"])
        self.assertTrue(info["boss_teleported"])
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertGreater(abs(self.env.boss_x - self.env.player_x), distance_before)
        self.assertEqual(
            reward,
            self.env.STEP_PENALTY
            + self.env.ATTACK_RANGE_REWARD
            + self.env.BOSS_HIT_REWARD
            + self.env.VICTORY_REWARD,
        )

    # 覆盖 player attack recovery blocks only another attack 场景，防止对应行为发生回归。
    def test_player_attack_recovery_blocks_only_another_attack(self) -> None:
        self._disable_spikes()
        # 计算并保存 attack，供后续逻辑直接复用。
        attack = self.env.ACTIONS.index("attack")
        jump = self.env.ACTIONS.index("jump")
        # 计算并保存 dash，供后续逻辑直接复用。
        dash = self.env.ACTIONS.index("dash")

        self.env.step(attack)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            self.env.player_attack_recovery_timer,
            self.env.FRAMES_PER_SECOND,
        )
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(jump)
        self.assertGreater(self.env.player_y, 0)

        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(attack)
        self.assertIsNone(self.env.sword_hitbox)
        # 计算并保存 recovery_after_blocked_attack，供后续逻辑直接复用。
        recovery_after_blocked_attack = self.env.player_attack_recovery_timer

        start_x = self.env.player_x
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(dash)
        self.assertNotEqual(self.env.player_x, start_x)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            self.env.player_attack_recovery_timer,
            recovery_after_blocked_attack - 1,
        )

    # 覆盖 player can move left right and falls during attack recovery 场景，防止对应行为发生回归。
    def test_player_can_move_left_right_and_falls_during_attack_recovery(self) -> None:
        self._disable_spikes()
        # 更新 self.env.player_y，使实例状态与当前帧保持一致。
        self.env.player_y = 30
        self.env.player_velocity_y = 0
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(self.env.ACTIONS.index("attack"))
        attack_y = self.env.player_y
        # 计算并保存 start_x，供后续逻辑直接复用。
        start_x = self.env.player_x

        self.env.step(self.env.ACTIONS.index("left"))
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.player_x, start_x - 3)
        self.assertLess(self.env.player_y, attack_y)
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(self.env.ACTIONS.index("jump"))
        self.assertLess(self.env.player_y, attack_y)

        # 计算并保存 sword，供后续逻辑直接复用。
        sword = self.env.sword_hitbox
        self.assertIsNone(sword)
        # 逐项处理当前序列，并累积这一轮所需的结果。
        for _ in range(self.env.FRAMES_PER_SECOND):
            if self.env.is_grounded:
                # 准备测试所需状态，并隔离无关的随机因素。
                break
            self.env.step(self.env.ACTIONS.index("wait"))
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(self.env.is_grounded)

    def test_touching_boss_damages_player(self) -> None:
        # 调用 self._disable_spikes 构造或推进测试场景。
        self._disable_spikes()
        self.env.player_x = self.env.boss_x
        # 更新 self.env.player_y，使实例状态与当前帧保持一致。
        self.env.player_y = self.env.boss_y
        self.env.player_velocity_y = 0
        # 计算并保存 hp_before，供后续逻辑直接复用。
        hp_before = self.env.player_hp

        self.env.step(self.env.ACTIONS.index("wait"))

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.player_hp, hp_before - 1)

    def test_ground_spike_targets_a_random_past_x_position(self) -> None:
        # 计算并保存 history，供后续逻辑直接复用。
        history = [40.0 + index for index in range(13)]
        self.env._player_x_history = deque(history, maxlen=13)
        # 调用 self.env._summon_boss_attack 构造或推进测试场景。
        self.env._summon_boss_attack()

        age = self.env.attack_target_age
        # 计算并保存 target_x，供后续逻辑直接复用。
        target_x = history[-(age + 1)]
        expected = target_x + self.env.PLAYER_WIDTH / 2 - self.env.BOSS_ATTACK_WIDTH / 2
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertGreaterEqual(age, self.env.ATTACK_TRACK_MIN_FRAMES)
        self.assertLessEqual(age, self.env.ATTACK_TRACK_MAX_FRAMES)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.attack_hitbox.x, expected)

    def test_ground_spike_collision_damages_player(self) -> None:
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        self.env.step(wait)
        # 在结束条件满足前持续推进当前流程。
        while self.env.attack_phase == self.env.ATTACK_WARNING:
            self.env.step(wait)
        # 计算并保存 hp_before，供后续逻辑直接复用。
        hp_before = self.env.player_hp
        self.env.step(wait)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.player_hp, hp_before - 1)
        self.assertIsNotNone(self.env.attack_hitbox)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.attack_phase, self.env.ATTACK_ACTIVE)

    def test_ground_spike_has_long_active_phase_and_ten_frame_recovery(self) -> None:
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        self.env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertGreaterEqual(self.env.attack_timer, 12)
        self.assertLessEqual(self.env.attack_timer, 18)
        # 更新 self.env.attack_timer，使实例状态与当前帧保持一致。
        self.env.attack_timer = 1
        self.env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.attack_phase, self.env.ATTACK_ACTIVE)
        self.assertEqual(self.env.attack_timer, 30)

        # 计算并保存 hp_before，供后续逻辑直接复用。
        hp_before = self.env.player_hp
        for _ in range(29):
            # 调用 self.env.step 构造或推进测试场景。
            self.env.step(wait)
        self.assertIsNotNone(self.env.attack_hitbox)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.attack_timer, 1)
        self.env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.player_hp, hp_before - 1)
        self.assertIsNone(self.env.attack_hitbox)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.attack_phase, self.env.ATTACK_RECOVERY)
        self.assertEqual(self.env.attack_timer, 10)

        # 逐项处理当前序列，并累积这一轮所需的结果。
        for _ in range(10):
            self.env.step(wait)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(self.env.attack_phase, self.env.ATTACK_IDLE)
        self.assertEqual(self.env.attack_timer, 0)

    # 覆盖 jump can clear ground spike 场景，防止对应行为发生回归。
    def test_jump_can_clear_ground_spike(self) -> None:
        jump = self.env.ACTIONS.index("jump")
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        self.env.step(wait)
        # 更新 self.env.attack_timer，使实例状态与当前帧保持一致。
        self.env.attack_timer = 4
        hp_before = self.env.player_hp
        # 调用 self.env.step 构造或推进测试场景。
        self.env.step(jump)
        for _ in range(4):
            # 调用 self.env.step 构造或推进测试场景。
            self.env.step(wait)

        self.assertEqual(self.env.player_hp, hp_before)

    # 覆盖 completely avoiding ground spike is rewarded 场景，防止对应行为发生回归。
    def test_completely_avoiding_ground_spike_is_rewarded(self) -> None:
        right = self.env.ACTIONS.index("right")
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        self.env.step(wait)
        # 计算并保存 reward，供后续逻辑直接复用。
        reward = 0.0
        rewarded_steps = 0
        # 计算并保存 info，供后续逻辑直接复用。
        info = {}

        while self.env.attack_phase != self.env.ATTACK_RECOVERY:
            # 计算并保存 _、step_reward、_、_、info，供后续逻辑直接复用。
            _, step_reward, _, _, info = self.env.step(right)
            reward += step_reward
            # 计算并保存 rewarded_steps，供后续逻辑直接复用。
            rewarded_steps += 1

        self.assertEqual(self.env.player_hp, self.env.INITIAL_PLAYER_HP)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertAlmostEqual(
            reward,
            rewarded_steps * self.env.STEP_PENALTY + self.env.DODGE_REWARD,
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(info["spike_dodged"])

    def test_every_successful_dodge_is_rewarded(self) -> None:
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")

        def avoid_one_spike() -> tuple[float, dict[str, object]]:
            # 更新 self.env.attack_phase，使实例状态与当前帧保持一致。
            self.env.attack_phase = self.env.ATTACK_ACTIVE
            self.env.attack_timer = 1
            # 更新 self.env._attack_hitbox，使实例状态与当前帧保持一致。
            self.env._attack_hitbox = Rect(250, 0, 36, 24)
            self.env._attack_has_hit = False
            # 更新 self.env._attack_hurt_player，使实例状态与当前帧保持一致。
            self.env._attack_hurt_player = False
            _, reward, _, _, info = self.env.step(wait)
            # 返回已经整理好的结果，供上层流程继续使用。
            return reward, info

        first_reward, first_info = avoid_one_spike()
        # 计算并保存 second_reward、second_info，供后续逻辑直接复用。
        second_reward, second_info = avoid_one_spike()
        third_reward, third_info = avoid_one_spike()

        # 计算并保存 expected，供后续逻辑直接复用。
        expected = self.env.STEP_PENALTY + self.env.DODGE_REWARD
        self.assertEqual(first_reward, expected)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(second_reward, expected)
        self.assertEqual(third_reward, expected)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(first_info["spike_dodged"])
        self.assertTrue(second_info["spike_dodged"])
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(third_info["spike_dodged"])

    def test_remaining_in_spike_box_after_grace_period_is_penalized_once(self) -> None:
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        self.env.boss_y = 200
        # 更新 self.env.attack_phase，使实例状态与当前帧保持一致。
        self.env.attack_phase = self.env.ATTACK_WARNING
        self.env.attack_timer = 12
        # 更新 self.env.attack_warning_elapsed，使实例状态与当前帧保持一致。
        self.env.attack_warning_elapsed = self.env.SPIKE_ESCAPE_GRACE_FRAMES - 1
        self.env._attack_hitbox = Rect(self.env.player_x, 0, 36, 12)

        # 计算并保存 _、first_reward、_、_、first_info，供后续逻辑直接复用。
        _, first_reward, _, _, first_info = self.env.step(wait)
        _, second_reward, _, _, second_info = self.env.step(wait)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(
            first_reward,
            self.env.STEP_PENALTY + self.env.SPIKE_ESCAPE_TIMEOUT_PENALTY,
        )
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertTrue(first_info["spike_escape_timeout"])
        self.assertEqual(second_reward, self.env.STEP_PENALTY)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(second_info["spike_escape_timeout"])

    def test_leaving_spike_box_before_grace_period_avoids_penalty(self) -> None:
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        self.env.boss_y = 200
        # 更新 self.env.attack_phase，使实例状态与当前帧保持一致。
        self.env.attack_phase = self.env.ATTACK_WARNING
        self.env.attack_timer = 12
        # 更新 self.env.attack_warning_elapsed，使实例状态与当前帧保持一致。
        self.env.attack_warning_elapsed = self.env.SPIKE_ESCAPE_GRACE_FRAMES - 1
        self.env._attack_hitbox = Rect(200, 0, 36, 12)

        # 计算并保存 _、reward、_、_、info，供后续逻辑直接复用。
        _, reward, _, _, info = self.env.step(wait)

        self.assertEqual(reward, self.env.STEP_PENALTY)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(info["spike_escape_timeout"])

    def test_jumping_above_spike_before_grace_period_avoids_penalty(self) -> None:
        # 计算并保存 wait，供后续逻辑直接复用。
        wait = self.env.ACTIONS.index("wait")
        self.env.boss_y = 200
        # 更新 self.env.player_y，使实例状态与当前帧保持一致。
        self.env.player_y = 20
        self.env.player_velocity_y = 0
        # 更新 self.env.attack_phase，使实例状态与当前帧保持一致。
        self.env.attack_phase = self.env.ATTACK_WARNING
        self.env.attack_timer = 12
        # 更新 self.env.attack_warning_elapsed，使实例状态与当前帧保持一致。
        self.env.attack_warning_elapsed = self.env.SPIKE_ESCAPE_GRACE_FRAMES - 1
        self.env._attack_hitbox = Rect(self.env.player_x, 0, 36, 12)

        # 计算并保存 _、reward、_、_、info，供后续逻辑直接复用。
        _, reward, _, _, info = self.env.step(wait)

        self.assertEqual(reward, self.env.STEP_PENALTY)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertFalse(info["spike_escape_timeout"])


if __name__ == "__main__":
    # 调用 unittest.main 构造或推进测试场景。
    unittest.main()
