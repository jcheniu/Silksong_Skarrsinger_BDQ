"""Tests for manual input handling without opening a Turtle window."""

from collections import deque
# 导入本模块依赖的类型与运行时工具。
import random
import unittest
# 导入本模块依赖的类型与运行时工具。
from unittest.mock import MagicMock, patch

from .boss_env import BossDodgeEnv
# 导入本模块依赖的类型与运行时工具。
from .train_q import ACTION_REPEAT
from .visualize import Visualizer


# 定义 FakeCanvas，组织相关状态和操作接口。
class FakeCanvas:
    def __init__(self) -> None:
        # 更新 self.jobs，使实例状态与当前帧保持一致。
        self.jobs: dict[str, object] = {}
        self.cancelled: list[str] = []

    # 定义 after，集中处理这一阶段的输入与状态变化。
    def after(self, _delay: int, callback: object) -> str:
        job = f"job-{len(self.jobs)}"
        # 计算并保存 当前状态，供后续逻辑直接复用。
        self.jobs[job] = callback
        return job

    # 定义 after_cancel，集中处理这一阶段的输入与状态变化。
    def after_cancel(self, job: str) -> None:
        self.cancelled.append(job)
        # 调用 self.jobs.pop 构造或推进测试场景。
        self.jobs.pop(job, None)


class FakeScreen:
    # 定义 __init__，集中处理这一阶段的输入与状态变化。
    def __init__(self) -> None:
        self.canvas = FakeCanvas()

    # 定义 getcanvas，集中处理这一阶段的输入与状态变化。
    def getcanvas(self) -> FakeCanvas:
        return self.canvas


# 定义 input_only_visualizer，集中处理这一阶段的输入与状态变化。
def input_only_visualizer() -> Visualizer:
    visualizer = Visualizer.__new__(Visualizer)
    # 计算并保存 visualizer.env，供后续逻辑直接复用。
    visualizer.env = BossDodgeEnv(seed=7)
    visualizer.observation, _ = visualizer.env.reset(seed=7)
    # 计算并保存 visualizer.q_values，供后续逻辑直接复用。
    visualizer.q_values = {}
    visualizer.rng = random.Random(7)
    # 计算并保存 visualizer._policy_action_remaining，供后续逻辑直接复用。
    visualizer._policy_action_remaining = 0
    visualizer._current_policy_action = visualizer.env.ACTIONS.index("wait")
    # 计算并保存 visualizer.screen，供后续逻辑直接复用。
    visualizer.screen = FakeScreen()
    visualizer._held_keys = {}
    # 计算并保存 visualizer._pending_actions，供后续逻辑直接复用。
    visualizer._pending_actions = deque()
    visualizer._release_jobs = {}
    # 返回已经整理好的结果，供上层流程继续使用。
    return visualizer


class ManualInputTests(unittest.TestCase):
    # 覆盖 scene creates player in front without private canvas attributes 场景，防止对应行为发生回归。
    def test_scene_creates_player_in_front_without_private_canvas_attributes(self) -> None:
        creation_order: list[str] = []

        # 定义 make_sprite，集中处理这一阶段的输入与状态变化。
        def make_sprite(_width: float, _height: float, color: str) -> MagicMock:
            names = {
                "#7c3aed": "boss",
                "#f59e0b": "attack",
                "#06b6d4": "sword",
                "#2563eb": "player",
            }
            # 调用 creation_order.append 构造或推进测试场景。
            creation_order.append(names[color])
            return MagicMock()

        # 在受控上下文中执行操作，确保资源和异常得到正确处理。
        with (
            patch("hk_rl.visualize.turtle.Screen", return_value=MagicMock()),
            patch("hk_rl.visualize.turtle.Turtle", return_value=MagicMock()),
            patch.object(Visualizer, "_draw_arena", side_effect=lambda: creation_order.append("arena")),
            patch.object(Visualizer, "_sprite", side_effect=make_sprite),
        ):
            # 调用 Visualizer 构造或推进测试场景。
            Visualizer(q_data=None, seed=7, delay_ms=16, manual=False)

        self.assertEqual(
            creation_order,
            ["arena", "boss", "attack", "sword", "player"],
        )

    # 覆盖 held direction is returned on every frame 场景，防止对应行为发生回归。
    def test_held_direction_is_returned_on_every_frame(self) -> None:
        visualizer = input_only_visualizer()
        # 计算并保存 right，供后续逻辑直接复用。
        right = visualizer.env.ACTIONS.index("right")

        visualizer._press_key("Right", right)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(visualizer._manual_action(), right)
        self.assertEqual(visualizer._manual_action(), right)

    # 覆盖 jump triggers once while held 场景，防止对应行为发生回归。
    def test_jump_triggers_once_while_held(self) -> None:
        visualizer = input_only_visualizer()
        # 计算并保存 jump，供后续逻辑直接复用。
        jump = visualizer.env.ACTIONS.index("jump")
        wait = visualizer.env.ACTIONS.index("wait")

        # 调用 visualizer._press_key 构造或推进测试场景。
        visualizer._press_key("space", jump)
        visualizer._press_key("space", jump)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(visualizer._manual_action(), jump)
        self.assertEqual(visualizer._manual_action(), wait)

    # 覆盖 auto repeat press cancels delayed release 场景，防止对应行为发生回归。
    def test_auto_repeat_press_cancels_delayed_release(self) -> None:
        visualizer = input_only_visualizer()
        # 计算并保存 jump，供后续逻辑直接复用。
        jump = visualizer.env.ACTIONS.index("jump")

        visualizer._press_key("space", jump)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(visualizer._manual_action(), jump)
        visualizer._release_key("space")
        # 计算并保存 release_job，供后续逻辑直接复用。
        release_job = visualizer._release_jobs["space"]
        visualizer._press_key("space", jump)

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertIn(release_job, visualizer.screen.canvas.cancelled)
        self.assertFalse(visualizer._pending_actions)

    # 覆盖 unseen policy state does not always select left 场景，防止对应行为发生回归。
    def test_unseen_policy_state_does_not_always_select_left(self) -> None:
        visualizer = input_only_visualizer()

        # 计算并保存 selected，供后续逻辑直接复用。
        selected = {visualizer._policy_action() for _ in range(30)}

        self.assertGreater(len(selected), 1)
        # 核对关键输出，确认环境行为满足测试约束。
        self.assertNotEqual(selected, {visualizer.env.ACTIONS.index("left")})

    def test_policy_reuses_each_selected_action_for_two_frames(self) -> None:
        # 计算并保存 visualizer，供后续逻辑直接复用。
        visualizer = input_only_visualizer()
        visualizer.rng = MagicMock()
        # 计算并保存 visualizer.rng.choice.side_effect，供后续逻辑直接复用。
        visualizer.rng.choice.side_effect = [1, 2]

        selected = [
            visualizer._policy_action()
            for _ in range(ACTION_REPEAT + 1)
        ]

        # 核对关键输出，确认环境行为满足测试约束。
        self.assertEqual(selected, [1, 1, 2])


if __name__ == "__main__":
    # 调用 unittest.main 构造或推进测试场景。
    unittest.main()
