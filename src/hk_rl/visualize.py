"""Visualize a trained policy or play the boss arena with ``turtle``."""

from __future__ import annotations

# 导入本模块依赖的类型与运行时工具。
import argparse
from collections import deque
# 导入本模块依赖的类型与运行时工具。
import json
from pathlib import Path
# 导入本模块依赖的类型与运行时工具。
import random
import turtle
# 导入本模块依赖的类型与运行时工具。
from typing import Any

from .boss_env import BossDodgeEnv, Rect
# 导入本模块依赖的类型与运行时工具。
from .train_q import (
    ACTION_REPEAT,
    STATE_ENCODING,
    available_action_indices,
    encode_state,
    select_greedy_action,
)


# 配置 WINDOW_SIZE，统一约束后续计算使用的规则参数。
WINDOW_SIZE = 720
WORLD_MARGIN = 20
# 配置 PIXELS_PER_UNIT，统一约束后续计算使用的规则参数。
PIXELS_PER_UNIT = WINDOW_SIZE / (
    BossDodgeEnv.ARENA_WIDTH + 2 * WORLD_MARGIN
)


# 定义 Visualizer，组织相关状态和操作接口。
class Visualizer:
    """Drive and render one environment episode."""

    # 定义 __init__，集中处理这一阶段的输入与状态变化。
    def __init__(
        self,
        *,
        q_data: dict[str, Any] | None,
        seed: int,
        delay_ms: int,
        manual: bool,
    ) -> None:
        # 计算并保存 initial_boss_hp，供后续逻辑直接复用。
        initial_boss_hp = (
            BossDodgeEnv.INITIAL_BOSS_HP
            if q_data is None
            else int(q_data.get("boss_hp", BossDodgeEnv.INITIAL_BOSS_HP))
        )
        # 更新 self.env，使实例状态与当前帧保持一致。
        self.env = BossDodgeEnv(seed=seed, initial_boss_hp=initial_boss_hp)
        self.observation, _ = self.env.reset(seed=seed)
        # 更新 self.q_values，使实例状态与当前帧保持一致。
        self.q_values = {} if q_data is None else q_data["q_values"]
        self.rng = random.Random(seed)
        # 更新 self._policy_action_remaining，使实例状态与当前帧保持一致。
        self._policy_action_remaining = 0
        self._current_policy_action = self.env.ACTIONS.index("wait")
        # 更新 self.delay_ms，使实例状态与当前帧保持一致。
        self.delay_ms = delay_ms
        self.manual = manual
        # 更新 self._held_keys，使实例状态与当前帧保持一致。
        self._held_keys: dict[str, int] = {}
        self._pending_actions: deque[int] = deque()
        # 更新 self._release_jobs，使实例状态与当前帧保持一致。
        self._release_jobs: dict[str, str] = {}
        self.finished = False

        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if q_data is not None and tuple(q_data.get("actions", ())) != self.env.ACTIONS:
            raise ValueError("checkpoint actions do not match the current environment")

        # 更新 self.screen，使实例状态与当前帧保持一致。
        self.screen = turtle.Screen()
        self.screen.title("Boss Dodge RL")
        # 调用 self.screen.setup，推进当前处理步骤。
        self.screen.setup(WINDOW_SIZE, WINDOW_SIZE)
        self.screen.setworldcoordinates(
            -WORLD_MARGIN,
            -WORLD_MARGIN,
            self.env.ARENA_WIDTH + WORLD_MARGIN,
            self.env.ARENA_HEIGHT + WORLD_MARGIN,
        )
        # 调用 self.screen.bgcolor，推进当前处理步骤。
        self.screen.bgcolor("#f3f4f6")
        self.screen.tracer(0)

        # Turtle uses creation order as drawing order. Build the arena first,
        # then effects and enemies, and create the player last in the scene.
        self._draw_arena()
        self.boss_sprite = self._sprite(
            self.env.BOSS_WIDTH,
            self.env.BOSS_HEIGHT,
            "#7c3aed",
        )
        # 更新 self.attack_sprite，使实例状态与当前帧保持一致。
        self.attack_sprite = self._sprite(
            self.env.BOSS_ATTACK_WIDTH,
            self.env.BOSS_ATTACK_HEIGHT,
            "#f59e0b",
        )
        # 调用 self.attack_sprite.hideturtle，推进当前处理步骤。
        self.attack_sprite.hideturtle()
        self.sword_sprite = self._sprite(
            self.env.SWORD_WIDTH,
            self.env.SWORD_HEIGHT,
            "#06b6d4",
        )
        # 调用 self.sword_sprite.hideturtle，推进当前处理步骤。
        self.sword_sprite.hideturtle()
        self.player_sprite = self._sprite(
            self.env.PLAYER_WIDTH,
            self.env.PLAYER_HEIGHT,
            "#2563eb",
        )
        # 更新 self.hud，使实例状态与当前帧保持一致。
        self.hud = turtle.Turtle(visible=False)
        self.hud.penup()
        # 调用 self.hud.color，推进当前处理步骤。
        self.hud.color("#111827")
        self.banner = turtle.Turtle(visible=False)
        # 调用 self.banner.penup，推进当前处理步骤。
        self.banner.penup()
        self.banner.color("#111827")

        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.manual:
            self._bind_controls()
        # 调用 self._render，推进当前处理步骤。
        self._render()

    def _sprite(self, width: float, height: float, color: str) -> turtle.Turtle:
        # 计算并保存 sprite，供后续逻辑直接复用。
        sprite = turtle.Turtle(shape="square", visible=False)
        sprite.penup()
        # 调用 sprite.color，推进当前处理步骤。
        sprite.color(color)
        sprite.shapesize(
            stretch_wid=height * PIXELS_PER_UNIT / 20,
            stretch_len=width * PIXELS_PER_UNIT / 20,
        )
        # 调用 sprite.showturtle，推进当前处理步骤。
        sprite.showturtle()
        return sprite

    # 定义 _draw_arena，集中处理这一阶段的输入与状态变化。
    def _draw_arena(self) -> None:
        pen = turtle.Turtle(visible=False)
        # 调用 pen.speed，推进当前处理步骤。
        pen.speed(0)
        pen.color("#374151")
        # 调用 pen.pensize，推进当前处理步骤。
        pen.pensize(2)
        pen.penup()
        # 调用 pen.goto，推进当前处理步骤。
        pen.goto(0, 0)
        pen.pendown()
        # 逐项处理当前序列，并累积这一轮所需的结果。
        for point in (
            (self.env.ARENA_WIDTH, 0),
            (self.env.ARENA_WIDTH, self.env.ARENA_HEIGHT),
            (0, self.env.ARENA_HEIGHT),
            (0, 0),
        ):
            # 调用 pen.goto，推进当前处理步骤。
            pen.goto(*point)

    def _bind_controls(self) -> None:
        # 计算并保存 bindings，供后续逻辑直接复用。
        bindings = {
            "Left": "left",
            "Right": "right",
            "d": "dash",
            "a": "attack",
            "w": "wait",
            "space": "jump",
        }
        # 逐项处理当前序列，并累积这一轮所需的结果。
        for key, action_name in bindings.items():
            action = self.env.ACTIONS.index(action_name)
            # 调用 self.screen.onkeypress，推进当前处理步骤。
            self.screen.onkeypress(
                lambda selected_key=key, selected=action: self._press_key(
                    selected_key,
                    selected,
                ),
                key,
            )
            # 调用 self.screen.onkeyrelease，推进当前处理步骤。
            self.screen.onkeyrelease(
                lambda selected_key=key: self._release_key(selected_key),
                key,
            )
        # 调用 self.screen.listen，推进当前处理步骤。
        self.screen.listen()

    def _press_key(self, key: str, action: int) -> None:
        # 计算并保存 release_job，供后续逻辑直接复用。
        release_job = self._release_jobs.pop(key, None)
        if release_job is not None:
            # 调用 相关逻辑.after_cancel，推进当前处理步骤。
            self.screen.getcanvas().after_cancel(release_job)
        if key in self._held_keys:
            # 返回已经整理好的结果，供上层流程继续使用。
            return
        self._held_keys[key] = action
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.env.ACTIONS[action] not in ("left", "right", "wait"):
            self._pending_actions.append(action)

    # 定义 _release_key，集中处理这一阶段的输入与状态变化。
    def _release_key(self, key: str) -> None:
        # Tk may emit release/press pairs while a key auto-repeats. Delay the
        # release briefly so the following press can cancel the false release.
        old_job = self._release_jobs.pop(key, None)
        if old_job is not None:
            # 调用 相关逻辑.after_cancel，推进当前处理步骤。
            self.screen.getcanvas().after_cancel(old_job)
        job = self.screen.getcanvas().after(20, lambda: self._finish_release(key))
        # 计算并保存 当前状态，供后续逻辑直接复用。
        self._release_jobs[key] = job

    def _finish_release(self, key: str) -> None:
        # 调用 self._release_jobs.pop，推进当前处理步骤。
        self._release_jobs.pop(key, None)
        self._held_keys.pop(key, None)

    # 定义 _manual_action，集中处理这一阶段的输入与状态变化。
    def _manual_action(self) -> int:
        if self._pending_actions:
            # 返回已经整理好的结果，供上层流程继续使用。
            return self._pending_actions.popleft()
        for action in reversed(self._held_keys.values()):
            # 根据当前条件选择对应分支，保持状态转换符合规则。
            if self.env.ACTIONS[action] in ("left", "right", "wait"):
                return action
        # 返回已经整理好的结果，供上层流程继续使用。
        return self.env.ACTIONS.index("wait")

    def _policy_action(self) -> int:
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self._policy_action_remaining == 0:
            state = encode_state(self.observation)
            # 计算并保存 key，供后续逻辑直接复用。
            key = "|".join(map(str, state))
            values = self.q_values.get(key, [0.0] * len(self.env.ACTIONS))
            # 更新 self._current_policy_action，使实例状态与当前帧保持一致。
            self._current_policy_action = select_greedy_action(
                values,
                self.rng,
                available_action_indices(self.env),
            )
            # 更新 self._policy_action_remaining，使实例状态与当前帧保持一致。
            self._policy_action_remaining = ACTION_REPEAT
        self._policy_action_remaining -= 1
        # 返回已经整理好的结果，供上层流程继续使用。
        return self._current_policy_action

    def _tick(self) -> None:
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.finished:
            return
        # 计算并保存 action，供后续逻辑直接复用。
        action = self._manual_action() if self.manual else self._policy_action()
        self.observation, _, terminated, truncated, info = self.env.step(action)
        # 调用 self._render，推进当前处理步骤。
        self._render(action_name=self.env.ACTIONS[action])
        if terminated or truncated:
            # 更新 self.finished，使实例状态与当前帧保持一致。
            self.finished = True
            result = "Victory" if info["won"] else "Defeat"
            # 调用 self.banner.goto，推进当前处理步骤。
            self.banner.goto(self.env.ARENA_WIDTH / 2, self.env.ARENA_HEIGHT / 2)
            self.banner.write(result, align="center", font=("Arial", 28, "bold"))
            # 调用 self.screen.update，推进当前处理步骤。
            self.screen.update()
            return
        # 调用 self.screen.ontimer，推进当前处理步骤。
        self.screen.ontimer(self._tick, self.delay_ms)

    def _render(self, action_name: str = "wait") -> None:
        # 调用 self._place，推进当前处理步骤。
        self._place(self.player_sprite, self.env.player_hitbox)
        self._place(self.boss_sprite, self.env.boss_hitbox)

        # 计算并保存 attack_box，供后续逻辑直接复用。
        attack_box = self.env.attack_hitbox
        if attack_box is None:
            # 调用 self.attack_sprite.hideturtle，推进当前处理步骤。
            self.attack_sprite.hideturtle()
        else:
            # 调用 self.attack_sprite.color，推进当前处理步骤。
            self.attack_sprite.color(
                "#dc2626"
                if self.env.attack_phase == self.env.ATTACK_ACTIVE
                else "#f59e0b"
            )
            # 调用 self._place，推进当前处理步骤。
            self._place(self.attack_sprite, attack_box)
            self.attack_sprite.showturtle()

        # 计算并保存 sword_box，供后续逻辑直接复用。
        sword_box = self.env.sword_hitbox
        if sword_box is None:
            # 调用 self.sword_sprite.hideturtle，推进当前处理步骤。
            self.sword_sprite.hideturtle()
        else:
            # 调用 self._place，推进当前处理步骤。
            self._place(self.sword_sprite, sword_box)
            self.sword_sprite.showturtle()

        # 调用 self.hud.clear，推进当前处理步骤。
        self.hud.clear()
        self.hud.goto(0, self.env.ARENA_HEIGHT + 7)
        # 计算并保存 mode，供后续逻辑直接复用。
        mode = "MANUAL" if self.manual else "Q POLICY"
        self.hud.write(
            f"{mode}   Player HP {self.env.player_hp}   Boss HP {self.env.boss_hp}"
            f"   Frame {self.env.steps}   Action {action_name}"
            f"   Recovery {self.env.player_attack_recovery_timer}"
            f"   Dash {self.env.player_dash_timer}"
            f"   Dash recovery {self.env.player_dash_recovery_timer}",
            align="left",
            font=("Arial", 10, "normal"),
        )
        # 调用 self.screen.update，推进当前处理步骤。
        self.screen.update()

    @staticmethod
    def _place(sprite: turtle.Turtle, box: Rect) -> None:
        # 调用 sprite.goto，推进当前处理步骤。
        sprite.goto(box.x + box.width / 2, box.y + box.height / 2)

    def run(self) -> None:
        # 调用 self.screen.ontimer，推进当前处理步骤。
        self.screen.ontimer(self._tick, self.delay_ms)
        self.screen.mainloop()


# 定义 load_checkpoint，集中处理这一阶段的输入与状态变化。
def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a JSON Q-table checkpoint."""
    # 在受控上下文中执行操作，确保资源和异常得到正确处理。
    with path.open(encoding="utf-8") as checkpoint_file:
        data = json.load(checkpoint_file)
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if "actions" not in data or "q_values" not in data:
        raise ValueError(f"invalid checkpoint: {path}")
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if data.get("state_encoding") != STATE_ENCODING:
        raise ValueError(f"checkpoint uses an outdated state encoding: {path}; train again")
    # 返回已经整理好的结果，供上层流程继续使用。
    return data


def main() -> None:
    # 计算并保存 parser，供后续逻辑直接复用。
    parser = argparse.ArgumentParser(description="Visualize the Boss Dodge environment")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/q_table.json"))
    # 调用 parser.add_argument，推进当前处理步骤。
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument(
        "--delay",
        type=int,
        default=round(1000 / BossDodgeEnv.FRAMES_PER_SECOND),
        help="milliseconds between steps",
    )
    # 调用 parser.add_argument，推进当前处理步骤。
    parser.add_argument("--manual", action="store_true", help="control the player with the keyboard")
    args = parser.parse_args()
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if args.delay < 1:
        parser.error("--delay must be at least 1")

    # 计算并保存 q_data，供后续逻辑直接复用。
    q_data = None
    if not args.manual:
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint not found: {args.checkpoint}; train first or use --manual")
        # 计算并保存 q_data，供后续逻辑直接复用。
        q_data = load_checkpoint(args.checkpoint)

    Visualizer(
        q_data=q_data,
        seed=args.seed,
        delay_ms=args.delay,
        manual=args.manual,
    ).run()


# 根据当前条件选择对应分支，保持状态转换符合规则。
if __name__ == "__main__":
    main()
