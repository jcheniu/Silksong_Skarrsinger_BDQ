"""A small two-dimensional boss arena for tabular RL experiments."""

from __future__ import annotations

# 导入本模块依赖的类型与运行时工具。
from collections import deque
from dataclasses import dataclass
# 导入本模块依赖的类型与运行时工具。
import random
from typing import Any

from .action_catalog import SIMULATOR_ACTION_NAMES


# 声明下方接口的调用方式与对象属性。
@dataclass(frozen=True)
class Rect:
    """Axis-aligned collision box using bottom-left coordinates."""

    # 记录 x 字段，构成该对象对外提供的完整状态。
    x: float
    y: float
    # 记录 width 字段，构成该对象对外提供的完整状态。
    width: float
    height: float

    # 定义 intersects，集中处理这一阶段的输入与状态变化。
    def intersects(self, other: "Rect") -> bool:
        """Return whether this box overlaps ``other`` with positive area."""
        # 返回已经整理好的结果，供上层流程继续使用。
        return (
            self.x < other.x + other.width
            and self.x + self.width > other.x
            and self.y < other.y + other.height
            and self.y + self.height > other.y
        )


# 声明下方接口的调用方式与对象属性。
@dataclass(frozen=True)
class Observation:
    """The complete state vector visible to the learning agent."""

    # 记录 player_x 字段，构成该对象对外提供的完整状态。
    player_x: float
    player_y: float
    # 记录 player_velocity_y 字段，构成该对象对外提供的完整状态。
    player_velocity_y: float
    player_facing: int
    # 记录 player_attack_recovery_timer 字段，构成该对象对外提供的完整状态。
    player_attack_recovery_timer: int
    player_dash_timer: int
    # 记录 player_dash_direction 字段，构成该对象对外提供的完整状态。
    player_dash_direction: int
    player_dash_recovery_timer: int
    # 记录 boss_x 字段，构成该对象对外提供的完整状态。
    boss_x: float
    boss_y: float
    # 记录 boss_velocity_x 字段，构成该对象对外提供的完整状态。
    boss_velocity_x: int
    attack_x: float
    # 记录 attack_y 字段，构成该对象对外提供的完整状态。
    attack_y: float
    attack_phase: int
    # 记录 attack_timer 字段，构成该对象对外提供的完整状态。
    attack_timer: int
    player_hp: int
    # 记录 boss_hp 字段，构成该对象对外提供的完整状态。
    boss_hp: int
    invulnerable_timer: int

    # 定义 as_tuple，集中处理这一阶段的输入与状态变化。
    def as_tuple(self) -> tuple[float | int, ...]:
        return (
            self.player_x,
            self.player_y,
            self.player_velocity_y,
            self.player_facing,
            self.player_attack_recovery_timer,
            self.player_dash_timer,
            self.player_dash_direction,
            self.player_dash_recovery_timer,
            self.boss_x,
            self.boss_y,
            self.boss_velocity_x,
            self.attack_x,
            self.attack_y,
            self.attack_phase,
            self.attack_timer,
            self.player_hp,
            self.boss_hp,
            self.invulnerable_timer,
        )


# 定义 BossDodgeEnv，组织相关状态和操作接口。
class BossDodgeEnv:
    """A 300 by 300 arena with frame-based movement and AABB hitboxes."""

    # 配置 ARENA_WIDTH，统一约束后续计算使用的规则参数。
    ARENA_WIDTH = 300
    ARENA_HEIGHT = 300

    # 配置 PLAYER_WIDTH，统一约束后续计算使用的规则参数。
    PLAYER_WIDTH = 10
    PLAYER_HEIGHT = 20
    # 配置 BOSS_WIDTH，统一约束后续计算使用的规则参数。
    BOSS_WIDTH = 30
    BOSS_HEIGHT = 50
    # 配置 SWORD_WIDTH，统一约束后续计算使用的规则参数。
    SWORD_WIDTH = PLAYER_WIDTH * 2
    SWORD_HEIGHT = PLAYER_HEIGHT
    # 配置 BOSS_ATTACK_WIDTH，统一约束后续计算使用的规则参数。
    BOSS_ATTACK_WIDTH = 36
    BOSS_ATTACK_HEIGHT = 12

    # 配置 BOSS_FLOAT_Y，统一约束后续计算使用的规则参数。
    BOSS_FLOAT_Y = 45
    PLAYER_SPEED = 3
    # 配置 DASH_SPEED，统一约束后续计算使用的规则参数。
    DASH_SPEED = 9
    DASH_FRAMES = 5
    # 配置 BOSS_SPEED，统一约束后续计算使用的规则参数。
    BOSS_SPEED = 1
    BOSS_TELEPORT_DISTANCE = 80
    # 配置 POSITION_BIN，统一约束后续计算使用的规则参数。
    POSITION_BIN = 10
    FRAMES_PER_SECOND = 60
    # 配置 JUMP_SPEED，统一约束后续计算使用的规则参数。
    JUMP_SPEED = 9.0
    GRAVITY = 0.65
    # 配置 TERMINAL_VELOCITY，统一约束后续计算使用的规则参数。
    TERMINAL_VELOCITY = -12.0
    DASH_INVULNERABILITY_FRAMES = DASH_FRAMES
    # 配置 HURT_INVULNERABILITY_FRAMES，统一约束后续计算使用的规则参数。
    HURT_INVULNERABILITY_FRAMES = 8
    ATTACK_WARNING_MIN_FRAMES = 12
    # 配置 ATTACK_WARNING_MAX_FRAMES，统一约束后续计算使用的规则参数。
    ATTACK_WARNING_MAX_FRAMES = 18
    ATTACK_ACTIVE_FRAMES = 30
    # 配置 BOSS_ATTACK_RECOVERY_FRAMES，统一约束后续计算使用的规则参数。
    BOSS_ATTACK_RECOVERY_FRAMES = 10
    PLAYER_ATTACK_RECOVERY_FRAMES = FRAMES_PER_SECOND
    # 配置 PLAYER_DASH_RECOVERY_FRAMES，统一约束后续计算使用的规则参数。
    PLAYER_DASH_RECOVERY_FRAMES = FRAMES_PER_SECOND
    ATTACK_TRACK_MIN_FRAMES = 2
    # 配置 ATTACK_TRACK_MAX_FRAMES，统一约束后续计算使用的规则参数。
    ATTACK_TRACK_MAX_FRAMES = 12
    INITIAL_PLAYER_HP = 3
    # 配置 INITIAL_BOSS_HP，统一约束后续计算使用的规则参数。
    INITIAL_BOSS_HP = 6

    DODGE_REWARD = 0.2
    # 配置 ATTACK_RANGE_REWARD，统一约束后续计算使用的规则参数。
    ATTACK_RANGE_REWARD = 0.25
    BOSS_HIT_REWARD = 3.0
    # 配置 PLAYER_HURT_PENALTY，统一约束后续计算使用的规则参数。
    PLAYER_HURT_PENALTY = -3.0
    STEP_PENALTY = -0.002
    # 配置 VICTORY_REWARD，统一约束后续计算使用的规则参数。
    VICTORY_REWARD = 10.0
    PROGRESS_PENALTY_INTERVAL = 200
    # 配置 PROGRESS_PENALTY_SCALE，统一约束后续计算使用的规则参数。
    PROGRESS_PENALTY_SCALE = 1.0
    SPIKE_ESCAPE_GRACE_FRAMES = 8
    # 配置 SPIKE_ESCAPE_TIMEOUT_PENALTY，统一约束后续计算使用的规则参数。
    SPIKE_ESCAPE_TIMEOUT_PENALTY = -0.2

    ATTACK_IDLE = 0
    # 配置 ATTACK_WARNING，统一约束后续计算使用的规则参数。
    ATTACK_WARNING = 1
    ATTACK_ACTIVE = 2
    # 配置 ATTACK_RECOVERY，统一约束后续计算使用的规则参数。
    ATTACK_RECOVERY = 3
    ATTACK_PHASE_NAMES = ("idle", "warning", "active", "recovery")

    # Original action indices remain stable; jump is the appended action.
    ACTIONS = SIMULATOR_ACTION_NAMES

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int | None = None,
        initial_boss_hp: int | None = None,
    ) -> None:
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if initial_boss_hp is None:
            initial_boss_hp = self.INITIAL_BOSS_HP
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if not 1 <= initial_boss_hp <= self.INITIAL_BOSS_HP:
            raise ValueError(
                f"initial_boss_hp must be in [1, {self.INITIAL_BOSS_HP}]"
            )
        # 更新 self.rng，使实例状态与当前帧保持一致。
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        # 更新 self.initial_boss_hp，使实例状态与当前帧保持一致。
        self.initial_boss_hp = initial_boss_hp
        self.reset()

    # 声明下方接口的调用方式与对象属性。
    @property
    def is_grounded(self) -> bool:
        # 返回已经整理好的结果，供上层流程继续使用。
        return self.player_y == 0 and self.player_velocity_y == 0

    @property
    def player_hitbox(self) -> Rect:
        # 返回已经整理好的结果，供上层流程继续使用。
        return Rect(self.player_x, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

    @property
    def boss_hitbox(self) -> Rect:
        # 返回已经整理好的结果，供上层流程继续使用。
        return Rect(self.boss_x, self.boss_y, self.BOSS_WIDTH, self.BOSS_HEIGHT)

    @property
    def boss_reward_hitbox(self) -> Rect:
        """The first-approach reward zone around both sides of the Boss."""
        # 返回已经整理好的结果，供上层流程继续使用。
        return Rect(
            self.boss_x - self.SWORD_WIDTH,
            self.boss_y,
            self.BOSS_WIDTH + 2 * self.SWORD_WIDTH,
            self.BOSS_HEIGHT,
        )

    # 声明下方接口的调用方式与对象属性。
    @property
    def sword_hitbox(self) -> Rect | None:
        """The sword light emitted during the current attack frame."""
        # 返回已经整理好的结果，供上层流程继续使用。
        return self._sword_hitbox

    @property
    def attack_hitbox(self) -> Rect | None:
        """The Boss's currently telegraphed ground-spike box."""
        # 返回已经整理好的结果，供上层流程继续使用。
        return self._attack_hitbox

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[tuple[float | int, ...], dict[str, Any]]:
        """Start a new battle and return its initial observation."""
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if seed is not None:
            self.rng.seed(seed)

        # 更新 self.player_x，使实例状态与当前帧保持一致。
        self.player_x = 80.0
        self.player_y = 0.0
        # 更新 self.player_velocity_y，使实例状态与当前帧保持一致。
        self.player_velocity_y = 0.0
        self.player_facing = 1
        # 更新 self.player_attack_recovery_timer，使实例状态与当前帧保持一致。
        self.player_attack_recovery_timer = 0
        self.player_dash_timer = 0
        # 更新 self.player_dash_direction，使实例状态与当前帧保持一致。
        self.player_dash_direction = 1
        self.player_dash_recovery_timer = 0
        # 更新 self.boss_x，使实例状态与当前帧保持一致。
        self.boss_x = 190.0
        self.boss_y = float(self.BOSS_FLOAT_Y)
        # 更新 self.boss_velocity_x，使实例状态与当前帧保持一致。
        self.boss_velocity_x = self.rng.choice((-self.BOSS_SPEED, self.BOSS_SPEED))
        self.attack_phase = self.ATTACK_IDLE
        # 更新 self.attack_timer，使实例状态与当前帧保持一致。
        self.attack_timer = 0
        self.attack_target_age = 0
        # 更新 self._attack_has_hit，使实例状态与当前帧保持一致。
        self._attack_has_hit = False
        self._attack_hurt_player = False
        # 更新 self._attack_hitbox，使实例状态与当前帧保持一致。
        self._attack_hitbox: Rect | None = None
        self._sword_hitbox: Rect | None = None
        # 更新 self._player_x_history，使实例状态与当前帧保持一致。
        self._player_x_history: deque[float] = deque(
            [self.player_x],
            maxlen=self.ATTACK_TRACK_MAX_FRAMES + 1,
        )
        # 更新 self.player_hp，使实例状态与当前帧保持一致。
        self.player_hp = self.INITIAL_PLAYER_HP
        self.boss_hp = self.initial_boss_hp
        # 更新 self.invulnerable_timer，使实例状态与当前帧保持一致。
        self.invulnerable_timer = 0
        self._last_boss_hit = False
        # 更新 self._last_boss_teleported，使实例状态与当前帧保持一致。
        self._last_boss_teleported = False
        self._last_spike_dodged = False
        # 更新 self._last_spike_escape_timeout，使实例状态与当前帧保持一致。
        self._last_spike_escape_timeout = False
        self._last_entered_attack_range = False
        # 更新 self._last_progress_penalty，使实例状态与当前帧保持一致。
        self._last_progress_penalty = 0.0
        self._entered_attack_range = False
        # 更新 self.steps，使实例状态与当前帧保持一致。
        self.steps = 0
        return self._observation(), self._info()

    # 定义 step，集中处理这一阶段的输入与状态变化。
    def step(
        self,
        action: int,
    ) -> tuple[tuple[float | int, ...], float, bool, bool, dict[str, Any]]:
        """Advance one frame and return the standard RL transition tuple."""
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError(f"action must be in [0, {len(self.ACTIONS) - 1}]")

        # 更新 self.steps，使实例状态与当前帧保持一致。
        self.steps += 1
        reward = self.STEP_PENALTY
        # 更新 self._sword_hitbox，使实例状态与当前帧保持一致。
        self._sword_hitbox = None
        self._last_boss_hit = False
        # 更新 self._last_boss_teleported，使实例状态与当前帧保持一致。
        self._last_boss_teleported = False
        self._last_spike_dodged = False
        # 更新 self._last_spike_escape_timeout，使实例状态与当前帧保持一致。
        self._last_spike_escape_timeout = False
        self._last_entered_attack_range = False
        # 更新 self._last_progress_penalty，使实例状态与当前帧保持一致。
        self._last_progress_penalty = 0.0
        if self.invulnerable_timer > 0:
            # 更新 self.invulnerable_timer，使实例状态与当前帧保持一致。
            self.invulnerable_timer -= 1

        requested_name = self.ACTIONS[action]
        # 计算并保存 attack_recovering，供后续逻辑直接复用。
        attack_recovering = self.player_attack_recovery_timer > 0
        dash_recovering = self.player_dash_recovery_timer > 0
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if attack_recovering:
            self.player_attack_recovery_timer -= 1
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if dash_recovering:
            self.player_dash_recovery_timer -= 1
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.player_dash_timer > 0:
            name = "dash_active"
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        elif attack_recovering and requested_name == "attack":
            name = "wait"
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        elif dash_recovering and requested_name == "dash":
            name = "wait"
        # 处理前述条件未覆盖的其余情况。
        else:
            name = requested_name
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if name == "left":
            self.player_x -= self.PLAYER_SPEED
            # 更新 self.player_facing，使实例状态与当前帧保持一致。
            self.player_facing = -1
        elif name == "right":
            # 更新 self.player_x，使实例状态与当前帧保持一致。
            self.player_x += self.PLAYER_SPEED
            self.player_facing = 1
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        elif name == "dash":
            self.player_dash_timer = self.DASH_FRAMES
            # 更新 self.player_dash_direction，使实例状态与当前帧保持一致。
            self.player_dash_direction = self.player_facing
            self.invulnerable_timer = self.DASH_INVULNERABILITY_FRAMES
            # 更新 self.player_x，使实例状态与当前帧保持一致。
            self.player_x += self.player_dash_direction * self.DASH_SPEED
            self.player_dash_timer -= 1
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        elif name == "dash_active":
            self.player_x += self.player_dash_direction * self.DASH_SPEED
            # 更新 self.player_dash_timer，使实例状态与当前帧保持一致。
            self.player_dash_timer -= 1
            if self.player_dash_timer == 0:
                # 更新 self.player_dash_recovery_timer，使实例状态与当前帧保持一致。
                self.player_dash_recovery_timer = self.PLAYER_DASH_RECOVERY_FRAMES
        elif name == "jump" and self.is_grounded:
            # 更新 self.player_velocity_y，使实例状态与当前帧保持一致。
            self.player_velocity_y = self.JUMP_SPEED

        self.player_x = self._clamp(
            self.player_x,
            0,
            self.ARENA_WIDTH - self.PLAYER_WIDTH,
        )
        # Recovery restricts actions, but never suspends vertical physics.
        self._apply_gravity()
        self._move_boss()

        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if not self._entered_attack_range and self.player_hitbox.intersects(
            self.boss_reward_hitbox
        ):
            # 更新 self._entered_attack_range，使实例状态与当前帧保持一致。
            self._entered_attack_range = True
            self._last_entered_attack_range = True
            # 计算并保存 reward，供后续逻辑直接复用。
            reward += self.ATTACK_RANGE_REWARD

        if name == "attack":
            # 更新 self.player_attack_recovery_timer，使实例状态与当前帧保持一致。
            self.player_attack_recovery_timer = self.PLAYER_ATTACK_RECOVERY_FRAMES
            self._sword_hitbox = self._make_sword_hitbox()
            # 根据当前条件选择对应分支，保持状态转换符合规则。
            if self._sword_hitbox.intersects(self.boss_hitbox):
                self.boss_hp -= 1
                # 更新 self._last_boss_hit，使实例状态与当前帧保持一致。
                self._last_boss_hit = True
                self._teleport_boss_away()
                # 计算并保存 reward，供后续逻辑直接复用。
                reward += self.BOSS_HIT_REWARD

        self._player_x_history.append(self.player_x)
        # 计算并保存 spike_hit、attack_finished、spike_escape_timeout，供后续逻辑直接复用。
        spike_hit, attack_finished, spike_escape_timeout = (
            self._advance_boss_attack()
        )
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if spike_escape_timeout:
            self._last_spike_escape_timeout = True
            # 计算并保存 reward，供后续逻辑直接复用。
            reward += self.SPIKE_ESCAPE_TIMEOUT_PENALTY
        body_hit = self.boss_hp > 0 and self.player_hitbox.intersects(self.boss_hitbox)
        # 计算并保存 player_hurt，供后续逻辑直接复用。
        player_hurt = (spike_hit or body_hit) and self._hurt_player()
        if player_hurt:
            # 更新 self._attack_hurt_player，使实例状态与当前帧保持一致。
            self._attack_hurt_player = self._attack_hurt_player or spike_hit
            reward += self.PLAYER_HURT_PENALTY
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if attack_finished and not self._attack_hurt_player:
            self._last_spike_dodged = True
            # 计算并保存 reward，供后续逻辑直接复用。
            reward += self.DODGE_REWARD

        terminated = self.player_hp <= 0 or self.boss_hp <= 0
        # 计算并保存 truncated，供后续逻辑直接复用。
        truncated = self.max_steps is not None and self.steps >= self.max_steps and not terminated
        if (
            not terminated
            and self.steps % self.PROGRESS_PENALTY_INTERVAL == 0
        ):
            # 更新 self._last_progress_penalty，使实例状态与当前帧保持一致。
            self._last_progress_penalty = (
                self.PROGRESS_PENALTY_SCALE
                * self.boss_hp
                / self.INITIAL_BOSS_HP
            )
            # 计算并保存 reward，供后续逻辑直接复用。
            reward -= self._last_progress_penalty
        if self.boss_hp <= 0:
            # 计算并保存 reward，供后续逻辑直接复用。
            reward += self.VICTORY_REWARD
        return self._observation(), reward, terminated, truncated, self._info()

    # 定义 _apply_gravity，集中处理这一阶段的输入与状态变化。
    def _apply_gravity(self) -> None:
        """Integrate a smooth parabolic jump with fractional velocity."""
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.is_grounded:
            return

        # 更新 self.player_y，使实例状态与当前帧保持一致。
        self.player_y += self.player_velocity_y
        self.player_velocity_y = max(
            self.TERMINAL_VELOCITY,
            self.player_velocity_y - self.GRAVITY,
        )

        # 计算并保存 ceiling，供后续逻辑直接复用。
        ceiling = self.ARENA_HEIGHT - self.PLAYER_HEIGHT
        if self.player_y >= ceiling:
            # 更新 self.player_y，使实例状态与当前帧保持一致。
            self.player_y = float(ceiling)
            self.player_velocity_y = min(0.0, self.player_velocity_y)
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        elif self.player_y <= 0:
            self.player_y = 0.0
            # 更新 self.player_velocity_y，使实例状态与当前帧保持一致。
            self.player_velocity_y = 0.0

    def _move_boss(self) -> None:
        """Move the floating Boss by exactly one grid unit each frame."""
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.rng.random() < 0.04:
            self.boss_velocity_x *= -1
        # 更新 self.boss_x，使实例状态与当前帧保持一致。
        self.boss_x += self.boss_velocity_x
        maximum = self.ARENA_WIDTH - self.BOSS_WIDTH
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.boss_x <= 0:
            self.boss_x = 0.0
            # 更新 self.boss_velocity_x，使实例状态与当前帧保持一致。
            self.boss_velocity_x = self.BOSS_SPEED
        elif self.boss_x >= maximum:
            # 更新 self.boss_x，使实例状态与当前帧保持一致。
            self.boss_x = float(maximum)
            self.boss_velocity_x = -self.BOSS_SPEED

    # 定义 _teleport_boss_away，集中处理这一阶段的输入与状态变化。
    def _teleport_boss_away(self) -> None:
        """Instantly move the Boss along X in the direction away from the player."""
        # 计算并保存 player_center，供后续逻辑直接复用。
        player_center = self.player_x + self.PLAYER_WIDTH / 2
        boss_center = self.boss_x + self.BOSS_WIDTH / 2
        # 计算并保存 direction，供后续逻辑直接复用。
        direction = 1 if boss_center >= player_center else -1
        maximum = self.ARENA_WIDTH - self.BOSS_WIDTH
        # 更新 self.boss_x，使实例状态与当前帧保持一致。
        self.boss_x = self._clamp(
            self.boss_x + direction * self.BOSS_TELEPORT_DISTANCE,
            0,
            maximum,
        )
        # 更新 self.boss_velocity_x，使实例状态与当前帧保持一致。
        self.boss_velocity_x = direction * self.BOSS_SPEED
        self._last_boss_teleported = True
        # 更新 self._entered_attack_range，使实例状态与当前帧保持一致。
        self._entered_attack_range = False

    def _advance_boss_attack(self) -> tuple[bool, bool, bool]:
        """Advance warning, active, and recovery phases by one frame."""
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.attack_phase == self.ATTACK_IDLE:
            self._summon_boss_attack()
            # 返回已经整理好的结果，供上层流程继续使用。
            return False, False, False

        if self.attack_phase == self.ATTACK_WARNING:
            # 更新 self.attack_warning_elapsed，使实例状态与当前帧保持一致。
            self.attack_warning_elapsed += 1
            escape_timeout = (
                self.attack_warning_elapsed == self.SPIKE_ESCAPE_GRACE_FRAMES
                and self._attack_hitbox is not None
                and self._attack_hitbox.intersects(self.player_hitbox)
            )
            # 更新 self.attack_timer，使实例状态与当前帧保持一致。
            self.attack_timer -= 1
            if self.attack_timer == 0:
                # 更新 self.attack_phase，使实例状态与当前帧保持一致。
                self.attack_phase = self.ATTACK_ACTIVE
                self.attack_timer = self.ATTACK_ACTIVE_FRAMES
            # 返回已经整理好的结果，供上层流程继续使用。
            return False, False, escape_timeout

        if self.attack_phase == self.ATTACK_ACTIVE:
            # 计算并保存 hit，供后续逻辑直接复用。
            hit = (
                not self._attack_has_hit
                and self._attack_hitbox is not None
                and self._attack_hitbox.intersects(self.player_hitbox)
            )
            # 更新 self._attack_has_hit，使实例状态与当前帧保持一致。
            self._attack_has_hit = self._attack_has_hit or hit
            self.attack_timer -= 1
            # 根据当前条件选择对应分支，保持状态转换符合规则。
            if self.attack_timer == 0:
                self._attack_hitbox = None
                # 更新 self.attack_phase，使实例状态与当前帧保持一致。
                self.attack_phase = self.ATTACK_RECOVERY
                self.attack_timer = self.BOSS_ATTACK_RECOVERY_FRAMES
                # 返回已经整理好的结果，供上层流程继续使用。
                return hit, True, False
            return hit, False, False

        # 更新 self.attack_timer，使实例状态与当前帧保持一致。
        self.attack_timer -= 1
        if self.attack_timer == 0:
            # 更新 self.attack_phase，使实例状态与当前帧保持一致。
            self.attack_phase = self.ATTACK_IDLE
        return False, False, False

    # 定义 _summon_boss_attack，集中处理这一阶段的输入与状态变化。
    def _summon_boss_attack(self) -> None:
        """Target the player's X position from a random number of past frames."""
        # 计算并保存 available_age，供后续逻辑直接复用。
        available_age = min(
            self.ATTACK_TRACK_MAX_FRAMES,
            len(self._player_x_history) - 1,
        )
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if available_age < self.ATTACK_TRACK_MIN_FRAMES:
            lookback = max(1, available_age)
        # 处理前述条件未覆盖的其余情况。
        else:
            lookback = self.rng.randint(self.ATTACK_TRACK_MIN_FRAMES, available_age)
        # 计算并保存 target_x，供后续逻辑直接复用。
        target_x = self._player_x_history[-(lookback + 1)]
        target_center = target_x + self.PLAYER_WIDTH / 2
        # 计算并保存 attack_x，供后续逻辑直接复用。
        attack_x = target_center - self.BOSS_ATTACK_WIDTH / 2
        attack_x = self._clamp(
            attack_x,
            0,
            self.ARENA_WIDTH - self.BOSS_ATTACK_WIDTH,
        )
        # 更新 self._attack_hitbox，使实例状态与当前帧保持一致。
        self._attack_hitbox = Rect(
            attack_x,
            0,
            self.BOSS_ATTACK_WIDTH,
            self.BOSS_ATTACK_HEIGHT,
        )
        # 更新 self.attack_phase，使实例状态与当前帧保持一致。
        self.attack_phase = self.ATTACK_WARNING
        self._attack_has_hit = False
        # 更新 self._attack_hurt_player，使实例状态与当前帧保持一致。
        self._attack_hurt_player = False
        self.attack_warning_elapsed = 0
        # 更新 self.attack_target_age，使实例状态与当前帧保持一致。
        self.attack_target_age = lookback
        self.attack_timer = self.rng.randint(
            self.ATTACK_WARNING_MIN_FRAMES,
            self.ATTACK_WARNING_MAX_FRAMES,
        )

    # 定义 _make_sword_hitbox，集中处理这一阶段的输入与状态变化。
    def _make_sword_hitbox(self) -> Rect:
        """Create a sword box twice the player's width on the facing side."""
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.player_facing > 0:
            x = self.player_x + self.PLAYER_WIDTH
        # 处理前述条件未覆盖的其余情况。
        else:
            x = self.player_x - self.SWORD_WIDTH
        # 返回已经整理好的结果，供上层流程继续使用。
        return Rect(x, self.player_y, self.SWORD_WIDTH, self.SWORD_HEIGHT)

    def _hurt_player(self) -> bool:
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if self.invulnerable_timer > 0:
            return False
        # 更新 self.player_hp，使实例状态与当前帧保持一致。
        self.player_hp -= 1
        self.invulnerable_timer = self.HURT_INVULNERABILITY_FRAMES
        # 返回已经整理好的结果，供上层流程继续使用。
        return True

    def _info(self) -> dict[str, Any]:
        # 返回已经整理好的结果，供上层流程继续使用。
        return {
            "action_names": self.ACTIONS,
            "won": self.boss_hp <= 0,
            "damage_taken": self.INITIAL_PLAYER_HP - self.player_hp,
            "is_grounded": self.is_grounded,
            "player_facing": self.player_facing,
            "player_attack_recovery_timer": self.player_attack_recovery_timer,
            "player_dash_timer": self.player_dash_timer,
            "player_dash_direction": self.player_dash_direction,
            "player_dash_recovery_timer": self.player_dash_recovery_timer,
            "attack_target_age": self.attack_target_age,
            "attack_phase": self.ATTACK_PHASE_NAMES[self.attack_phase],
            "boss_hit": self._last_boss_hit,
            "boss_teleported": self._last_boss_teleported,
            "spike_dodged": self._last_spike_dodged,
            "spike_escape_timeout": self._last_spike_escape_timeout,
            "entered_attack_range": self._last_entered_attack_range,
            "progress_penalty": self._last_progress_penalty,
            "player_hitbox": self.player_hitbox,
            "boss_hitbox": self.boss_hitbox,
            "boss_reward_hitbox": self.boss_reward_hitbox,
            "sword_hitbox": self.sword_hitbox,
            "attack_hitbox": self.attack_hitbox,
        }

    # 定义 _observation，集中处理这一阶段的输入与状态变化。
    def _observation(self) -> tuple[float | int, ...]:
        attack_x = -1.0 if self._attack_hitbox is None else self._attack_hitbox.x
        # 计算并保存 attack_y，供后续逻辑直接复用。
        attack_y = -1.0 if self._attack_hitbox is None else self._attack_hitbox.y
        return Observation(
            self.player_x,
            self.player_y,
            self.player_velocity_y,
            self.player_facing,
            self.player_attack_recovery_timer,
            self.player_dash_timer,
            self.player_dash_direction,
            self.player_dash_recovery_timer,
            self.boss_x,
            self.boss_y,
            self.boss_velocity_x,
            attack_x,
            attack_y,
            self.attack_phase,
            self.attack_timer,
            self.player_hp,
            self.boss_hp,
            self.invulnerable_timer,
        ).as_tuple()

    # 声明下方接口的调用方式与对象属性。
    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        # 返回已经整理好的结果，供上层流程继续使用。
        return min(maximum, max(minimum, value))
