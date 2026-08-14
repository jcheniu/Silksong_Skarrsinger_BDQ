"""Reward events derived from consecutive live-game telemetry snapshots.

The values and reward intent mirror the simulator used by train_dqn.py. This
module contains no DQN, replay buffer, action selection, or training code.
Raw Boss HP is intentionally absent; the plugin supplies only cumulative
damage dealt, which is converted to a proportional reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .real_state import StateFrame, encode_snapshot


STEP_PENALTY = -0.002
ATTACK_RANGE_REWARD = 0.25
DAMAGE_REWARD_PER_HP = 0.05
ILLEGAL_ACTION_PENALTY = -1.0
PLAYER_DAMAGE_PENALTY_PER_HP = -3.0
# Compatibility alias for callers that imported the old fixed-event constant.
PLAYER_HURT_PENALTY = PLAYER_DAMAGE_PENALTY_PER_HP
DODGE_REWARD = 0.2
VICTORY_REWARD = 10.0
SILK_SPEND_PENALTY_PER_UNIT = -0.02


@dataclass(frozen=True)
class RewardConfig:
    """Real-world geometry used only for the one-time approach reward."""

    attack_range_x: float = 6.0
    attack_range_y: float = 5.0


@dataclass(frozen=True)
class RewardFrame:
    """Auditable reward breakdown for one telemetry time slice."""

    total: float
    step: float
    entered_attack_range: float
    damage_reward: float
    damage_deal: int
    player_hurt: float
    player_damage_taken: int
    dodge: float
    victory: float
    silk_spent: int
    silk_penalty: float
    player_health_lost: int
    attack_finished: str | None
    player_dead: bool
    boss_dead: bool
    terminated: bool


def _health(snapshot: Mapping[str, object]) -> int | None:
    value = snapshot.get("player_health")
    if not isinstance(value, Mapping):
        return None
    health = value.get("health")
    if isinstance(health, bool):
        return None
    try:
        return int(health)
    except (TypeError, ValueError):
        return None


def _silk(snapshot: Mapping[str, object]) -> int | None:
    resources = snapshot.get("player_resources")
    if not isinstance(resources, Mapping):
        return None
    silk = resources.get("silk")
    if isinstance(silk, bool):
        return None
    try:
        value = int(silk)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _in_attack_range(state: StateFrame, config: RewardConfig) -> bool:
    if state.player is None or state.boss is None:
        return False
    player_x = float(state.player.get("x", 0.0))
    player_y = float(state.player.get("y", 0.0))
    boss_x = float(state.boss.get("x", 0.0))
    boss_y = float(state.boss.get("y", 0.0))
    return (
        abs(boss_x - player_x) <= config.attack_range_x
        and abs(boss_y - player_y) <= config.attack_range_y
    )


class RewardTracker:
    """Convert an ordered snapshot stream into simulator-style rewards."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.reset()

    def reset(self) -> None:
        """Begin a new episode and clear all one-shot event latches."""

        self._previous_health: int | None = None
        self._previous_silk: int | None = None
        self._previous_reaction = "normal"
        self._previous_boss_damage_events: int | None = None
        self._previous_boss_damage_total: int | None = None
        self._entered_attack_range = False
        self._current_attack: str | None = None
        self._current_attack_hurt_player = False
        self._current_attack_became_active = False
        self._current_attack_observed_from_start = False
        self._previous_attack_phase = "idle"
        self._victory_awarded = False

    def step(self, snapshot: Mapping[str, object]) -> RewardFrame:
        """Score one snapshot relative to all earlier snapshots this episode."""

        state = encode_snapshot(snapshot)
        health = _health(snapshot)
        silk = _silk(snapshot)
        health_lost = (
            max(0, self._previous_health - health)
            if self._previous_health is not None and health is not None
            else 0
        )
        player_hurt = PLAYER_DAMAGE_PENALTY_PER_HP * health_lost
        silk_spent = (
            max(0, self._previous_silk - silk)
            if self._previous_silk is not None and silk is not None
            else 0
        )
        silk_penalty = SILK_SPEND_PENALTY_PER_UNIT * silk_spent

        entered_attack_range = 0.0
        if not self._entered_attack_range and _in_attack_range(state, self.config):
            self._entered_attack_range = True
            entered_attack_range = ATTACK_RANGE_REWARD

        damage_deal = 0
        raw_damage_total = snapshot.get("boss_damage_total")
        damage_total = (
            int(raw_damage_total)
            if isinstance(raw_damage_total, (int, float))
            and not isinstance(raw_damage_total, bool)
            and float(raw_damage_total).is_integer()
            and int(raw_damage_total) >= 0
            else None
        )
        if damage_total is not None:
            if (
                self._previous_boss_damage_total is not None
                and damage_total >= self._previous_boss_damage_total
            ):
                damage_deal = damage_total - self._previous_boss_damage_total
            self._previous_boss_damage_total = damage_total

        boss_hits = 0
        raw_damage_events = snapshot.get("boss_damage_events")
        damage_events = (
            int(raw_damage_events)
            if isinstance(raw_damage_events, (int, float))
            and not isinstance(raw_damage_events, bool)
            and float(raw_damage_events).is_integer()
            and int(raw_damage_events) >= 0
            else None
        )
        if damage_events is not None:
            if (
                self._previous_boss_damage_events is not None
                and damage_events >= self._previous_boss_damage_events
            ):
                boss_hits = damage_events - self._previous_boss_damage_events
            self._previous_boss_damage_events = damage_events
        elif (
            state.reaction in {"hit", "stunned"}
            and state.reaction != self._previous_reaction
        ):
            # Compatibility with telemetry recorded before boss damage events.
            boss_hits = 1
        if damage_total is None and boss_hits:
            # Compatibility with old telemetry: exact damage was unavailable.
            damage_deal = boss_hits
        damage_reward = DAMAGE_REWARD_PER_HP * damage_deal

        attack_finished: str | None = None
        dodge = 0.0
        current_attack = state.attack_type if state.attack_type != "none" else None
        restarted_same_attack = (
            self._current_attack is not None
            and current_attack == self._current_attack
            and state.attack_phase == "anticipation"
            and self._previous_attack_phase != "anticipation"
        )
        attack_changed = (
            self._current_attack is not None
            and current_attack != self._current_attack
        )
        if restarted_same_attack or attack_changed:
            attack_finished = self._current_attack
            if (
                self._current_attack_observed_from_start
                and self._current_attack_became_active
                and not self._current_attack_hurt_player
            ):
                dodge = DODGE_REWARD
            self._current_attack = None

        if self._current_attack is None and current_attack is not None:
            self._current_attack = current_attack
            self._current_attack_hurt_player = False
            self._current_attack_observed_from_start = (
                state.attack_phase == "anticipation"
            )
            self._current_attack_became_active = state.attack_phase == "active"
        elif self._current_attack is not None and state.attack_phase == "active":
            self._current_attack_became_active = True
        if self._current_attack is not None and health_lost > 0:
            self._current_attack_hurt_player = True

        player_dead = health is not None and health <= 0
        if state.control_state == "Hornet Dead":
            player_dead = True
        boss_dead = state.reaction == "dead"
        victory = 0.0
        if boss_dead and not self._victory_awarded:
            self._victory_awarded = True
            victory = VICTORY_REWARD

        step_reward = STEP_PENALTY
        total = (
            step_reward
            + entered_attack_range
            + damage_reward
            + player_hurt
            + dodge
            + victory
            + silk_penalty
        )
        self._previous_health = health
        self._previous_silk = silk
        self._previous_reaction = state.reaction
        self._previous_attack_phase = state.attack_phase
        return RewardFrame(
            total=total,
            step=step_reward,
            entered_attack_range=entered_attack_range,
            damage_reward=damage_reward,
            damage_deal=damage_deal,
            player_hurt=player_hurt,
            player_damage_taken=health_lost,
            dodge=dodge,
            victory=victory,
            silk_spent=silk_spent,
            silk_penalty=silk_penalty,
            player_health_lost=health_lost,
            attack_finished=attack_finished,
            player_dead=player_dead,
            boss_dead=boss_dead,
            terminated=player_dead or boss_dead,
        )
