"""Reward events derived from consecutive live-game telemetry snapshots.

The values and reward intent mirror the simulator used by train_dqn.py. This
module contains no DQN, replay buffer, action selection, or training code.
Raw Boss HP is intentionally absent; the plugin supplies only cumulative
damage dealt, which is converted to a proportional reward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
import math
from typing import Mapping

from .real_actions import BRANCH_INDEX, AttackOpportunity, _entity_value
from .real_replay import Transition
from .real_state import (
    ARENA_CENTER_X,
    ARENA_HALF_WIDTH,
    COLLISION_RISK_INDEX,
    StateFrame,
    encode_snapshot,
)


# The 50 ms controller emits twice as many transitions as the old 100 ms
# controller, so per-tick shaping is halved to preserve its per-second weight.
STEP_PENALTY = -0.001
ATTACK_RANGE_REWARD = 0.2
DAMAGE_REWARD_PER_HP = 0.1
ILLEGAL_ACTION_PENALTY = -1.0
PLAYER_DAMAGE_PENALTY_PER_HP = -3.6
# Compatibility alias for callers that imported the old fixed-event constant.
PLAYER_HURT_PENALTY = PLAYER_DAMAGE_PENALTY_PER_HP
DODGE_REWARD = 0.4
PLAYER_PARRY_REWARD = 0.5
VICTORY_REWARD = 10.0
SILK_SPEND_PENALTY_PER_UNIT = -0.04
ATTACK_COMBO_GAP_SECONDS = 0.25
FALLBACK_SAMPLE_SECONDS = 0.1


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
    player_parries: int
    parry_reward: float
    victory: float
    silk_spent: int
    silk_penalty: float
    player_health_lost: int
    attack_finished: str | None
    attack_hurt_player: bool
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
        self._previous_player_parry_events: int | None = None
        self._entered_attack_range = False
        self._attack_types: list[str] = []
        self._attack_hurt_player = False
        self._attack_became_active = False
        self._attack_observed_from_start = False
        self._attack_last_seen_time: float | None = None
        self._saw_idle_before_attack = False
        self._clock: float | None = None
        self._raw_timestamp: float | None = None
        self._victory_awarded = False

    def _advance_clock(self, timestamp: float) -> float:
        if self._clock is None:
            self._clock = timestamp
        elif self._raw_timestamp is not None and timestamp > self._raw_timestamp:
            self._clock += timestamp - self._raw_timestamp
        else:
            self._clock += FALLBACK_SAMPLE_SECONDS
        self._raw_timestamp = timestamp
        return self._clock

    def _finish_attack_window(self) -> tuple[str | None, float, bool]:
        if not self._attack_types:
            return None, 0.0, False
        name = "+".join(self._attack_types)
        hurt_player = self._attack_hurt_player
        dodge = (
            DODGE_REWARD
            if self._attack_observed_from_start
            and self._attack_became_active
            and not hurt_player
            else 0.0
        )
        self._attack_types = []
        self._attack_hurt_player = False
        self._attack_became_active = False
        self._attack_observed_from_start = False
        self._attack_last_seen_time = None
        return name, dodge, hurt_player

    def step(self, snapshot: Mapping[str, object]) -> RewardFrame:
        """Score one snapshot relative to all earlier snapshots this episode."""

        state = encode_snapshot(snapshot)
        clock = self._advance_clock(state.timestamp)
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

        raw_parry_events = snapshot.get("player_parry_events")
        parry_events = (
            int(raw_parry_events)
            if isinstance(raw_parry_events, (int, float))
            and not isinstance(raw_parry_events, bool)
            and float(raw_parry_events).is_integer()
            and int(raw_parry_events) >= 0
            else None
        )
        new_parry_events = 0
        if parry_events is not None:
            if (
                self._previous_player_parry_events is not None
                and parry_events >= self._previous_player_parry_events
            ):
                new_parry_events = parry_events - self._previous_player_parry_events
            self._previous_player_parry_events = parry_events

        attack_finished: str | None = None
        attack_hurt_player = False
        dodge = 0.0
        current_attack = state.attack_type if state.attack_type != "none" else None
        gap_expired = (
            self._attack_last_seen_time is not None
            and clock - self._attack_last_seen_time >= ATTACK_COMBO_GAP_SECONDS
        )
        if self._attack_types and not gap_expired and health_lost > 0:
            self._attack_hurt_player = True
        if self._attack_types and gap_expired:
            attack_finished, dodge, attack_hurt_player = self._finish_attack_window()

        if current_attack is not None:
            if not self._attack_types:
                self._attack_observed_from_start = (
                    state.attack_phase == "anticipation" or self._saw_idle_before_attack
                )
                self._saw_idle_before_attack = False
            if current_attack not in self._attack_types:
                self._attack_types.append(current_attack)
            self._attack_became_active |= state.attack_phase == "active"
            self._attack_hurt_player |= health_lost > 0
            self._attack_last_seen_time = clock
        else:
            self._saw_idle_before_attack = True

        player_parries = (
            new_parry_events
            if current_attack is not None
            or self._attack_types and not gap_expired
            else 0
        )
        parry_reward = PLAYER_PARRY_REWARD * player_parries

        player_dead = health is not None and health <= 0
        if state.control_state == "Hornet Dead":
            player_dead = True
        boss_dead = state.reaction == "dead"
        victory = 0.0
        if boss_dead and not self._victory_awarded:
            self._victory_awarded = True
            victory = VICTORY_REWARD
        if (player_dead or boss_dead) and self._attack_types:
            finished, terminal_dodge, terminal_hurt = self._finish_attack_window()
            if attack_finished is None:
                attack_finished = finished
                dodge = terminal_dodge
                attack_hurt_player = terminal_hurt

        step_reward = STEP_PENALTY
        total = (
            step_reward
            + entered_attack_range
            + damage_reward
            + player_hurt
            + dodge
            + parry_reward
            + victory
            + silk_penalty
        )
        self._previous_health = health
        self._previous_silk = silk
        self._previous_reaction = state.reaction
        return RewardFrame(
            total=total,
            step=step_reward,
            entered_attack_range=entered_attack_range,
            damage_reward=damage_reward,
            damage_deal=damage_deal,
            player_hurt=player_hurt,
            player_damage_taken=health_lost,
            dodge=dodge,
            player_parries=player_parries,
            parry_reward=parry_reward,
            victory=victory,
            silk_spent=silk_spent,
            silk_penalty=silk_penalty,
            player_health_lost=health_lost,
            attack_finished=attack_finished,
            attack_hurt_player=attack_hurt_player,
            player_dead=player_dead,
            boss_dead=boss_dead,
            terminated=player_dead or boss_dead,
        )


# Delayed reward and credit-assignment protocol.
COMBAT_HURT_EVENT_PENALTY = -0.75
EVADE_SUCCESS_REWARD = 0.75
EVADE_FAILURE_PENALTY = -1.0
HARPOON_SUCCESS_BONUS_FRACTION = 0.50
ATTACK_END_GRACE_SECONDS = 0.6
COMBAT_HURT_CREDIT_WINDOW_STEPS = 12
COMBAT_HURT_CREDIT_DECAY = math.sqrt(0.85)
CREDIT_FINALIZATION_STEPS = DAMAGE_CREDIT_WINDOW_STEPS = 40
ATTACK_ANIMATION_COMMITMENT_STEPS = 4
CHARGE_RELEASE_COMMITMENT_STEPS = 10
SPELL_ANIMATION_COMMITMENT_STEPS = 6
ATTACK_MISS_PENALTY = -0.2
SPELL_MISS_PENALTY = -0.8
LONG_NO_DAMAGE_STEPS = 100
LONG_NO_DAMAGE_PENALTY = -0.5
STAGNATION_WINDOW_STEPS = 20
STAGNATION_REGION_FRACTION = 0.1
STAGNATION_PENALTY = -0.025
BOSS_PROXIMITY_REWARD = 0.05
BOSS_PROXIMITY_X = 12.0
BOSS_PROXIMITY_Y = 8.0
ARENA_BOUNDARY_FRACTION = 0.12
ARENA_BOUNDARY_PENALTY = -0.1
CONTACT_RISK_INCREASE_SCALE = -0.25
ZERO_SPACE_RADIUS_X = 1.2
ZERO_SPACE_RADIUS_Y = 1.6
ZERO_SPACE_ENTRY_PENALTY = -0.5
ZERO_SPACE_HOLD_PENALTY = -0.025
ZERO_SPACE_HURT_PENALTY = -0.75
ZERO_SPACE_HURT_WINDOW_SECONDS = 0.4
ZERO_SPACE_OFFENSIVE_CLAWBACK_FRACTION = 1.0

@dataclass
class ActionOutcomeTrial:
    pending: "PendingTransition"
    action_kind: str
    penalize_miss: bool = True
    remaining_steps: int = DAMAGE_CREDIT_WINDOW_STEPS
    completion_steps: int = ATTACK_ANIMATION_COMMITMENT_STEPS
    opportunity_confirmed: bool = False
    completed: bool = False
    interrupted: bool = False
    hit: bool = False


@dataclass
class CombatRiskTrial:
    pending: "PendingTransition"
    action_kind: str
    remaining_steps: int = COMBAT_HURT_CREDIT_WINDOW_STEPS
    overlapped_active_threat: bool = False


@dataclass
class PendingTransition:
    transition: Transition
    created_step: int
    macro_id: int = 0
    attack_ids: set[int] = field(default_factory=set)
    delayed_reward: float = 0.0
    zero_space_exposed: bool = False
    zero_space_hurt: bool = False
    offensive_reward_credit: float = 0.0
    offensive_reward_clawback: float = 0.0

    def add_reward(self, value: float) -> None:
        self.delayed_reward += float(value)

    def add_offensive_reward(self, value: float) -> None:
        amount = float(value)
        self.delayed_reward += amount
        self.offensive_reward_credit += amount

    def claw_back_offensive_reward(self, fraction: float) -> float:
        target = self.offensive_reward_credit * max(0.0, float(fraction))
        amount = max(0.0, target - self.offensive_reward_clawback)
        if amount > 0.0:
            self.delayed_reward -= amount
            self.offensive_reward_clawback += amount
        return amount

    def finalize(self) -> Transition:
        return replace(
            self.transition,
            reward=self.transition.reward + self.delayed_reward,
        )


@dataclass
class BossAttackCreditWindow:
    attack_id: int
    attack_type: str = "unknown"
    active_seen: bool = False
    hurt_player: bool = False
    finished_step: int | None = None
    transitions: list[PendingTransition] = field(default_factory=list)


@dataclass
class EpisodeMetrics:
    episode: int
    evaluation: bool = False
    steps: int = 0
    reward: float = 0.0
    gradient_updates: int = 0
    loss_total: float = 0.0
    actual_action_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    policy_q_total: float = 0.0
    policy_q_samples: int = 0
    damage_deal: int = 0
    player_hurts: int = 0
    player_damage_taken: int = 0
    dodges: int = 0
    won: bool = False
    damage_reward: float = 0.0
    dodge_reward: float = 0.0
    attack_range_reward: float = 0.0
    player_hurt_penalty: float = 0.0
    victory_reward: float = 0.0
    silk_spent: int = 0
    silk_penalty: float = 0.0
    dodge_backfill_reward: float = 0.0
    movement_dodge_reward: float = 0.0
    evade_failure_backfill_penalty: float = 0.0
    combat_hurt_penalty: float = 0.0
    failed_dodges: int = 0
    damage_credit_reward: float = 0.0
    unattributed_damage_reward: float = 0.0
    player_parries: int = 0
    parry_reward: float = 0.0
    parry_credit_reward: float = 0.0
    unattributed_parry_reward: float = 0.0
    offensive_misses: int = 0
    attack_misses: int = 0
    spell_misses: int = 0
    attack_opportunities: int = 0
    confirmed_range_attacks: int = 0
    fringe_range_attacks: int = 0
    confirmed_range_misses: int = 0
    out_of_range_attack_frames: int = 0
    unattributed_dodges: int = 0
    harpoon_damage_reward: float = 0.0
    harpoon_hit_bonus_reward: float = 0.0
    harpoon_evade_bonus_reward: float = 0.0
    offensive_miss_penalty: float = 0.0
    dodges_by_attack: dict[str, int] = field(default_factory=dict)
    failed_dodges_by_attack: dict[str, int] = field(default_factory=dict)
    illegal_actions: int = 0
    illegal_action_penalty: float = 0.0
    macro_hurt_penalty: float = 0.0
    long_no_damage_events: int = 0
    long_no_damage_penalty: float = 0.0
    stagnation_events: int = 0
    stagnation_penalty: float = 0.0
    boss_proximity_reward: float = 0.0
    boundary_events: int = 0
    boundary_penalty: float = 0.0
    collision_risk_penalty: float = 0.0
    zero_space_entries: int = 0
    zero_space_entry_penalty: float = 0.0
    zero_space_hold_ticks: int = 0
    zero_space_hold_penalty: float = 0.0
    zero_space_hurts: int = 0
    zero_space_hurt_penalty: float = 0.0
    zero_space_proximity_clawback: float = 0.0
    zero_space_offensive_clawback: float = 0.0

    def as_dict(
        self,
        epsilon: float,
        replay_size: int,
        global_step: int,
    ) -> dict[str, object]:
        item = asdict(self)
        item["mean_loss"] = (
            self.loss_total / self.gradient_updates
            if self.gradient_updates
            else None
        )
        item["mean_policy_q"] = (
            self.policy_q_total / self.policy_q_samples
            if self.policy_q_samples
            else None
        )
        item["epsilon"] = epsilon
        item["replay_size"] = replay_size
        item["global_step"] = global_step
        del item["loss_total"]
        del item["policy_q_total"]
        del item["policy_q_samples"]
        return item


class RewardCreditMixin:
    """Delayed reward assignment shared by the live trainer."""

    @staticmethod
    def _macro_key(
        action: tuple[int, ...], temporal_owner: str | None = None
    ) -> tuple[str, int]:
        jump, movement, combat = action
        if temporal_owner == "skill_s":
            return ("movement", 5)
        if combat != 0:
            return ("combat", combat)
        if movement != 0:
            return ("movement", movement)
        if jump != 0:
            return ("jump", jump)
        return ("neutral", 0)

    def _record_macro_action(
        self,
        action: tuple[int, ...],
        temporal_owner: str | None,
        player_x: float,
    ) -> None:
        key = self._macro_key(action, temporal_owner)
        if key != self.current_macro_key:
            self.current_macro_key = key
            self.previous_macro_id = self.next_macro_id
            self.next_macro_id += 1
            self.recent_macro_ids.append(self.previous_macro_id)
            self.current_macro_positions.clear()
            self.current_macro_positions.append(player_x)

    def _apply_player_hurt_credit(self, reward: RewardFrame) -> None:
        if reward.player_hurt >= 0:
            return
        macro_ids = set(self.recent_macro_ids)
        candidates = [
            item
            for item in self.pending_credit_transitions
            if item.macro_id in macro_ids
        ]
        if not candidates:
            return
        share = reward.player_hurt / len(candidates)
        for item in candidates:
            item.add_reward(share)
        self.metrics.macro_hurt_penalty += reward.player_hurt
        self.metrics.reward += reward.player_hurt

    @staticmethod
    def _inside_boss_proximity(state: StateFrame) -> bool:
        if state.player is None or state.boss is None:
            return False
        dx = abs(_entity_value(state.boss, "x") - _entity_value(state.player, "x"))
        dy = abs(_entity_value(state.boss, "y") - _entity_value(state.player, "y"))
        inside_large_zone = dx <= BOSS_PROXIMITY_X and dy <= BOSS_PROXIMITY_Y
        inside_close_zone = dx <= 6.0 and dy <= 5.0
        return inside_large_zone and not inside_close_zone

    @staticmethod
    def _inside_zero_space(state: StateFrame) -> bool:
        """Return whether Hornet is inside the Boss-centered lower half ellipse."""

        if state.player is None or state.boss is None:
            return False
        dx = _entity_value(state.boss, "x") - _entity_value(state.player, "x")
        dy = _entity_value(state.boss, "y") - _entity_value(state.player, "y")
        if dy < 0.0:
            return False
        return (
            (dx / ZERO_SPACE_RADIUS_X) ** 2
            + (dy / ZERO_SPACE_RADIUS_Y) ** 2
            <= 1.0 + 1e-9
        )

    def _apply_zero_space_shaping(
        self,
        state: StateFrame,
        pending: PendingTransition,
    ) -> None:
        inside = self._inside_zero_space(state)
        pending.zero_space_exposed = inside
        if inside and not self.was_inside_zero_space:
            pending.add_reward(ZERO_SPACE_ENTRY_PENALTY)
            self.metrics.zero_space_entries += 1
            self.metrics.zero_space_entry_penalty += ZERO_SPACE_ENTRY_PENALTY
            self.metrics.reward += ZERO_SPACE_ENTRY_PENALTY
            if self.proximity_reward_balance > 0.0:
                clawback = -self.proximity_reward_balance
                pending.add_reward(clawback)
                self.metrics.zero_space_proximity_clawback += clawback
                self.metrics.reward += clawback
                self.proximity_reward_balance = 0.0
        elif inside:
            pending.add_reward(ZERO_SPACE_HOLD_PENALTY)
            self.metrics.zero_space_hold_ticks += 1
            self.metrics.zero_space_hold_penalty += ZERO_SPACE_HOLD_PENALTY
            self.metrics.reward += ZERO_SPACE_HOLD_PENALTY
        self.was_inside_zero_space = inside

    def _credit_offensive_reward(
        self,
        pending: PendingTransition,
        value: float,
    ) -> None:
        pending.add_offensive_reward(value)
        if pending.zero_space_hurt:
            clawback = pending.claw_back_offensive_reward(
                ZERO_SPACE_OFFENSIVE_CLAWBACK_FRACTION
            )
            self.metrics.zero_space_offensive_clawback -= clawback
            self.metrics.reward -= clawback

    def _apply_zero_space_hurt_credit(self, reward: RewardFrame) -> None:
        if reward.player_damage_taken <= 0:
            return
        candidates = [
            item
            for item in self.pending_credit_transitions
            if item.zero_space_exposed
            and self.credit_step - item.created_step
            <= self.zero_space_hurt_window_steps
        ]
        if not candidates:
            return
        share = ZERO_SPACE_HURT_PENALTY / len(candidates)
        clawback = 0.0
        for item in candidates:
            item.zero_space_hurt = True
            item.add_reward(share)
            clawback += item.claw_back_offensive_reward(
                ZERO_SPACE_OFFENSIVE_CLAWBACK_FRACTION
            )
        self.metrics.zero_space_hurts += 1
        self.metrics.zero_space_hurt_penalty += ZERO_SPACE_HURT_PENALTY
        self.metrics.zero_space_offensive_clawback -= clawback
        self.metrics.reward += ZERO_SPACE_HURT_PENALTY - clawback

    def _apply_dense_shaping(
        self,
        reward: RewardFrame,
        state: StateFrame,
        pending: PendingTransition,
    ) -> None:
        if reward.damage_deal > 0:
            self.no_damage_steps = 0
            self.proximity_reward_available = True
            self.proximity_reward_balance = 0.0
        else:
            self.no_damage_steps += 1
            if self.no_damage_steps % LONG_NO_DAMAGE_STEPS == 0:
                pending.add_reward(LONG_NO_DAMAGE_PENALTY)
                self.metrics.long_no_damage_events += 1
                self.metrics.long_no_damage_penalty += LONG_NO_DAMAGE_PENALTY
                self.metrics.reward += LONG_NO_DAMAGE_PENALTY

        inside_proximity = self._inside_boss_proximity(state)
        entered_proximity = inside_proximity and not self.was_inside_boss_proximity
        if entered_proximity and self.proximity_reward_available:
            pending.add_reward(BOSS_PROXIMITY_REWARD)
            self.metrics.boss_proximity_reward += BOSS_PROXIMITY_REWARD
            self.metrics.reward += BOSS_PROXIMITY_REWARD
            self.proximity_reward_balance += BOSS_PROXIMITY_REWARD
            self.proximity_reward_available = False
        self.was_inside_boss_proximity = inside_proximity

        player_x = _entity_value(state.player, "x")
        normalized_player_x = abs((player_x - ARENA_CENTER_X) / ARENA_HALF_WIDTH)
        boundary_threshold = 1.0 - 2.0 * ARENA_BOUNDARY_FRACTION
        if normalized_player_x >= boundary_threshold:
            pending.add_reward(ARENA_BOUNDARY_PENALTY)
            self.metrics.boundary_events += 1
            self.metrics.boundary_penalty += ARENA_BOUNDARY_PENALTY
            self.metrics.reward += ARENA_BOUNDARY_PENALTY
        self.current_macro_positions.append(player_x)
        movement = pending.transition.action_vector[BRANCH_INDEX["movement"]]
        previous_risk = pending.transition.state[COLLISION_RISK_INDEX]
        current_risk = pending.transition.next_state[COLLISION_RISK_INDEX]
        risk_increase = max(0.0, current_risk - previous_risk)
        if movement in (1, 2, 3, 4) and risk_increase > 0.0:
            collision_penalty = CONTACT_RISK_INCREASE_SCALE * risk_increase
            pending.add_reward(collision_penalty)
            self.metrics.collision_risk_penalty += collision_penalty
            self.metrics.reward += collision_penalty
        arena_width = 2.0 * ARENA_HALF_WIDTH
        stagnant = (
            movement in (1, 2, 3, 4)
            and len(self.current_macro_positions) == self.current_macro_positions.maxlen
            and max(self.current_macro_positions) - min(self.current_macro_positions)
            <= arena_width * STAGNATION_REGION_FRACTION
        )
        if stagnant:
            pending.add_reward(STAGNATION_PENALTY)
            self.metrics.stagnation_events += 1
            self.metrics.stagnation_penalty += STAGNATION_PENALTY
            self.metrics.reward += STAGNATION_PENALTY

    def _register_action_outcome(self, pending: PendingTransition) -> None:
        action_kinds = {
            "attack_x": ("attack", True),
            "skill_s": ("harpoon", False),
            "spell_shift": ("spell", True),
        }
        for event_name, (action_kind, penalize_miss) in action_kinds.items():
            if event_name in self.previous_started_branches:
                opportunity_confirmed = False
                completion_steps = ATTACK_ANIMATION_COMMITMENT_STEPS
                if action_kind == "attack":
                    combat_action = (
                        2
                        if self.previous_charge_released
                        else pending.transition.action_vector[BRANCH_INDEX["combat"]]
                    )
                    opportunity = self.previous_attack_opportunity
                    opportunity_confirmed = bool(
                        opportunity is not None
                        and opportunity.confirmed(combat_action)
                    )
                    if self.previous_charge_released:
                        completion_steps = CHARGE_RELEASE_COMMITMENT_STEPS
                    self.metrics.attack_opportunities += 1
                    if opportunity_confirmed:
                        self.metrics.confirmed_range_attacks += 1
                    else:
                        self.metrics.fringe_range_attacks += 1
                elif action_kind == "spell":
                    completion_steps = SPELL_ANIMATION_COMMITMENT_STEPS
                    opportunity_confirmed = bool(
                        self.previous_attack_opportunity is not None
                        and self.previous_attack_opportunity.boss_vulnerable
                    )
                self.action_outcome_trials.append(
                    ActionOutcomeTrial(
                        pending,
                        action_kind,
                        penalize_miss,
                        completion_steps=completion_steps,
                        opportunity_confirmed=opportunity_confirmed,
                    )
                )

    def _penalize_offensive_miss(self, trial: ActionOutcomeTrial) -> None:
        penalty = (
            ATTACK_MISS_PENALTY
            if trial.action_kind == "attack"
            else SPELL_MISS_PENALTY
        )
        trial.pending.add_reward(penalty)
        self.metrics.reward += penalty
        self.metrics.offensive_misses += 1
        self.metrics.offensive_miss_penalty += penalty
        if trial.action_kind == "attack":
            self.metrics.attack_misses += 1
            self.metrics.confirmed_range_misses += 1
        elif trial.action_kind == "spell":
            self.metrics.spell_misses += 1

    def _apply_action_outcomes(
        self,
        reward: RewardFrame,
        state: StateFrame,
    ) -> None:
        if reward.damage_reward > 0:
            if self.action_outcome_trials:
                newest_remaining = max(
                    trial.remaining_steps for trial in self.action_outcome_trials
                )
                candidates = [
                    trial
                    for trial in self.action_outcome_trials
                    if trial.remaining_steps == newest_remaining
                ]
                share = reward.damage_reward / max(1, len(candidates))
                for trial in candidates:
                    trial.hit = True
                    if trial.action_kind == "harpoon":
                        bonus = share * HARPOON_SUCCESS_BONUS_FRACTION
                        self._credit_offensive_reward(
                            trial.pending,
                            share + bonus,
                        )
                        self.metrics.harpoon_damage_reward += share
                        self.metrics.harpoon_hit_bonus_reward += bonus
                    else:
                        self._credit_offensive_reward(trial.pending, share)
                self.metrics.damage_credit_reward += reward.damage_reward
            else:
                self.metrics.unattributed_damage_reward += reward.damage_reward

        if reward.parry_reward > 0:
            attack_trials = [
                trial
                for trial in self.action_outcome_trials
                if trial.action_kind == "attack"
            ]
            if attack_trials:
                newest_remaining = max(
                    trial.remaining_steps for trial in attack_trials
                )
                candidates = [
                    trial
                    for trial in attack_trials
                    if trial.remaining_steps == newest_remaining
                ]
                share = reward.parry_reward / max(1, len(candidates))
                for trial in candidates:
                    trial.hit = True
                    trial.pending.add_reward(share)
                self.metrics.parry_credit_reward += reward.parry_reward
            else:
                self.metrics.unattributed_parry_reward += reward.parry_reward

        remaining: list[ActionOutcomeTrial] = []
        for trial in self.action_outcome_trials:
            if not trial.completed and not trial.interrupted:
                if reward.player_damage_taken > 0 or reward.terminated:
                    trial.interrupted = True
                elif state.phase_event != "none" or state.reaction == "dead":
                    trial.interrupted = True
                else:
                    trial.completion_steps -= 1
                    trial.completed = trial.completion_steps <= 0
            trial.remaining_steps -= 1
            if trial.remaining_steps > 0:
                remaining.append(trial)
                continue
            if (
                not trial.hit
                and trial.penalize_miss
                and trial.opportunity_confirmed
                and trial.completed
                and not trial.interrupted
            ):
                self._penalize_offensive_miss(trial)
        self.action_outcome_trials = remaining

    @staticmethod
    def _attack_credit_weight(action: tuple[int, ...]) -> float:
        jump, movement, _combat = action
        weight = 0.0
        if jump != 0:
            weight += 1.0
        if movement in (3, 4, 5):
            weight += 1.0
        elif movement in (1, 2):
            weight += 0.7
        return weight

    @staticmethod
    def _successful_evade_action(action: tuple[int, ...]) -> bool:
        jump, movement, combat = action
        return combat == 0 and (jump != 0 or movement in (1, 2, 3, 4, 5))

    @classmethod
    def _failed_evade_weight(cls, action: tuple[int, ...]) -> float:
        defensive_weight = cls._attack_credit_weight(action)
        return defensive_weight if defensive_weight > 0 else 0.25

    def _window(self, attack_id: int, attack_type: str = "unknown") -> BossAttackCreditWindow:
        window = self.attack_windows.get(attack_id)
        if window is None:
            window = BossAttackCreditWindow(attack_id, attack_type)
            self.attack_windows[attack_id] = window
        elif window.attack_type == "unknown" and attack_type != "unknown":
            window.attack_type = attack_type
        return window

    def _attach_to_attack(
        self,
        pending: PendingTransition,
        attack_id: int,
        attack_type: str = "unknown",
    ) -> None:
        if attack_id == 0 or attack_id in pending.attack_ids:
            return
        window = self._window(attack_id, attack_type)
        window.transitions.append(pending)
        pending.attack_ids.add(attack_id)

    def _resolve_attack_window(self, attack_id: int) -> None:
        window = self.attack_windows.pop(attack_id, None)
        if window is None:
            return
        unique = []
        seen: set[int] = set()
        for item in window.transitions:
            marker = id(item)
            if marker not in seen:
                seen.add(marker)
                unique.append(item)
            item.attack_ids.discard(attack_id)
        if not unique or not window.active_seen:
            return
        harpoon_bonus = 0.0
        if window.hurt_player:
            weights = [
                self._failed_evade_weight(item.transition.action_vector)
                for item in unique
            ]
            total_weight = sum(weights)
            applied_budget = EVADE_FAILURE_PENALTY if total_weight > 0 else 0.0
            if total_weight > 0:
                for item, weight in zip(unique, weights):
                    item.add_reward(EVADE_FAILURE_PENALTY * weight / total_weight)
        else:
            final_index = max(1, len(unique) - 1)
            weights = [
                1.0 - 0.5 * index / final_index
                for index in range(len(unique))
            ]
            total_weight = sum(weights)
            applied_budget = EVADE_SUCCESS_REWARD if total_weight > 0 else 0.0
            for item, weight in zip(unique, weights):
                value = EVADE_SUCCESS_REWARD * weight / total_weight
                item.add_reward(value)
            harpoon_items = [
                item
                for item in unique
                if item.transition.action_vector[BRANCH_INDEX["movement"]] == 5
            ]
            if harpoon_items:
                harpoon_bonus = (
                    EVADE_SUCCESS_REWARD * HARPOON_SUCCESS_BONUS_FRACTION
                )
                for item in harpoon_items:
                    item.add_reward(harpoon_bonus / len(harpoon_items))
        if window.hurt_player:
            self.metrics.failed_dodges += 1
            self.metrics.evade_failure_backfill_penalty += applied_budget
            self.metrics.failed_dodges_by_attack[window.attack_type] = (
                self.metrics.failed_dodges_by_attack.get(window.attack_type, 0) + 1
            )
        else:
            self.metrics.dodges += 1
            self.metrics.dodge_reward += DODGE_REWARD
            self.metrics.dodge_backfill_reward += applied_budget
            self.metrics.movement_dodge_reward += applied_budget
            self.metrics.harpoon_evade_bonus_reward += harpoon_bonus
            self.metrics.dodges_by_attack[window.attack_type] = (
                self.metrics.dodges_by_attack.get(window.attack_type, 0) + 1
            )
        self.metrics.reward += applied_budget + harpoon_bonus

    def _mark_attack_finished(self, attack_id: int) -> None:
        if attack_id == 0:
            return
        window = self._window(attack_id)
        if window.finished_step is None:
            window.finished_step = self.credit_step

    def _resolve_mature_attack_windows(self) -> None:
        mature = [
            attack_id
            for attack_id, window in self.attack_windows.items()
            if window.finished_step is not None
            and self.credit_step - window.finished_step >= self.attack_end_grace_steps
        ]
        for attack_id in mature:
            self._resolve_attack_window(attack_id)

    @staticmethod
    def _counter(mapping: Mapping[str, object], name: str) -> int:
        value = mapping.get(name, 0)
        if isinstance(value, bool):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _apply_attack_events(
        self,
        snapshot: Mapping[str, object],
        state: StateFrame,
        reward: RewardFrame,
        pending: PendingTransition,
    ) -> None:
        for attack_id, window in tuple(self.attack_windows.items()):
            if window.finished_step is not None:
                self._attach_to_attack(pending, attack_id, window.attack_type)

        raw = snapshot.get("boss_attack")
        if isinstance(raw, Mapping):
            attack_id = self._counter(raw, "id")
            attack_type = str(raw.get("type", "unknown"))
            phase = str(raw.get("phase", "idle"))
            if not self.attack_events_initialized:
                for name in self.attack_event_counters:
                    self.attack_event_counters[name] = self._counter(raw, name)
                self.attack_events_initialized = True
                self.active_attack_id = attack_id
                if attack_id:
                    window = self._window(attack_id, attack_type)
                    window.active_seen |= phase == "active"
                    window.hurt_player |= reward.player_damage_taken > 0
                    self._attach_to_attack(pending, attack_id, attack_type)
                return
            if self.active_attack_id:
                self._attach_to_attack(pending, self.active_attack_id)
            if attack_id:
                window = self._window(attack_id, attack_type)
                window.active_seen |= phase == "active"
                window.hurt_player |= reward.player_damage_taken > 0
                self._attach_to_attack(pending, attack_id, attack_type)
            hit_count = self._counter(raw, "player_hit_events")
            if hit_count > self.attack_event_counters["player_hit_events"]:
                hit_id = self._counter(raw, "last_player_hit_id")
                if hit_id:
                    late_window = self._window(hit_id)
                    late_window.hurt_player = True
                    self._attach_to_attack(pending, hit_id)
            active_count = self._counter(raw, "active_events")
            if active_count > self.attack_event_counters["active_events"]:
                if attack_id:
                    self._window(attack_id, attack_type).active_seen = True
            finished_count = self._counter(raw, "finished_events")
            if finished_count > self.attack_event_counters["finished_events"]:
                finished_id = self._counter(raw, "last_finished_id")
                if finished_id:
                    self._mark_attack_finished(finished_id)
            for name in self.attack_event_counters:
                self.attack_event_counters[name] = self._counter(raw, name)
            self.active_attack_id = attack_id
            return

        if state.attack_type != "none":
            if self.active_attack_id == 0:
                self.synthetic_attack_id -= 1
                self.active_attack_id = self.synthetic_attack_id
            window = self._window(self.active_attack_id, state.attack_type)
            window.active_seen |= state.attack_phase == "active"
            window.hurt_player |= reward.player_damage_taken > 0
            self._attach_to_attack(pending, self.active_attack_id, state.attack_type)
        elif self.active_attack_id and reward.attack_finished is not None:
            window = self._window(self.active_attack_id, reward.attack_finished)
            window.hurt_player |= reward.attack_hurt_player
            self._attach_to_attack(pending, self.active_attack_id)
            self._mark_attack_finished(self.active_attack_id)
            self.active_attack_id = 0

    def _register_combat_risk(self, pending: PendingTransition) -> None:
        event_kinds = {
            "attack_x": "attack",
            "spell_shift": "spell",
        }
        for event_name, action_kind in event_kinds.items():
            if event_name in self.previous_started_branches:
                self.combat_risk_trials.append(
                    CombatRiskTrial(pending, action_kind)
                )

    def _update_combat_risk_overlap(
        self,
        snapshot: Mapping[str, object],
        state: StateFrame,
    ) -> None:
        raw_attack = snapshot.get("boss_attack")
        explicit_active = (
            isinstance(raw_attack, Mapping)
            and str(raw_attack.get("phase", "idle")) == "active"
        )
        if explicit_active or state.attack_phase == "active":
            for trial in self.combat_risk_trials:
                trial.overlapped_active_threat = True

    def _apply_combat_hurt(self, player_damage_taken: int) -> None:
        if player_damage_taken <= 0:
            return
        candidates = [
            trial
            for trial in self.combat_risk_trials
            if trial.overlapped_active_threat
        ]
        if not candidates:
            return
        weights = [
            COMBAT_HURT_CREDIT_DECAY
            ** (COMBAT_HURT_CREDIT_WINDOW_STEPS - trial.remaining_steps)
            for trial in candidates
        ]
        penalty = COMBAT_HURT_EVENT_PENALTY
        total_weight = sum(weights)
        for trial, weight in zip(candidates, weights):
            trial.pending.add_reward(penalty * weight / total_weight)
        self.metrics.combat_hurt_penalty += penalty
        self.metrics.reward += penalty

    def _age_combat_risks(self) -> None:
        remaining = []
        for trial in self.combat_risk_trials:
            trial.remaining_steps -= 1
            if trial.remaining_steps > 0:
                remaining.append(trial)
        self.combat_risk_trials = remaining

    def _finalize_pending(self, force: bool = False) -> None:
        if force:
            for attack_id in list(self.attack_windows):
                self._resolve_attack_window(attack_id)
        retained: deque[PendingTransition] = deque()
        for pending in self.pending_credit_transitions:
            age = self.credit_step - pending.created_step
            ready = (
                age >= CREDIT_FINALIZATION_STEPS
                and not pending.attack_ids
            )
            if force or ready:
                finalized = pending.finalize()
                if not self.evaluation_mode:
                    self.replay.append(finalized)
                self.episode_finalized_reward += finalized.reward
            else:
                retained.append(pending)
        self.pending_credit_transitions = retained

    def _expire_action_outcomes(self) -> None:
        for trial in self.action_outcome_trials:
            if (
                trial.hit
                or not trial.penalize_miss
                or not trial.opportunity_confirmed
                or not trial.completed
                or trial.interrupted
            ):
                continue
            self._penalize_offensive_miss(trial)
        self.action_outcome_trials.clear()
