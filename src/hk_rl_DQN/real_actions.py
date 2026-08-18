"""Joint-action catalog, masks, geometry, and exploration policy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random
from typing import Mapping, Sequence

import torch
from torch import nn

from .final_project.action_executor import (
    BRANCH_NAMES,
    BRANCH_SIZES,
    BranchMasks,
    validate_action,
    validate_masks,
)
from .real_state import STATE_DIMENSIONS, StateFrame


EPSILON_START = 0.60
EPSILON_END = 0.05
EPSILON_DECAY_TRANSITIONS = 600_000
EPSILON_RECIPROCAL_SHAPE = 1.0
EXPLORATION_ACTIVATION_RATES = (0.45, 0.85, 0.30)
MOVEMENT_EXPLORATION_WEIGHTS = (0.0, 32.0, 32.0, 8.0, 8.0, 8.0)
COMBAT_EXPLORATION_WEIGHTS = (0.0, 30.0, 8.0, 8.0, 20.0, 20.0)
ACTION_LABELS = (
    ("released", "press_z", "hold_z"),
    (
        "neutral",
        "hold_left",
        "hold_right",
        "left_dash",
        "right_dash",
        "harpoon_s",
    ),
    ("neutral", "tap_x", "hold_x", "shift", "up_x", "down_x"),
)
BRANCH_INDEX = {name: index for index, name in enumerate(BRANCH_NAMES)}


def _policy_action_allowed(action: tuple[int, ...]) -> bool:
    jump, movement, combat = action
    if movement == 5:
        return action == (0, 5, 0)
    if movement in (3, 4) and combat not in (0, 1):
        return False
    if jump == 1 and combat not in (0, 2):
        return False
    return True


JOINT_ACTIONS = tuple(
    action
    for action in product(*(range(size) for size in BRANCH_SIZES))
    if _policy_action_allowed(action)
)
JOINT_ACTION_INDEX = {action: index for index, action in enumerate(JOINT_ACTIONS)}
JOINT_ACTION_COUNT = len(JOINT_ACTIONS)
if JOINT_ACTION_COUNT != 53:
    raise AssertionError(f"curated action catalog has {JOINT_ACTION_COUNT} actions, expected 53")
@dataclass(frozen=True)
class AttackOpportunity:
    """Three-zone attack geometry evaluated when an action starts."""

    boss_vulnerable: bool
    horizontal_allowed: bool
    horizontal_confirmed: bool
    up_allowed: bool
    up_confirmed: bool
    down_allowed: bool
    down_confirmed: bool

    def allowed(self, combat_action: int) -> bool:
        if combat_action in (1, 2):
            return self.horizontal_allowed
        if combat_action == 4:
            return self.up_allowed
        if combat_action == 5:
            return self.down_allowed
        return True

    def confirmed(self, combat_action: int) -> bool:
        if combat_action in (1, 2):
            return self.horizontal_confirmed
        if combat_action == 4:
            return self.up_confirmed
        if combat_action == 5:
            return self.down_confirmed
        return self.boss_vulnerable


@dataclass
class ActionExplorationState:
    """Short jump and direction commitment shared by explore and greedy play."""

    sticky_movement: int = 0
    movement_ticks: int = 0
    sticky_jump: int = 0
    jump_ticks: int = 0

    @property
    def remaining_ticks(self) -> int:
        """Compatibility view used by existing movement tests and tooling."""

        return self.movement_ticks

    def consume_movement(self, mask: Sequence[bool]) -> int | None:
        if self.movement_ticks <= 0 or self.sticky_movement not in (1, 2):
            self.clear_movement()
            return None
        if not mask[self.sticky_movement]:
            self.clear_movement()
            return None
        value = self.sticky_movement
        self.movement_ticks -= 1
        if self.movement_ticks == 0:
            self.sticky_movement = 0
        return value

    def consume_jump(self, mask: Sequence[bool]) -> int | None:
        if self.jump_ticks <= 0 or self.sticky_jump != 2:
            self.clear_jump()
            return None
        if not mask[self.sticky_jump]:
            self.clear_jump()
            return None
        self.jump_ticks -= 1
        if self.jump_ticks == 0:
            self.sticky_jump = 0
        return 2

    def start(self, movement: int, additional_ticks: int) -> None:
        self.sticky_movement = movement
        self.movement_ticks = additional_ticks

    def start_jump(self, additional_ticks: int) -> None:
        self.sticky_jump = 2
        self.jump_ticks = additional_ticks

    def reconcile(self, executed_action: Sequence[int]) -> None:
        executed_jump, executed_movement, _combat = validate_action(executed_action)
        if self.sticky_movement and executed_movement != self.sticky_movement:
            self.clear_movement()
        if self.sticky_jump and executed_jump not in (1, 2):
            self.clear_jump()

    def clear_movement(self) -> None:
        self.sticky_movement = 0
        self.movement_ticks = 0

    def clear_jump(self) -> None:
        self.sticky_jump = 0
        self.jump_ticks = 0

    def clear(self) -> None:
        self.clear_movement()
        self.clear_jump()


def joint_action_id(action: Sequence[int]) -> int:
    values = validate_action(action)
    try:
        return JOINT_ACTION_INDEX[values]
    except KeyError:
        raise ValueError(f"action is not in the curated joint catalog: {values}") from None


def decode_joint_action(action_id: int) -> tuple[int, ...]:
    if not 0 <= int(action_id) < JOINT_ACTION_COUNT:
        raise ValueError(f"joint action must be in [0, {JOINT_ACTION_COUNT - 1}]")
    return JOINT_ACTIONS[int(action_id)]


def joint_action_mask(branch_masks: BranchMasks) -> tuple[bool, ...]:
    masks = validate_masks(branch_masks)
    return tuple(
        all(masks[index][value] for index, value in enumerate(action))
        for action in JOINT_ACTIONS
    )


def coordinate_temporal_action(action: Sequence[int]) -> tuple[int, ...]:
    """Canonicalize atomic temporal actions before they reach the executor."""

    jump, movement, combat = validate_action(action)
    if movement == 5:
        return (0, 5, 0)
    return jump, movement, combat


def _entity_value(entity: Mapping[str, object] | None, name: str) -> float:
    if entity is None:
        return 0.0
    try:
        return float(entity.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def attack_opportunity(state: StateFrame) -> AttackOpportunity:
    """Classify hard-blocked, predictive fringe, and confirmed attack zones."""

    control = state.control_state.lower()
    boss_vulnerable = (
        state.boss is not None
        and state.boss_vulnerable is not False
        and state.reaction != "dead"
        and state.phase_event == "none"
        and not any(
            token in control
            for token in ("battle start", "entry", "roar", "hornet dead")
        )
    )
    if not boss_vulnerable or state.player is None or state.boss is None:
        return AttackOpportunity(False, False, False, False, False, False, False)

    dx = _entity_value(state.boss, "x") - _entity_value(state.player, "x")
    dy = _entity_value(state.boss, "y") - _entity_value(state.player, "y")
    relative_vx = _entity_value(state.boss, "velocity_x") - _entity_value(
        state.player, "velocity_x"
    )
    relative_vy = _entity_value(state.boss, "velocity_y") - _entity_value(
        state.player, "velocity_y"
    )
    predicted_dx = dx + relative_vx * 0.15
    predicted_dy = dy + relative_vy * 0.15
    facing = _entity_value(state.player, "facing")
    facing_target = facing == 0.0 or facing * predicted_dx >= -0.5
    airborne = state.observation[8] < 0.5

    horizontal_allowed = (
        abs(predicted_dx) <= 8.0
        and abs(predicted_dy) <= 4.5
    )
    horizontal_confirmed = (
        abs(predicted_dx) <= 6.5
        and abs(predicted_dy) <= 3.5
        and facing_target
    )
    up_allowed = abs(predicted_dx) <= 4.0 and -0.5 <= predicted_dy <= 7.0
    up_confirmed = abs(predicted_dx) <= 3.0 and 0.0 < predicted_dy <= 6.0
    down_allowed = (
        airborne
        and abs(predicted_dx) <= 4.0
        and -7.0 <= predicted_dy <= 0.5
    )
    down_confirmed = (
        airborne
        and abs(predicted_dx) <= 3.0
        and -6.0 <= predicted_dy < 0.0
    )
    return AttackOpportunity(
        True,
        horizontal_allowed,
        horizontal_confirmed,
        up_allowed,
        up_confirmed,
        down_allowed,
        down_confirmed,
    )


def apply_attack_opportunity_mask(
    branch_masks: BranchMasks,
    opportunity: AttackOpportunity,
    continuing_action: Sequence[int],
) -> tuple[BranchMasks, tuple[str, ...]]:
    """Apply only the hard outer range; fringe attacks remain legal probes."""

    masks = [list(branch) for branch in validate_masks(branch_masks)]
    continuing = validate_action(continuing_action)
    reasons: list[str] = []
    for combat_action, label in ((1, "horizontal"), (4, "up"), (5, "down")):
        if masks[BRANCH_INDEX["combat"]][combat_action] and not opportunity.allowed(
            combat_action
        ):
            masks[BRANCH_INDEX["combat"]][combat_action] = False
            reasons.append(f"{label} attack masked: outside predictive range")
    if (
        continuing[BRANCH_INDEX["combat"]] != 2
        and masks[BRANCH_INDEX["combat"]][2]
        and not opportunity.horizontal_allowed
    ):
        masks[BRANCH_INDEX["combat"]][2] = False
        reasons.append("charge start masked: outside predictive range")
    return validate_masks(masks), tuple(reasons)


def danger_requires_commitment_break(state: StateFrame) -> bool:
    if state.attack_phase == "active":
        return True
    if state.attack_phase != "anticipation" or state.player is None or state.boss is None:
        return False
    dx = _entity_value(state.boss, "x") - _entity_value(state.player, "x")
    dy = _entity_value(state.boss, "y") - _entity_value(state.player, "y")
    relative_vx = _entity_value(state.boss, "velocity_x") - _entity_value(
        state.player, "velocity_x"
    )
    closing = dx == 0.0 or dx * relative_vx < 0.0
    return closing and abs(dx) <= 10.0 and abs(dy) <= 7.0


def epsilon_for_transition(transition: int) -> float:
    fraction = min(1.0, max(0, transition) / EPSILON_DECAY_TRANSITIONS)
    if fraction >= 1.0:
        return EPSILON_END
    floor = 1.0 / (1.0 + EPSILON_RECIPROCAL_SHAPE)
    reciprocal = 1.0 / (1.0 + EPSILON_RECIPROCAL_SHAPE * fraction)
    normalized = (reciprocal - floor) / (1.0 - floor)
    return EPSILON_END + (EPSILON_START - EPSILON_END) * normalized


def _weighted_exploration_choice(
    candidates: Sequence[int],
    weights: Sequence[float],
    rng: random.Random,
) -> int:
    total_weight = sum(weights[index] for index in candidates)
    if total_weight <= 0:
        return rng.choice(list(candidates))
    threshold = rng.random() * total_weight
    selected = candidates[-1]
    for candidate in candidates:
        threshold -= weights[candidate]
        if threshold < 0:
            selected = candidate
            break
    return selected


def select_action(
    network: nn.Module,
    observation: Sequence[float],
    epsilon: float,
    rng: random.Random,
    device: torch.device,
    branch_masks: BranchMasks | None = None,
    exploration_state: ActionExplorationState | None = None,
    selected_q_values: list[float] | None = None,
) -> tuple[int, ...]:
    if len(observation) != STATE_DIMENSIONS:
        raise ValueError(f"expected {STATE_DIMENSIONS} state values")
    with torch.no_grad():
        values = network(
            torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        )
    masks = (
        validate_masks(branch_masks)
        if branch_masks is not None
        else tuple(
            tuple(True for _ in range(size)) for size in BRANCH_SIZES
        )
    )
    joint_mask = joint_action_mask(masks)
    sticky_jump = (
        exploration_state.consume_jump(masks[BRANCH_INDEX["jump_z"]])
        if exploration_state is not None
        else None
    )
    sticky_movement = (
        exploration_state.consume_movement(masks[BRANCH_INDEX["movement"]])
        if exploration_state is not None
        else None
    )
    explore = rng.random() < epsilon
    if explore:
        legal_actions = [
            action
            for action, allowed in zip(JOINT_ACTIONS, joint_mask)
            if allowed
            and (sticky_jump is None or action[BRANCH_INDEX["jump_z"]] == sticky_jump)
            and (
                sticky_movement is None
                or action[BRANCH_INDEX["movement"]] == sticky_movement
            )
        ]
        if not legal_actions:
            raise ValueError("curated action mask must allow at least one joint action")
        selected_action = legal_actions[0]
        for _attempt in range(100):
            sampled: list[int] = []
            for branch_index, (size, mask) in enumerate(zip(BRANCH_SIZES, masks)):
                available = [index for index, allowed in enumerate(mask) if allowed]
                if not available:
                    raise ValueError("every action branch must have an available value")
                if branch_index == BRANCH_INDEX["jump_z"] and sticky_jump is not None:
                    sampled.append(sticky_jump)
                    continue
                if (
                    branch_index == BRANCH_INDEX["movement"]
                    and sticky_movement is not None
                ):
                    sampled.append(sticky_movement)
                    continue
                non_neutral = [index for index in available if index != 0]
                activation_rate = EXPLORATION_ACTIVATION_RATES[branch_index]
                if non_neutral and rng.random() < activation_rate:
                    if branch_index == BRANCH_INDEX["movement"]:
                        sampled.append(
                            _weighted_exploration_choice(
                                non_neutral, MOVEMENT_EXPLORATION_WEIGHTS, rng
                            )
                        )
                    elif branch_index == BRANCH_INDEX["combat"]:
                        sampled.append(
                            _weighted_exploration_choice(
                                non_neutral, COMBAT_EXPLORATION_WEIGHTS, rng
                            )
                        )
                    else:
                        sampled.append(rng.choice(non_neutral))
                else:
                    sampled.append(0 if 0 in available else rng.choice(available))
            candidate = coordinate_temporal_action(sampled)
            if candidate in legal_actions:
                selected_action = candidate
                break
        else:
            selected_action = rng.choice(legal_actions)
        selected_id = joint_action_id(selected_action)
    else:
        scores = values.squeeze(0).clone()
        legal = [
            allowed
            and (sticky_jump is None or action[BRANCH_INDEX["jump_z"]] == sticky_jump)
            and (sticky_movement is None or action[BRANCH_INDEX["movement"]] == sticky_movement)
            for action, allowed in zip(JOINT_ACTIONS, joint_mask)
        ]
        scores[torch.tensor([not allowed for allowed in legal], device=device)] = -torch.inf
        selected_id = int(scores.argmax().item())
        selected_action = decode_joint_action(selected_id)
    if exploration_state is not None:
        if sticky_jump is None and selected_action[BRANCH_INDEX["jump_z"]] in (1, 2):
            exploration_state.start_jump(rng.choice([3, 5]))
        if (
            sticky_movement is None
            and selected_action[BRANCH_INDEX["movement"]] in (1, 2)
        ):
            exploration_state.start(
                selected_action[BRANCH_INDEX["movement"]], rng.choice([3, 5])
            )
    if selected_q_values is not None:
        selected_q_values.append(
            float(values[0, selected_id].detach().cpu().item())
        )
    return selected_action
