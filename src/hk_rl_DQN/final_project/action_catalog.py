"""Final, combat-only Hornet action vocabulary.

The key bindings were read from the local Silksong Unity registry. Actions
that share a key are separate intents for the real adapter; the adapter must
choose the valid intent from telemetry (silk, FSM state, and cooldown).
Healing is intentionally excluded from the policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    name: str
    key: str | None
    min_hold_ms: int
    consumes_silk: bool
    description: str


ACTION_CATALOG: tuple[ActionSpec, ...] = (
    ActionSpec("left", "LeftArrow", 100, False, "Move left"),
    ActionSpec("right", "RightArrow", 100, False, "Move right"),
    ActionSpec("wait", None, 100, False, "No intentional input"),
    ActionSpec("jump", "Z", 0, False, "Jump"),
    ActionSpec("jump_hold", "Z", 100, False, "Continue a long jump"),
    ActionSpec("double_jump", "Z", 0, False, "Airborne second jump"),
    ActionSpec("dash", "C", 50, False, "Tap C for a dash"),
    ActionSpec("quick_run", "C", 300, False, "Hold C for fast run"),
    ActionSpec("attack", "X", 50, False, "Tap X for a normal attack"),
    ActionSpec("attack_charge", "X", 1350, False, "Hold X continuously for 1.35s / 81 frames; interruption invalidates the charge"),
    ActionSpec("quick_cast", "LeftShift", 50, True, "Cast the equipped spell; consumes silk"),
    ActionSpec("harpoon_dash", "S", 50, False, "Harpoon dash / KeySupDash"),
    ActionSpec("taunt", "V", 50, False, "Battle taunt"),
)

ACTION_NAMES = tuple(spec.name for spec in ACTION_CATALOG)


def get_action(name: str) -> ActionSpec:
    for spec in ACTION_CATALOG:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown final-project action: {name}")
