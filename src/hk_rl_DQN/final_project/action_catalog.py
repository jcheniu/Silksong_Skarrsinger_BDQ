"""Atomic Hornet actions used by recording and cold-start tools.

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
    key: str | tuple[str, ...] | None
    min_hold_ms: int
    consumes_silk: bool
    description: str


ACTION_CATALOG: tuple[ActionSpec, ...] = (
    ActionSpec("left", "LeftArrow", 100, False, "Move left"),
    ActionSpec("right", "RightArrow", 100, False, "Move right"),
    ActionSpec("wait", None, 100, False, "No intentional input"),
    ActionSpec("jump", "Z", 0, False, "Jump"),
    ActionSpec(
        "jump_hold", "Z", 100, False, "Keep Z held for this control interval"
    ),
    ActionSpec("left_dash", ("LeftArrow", "C"), 50, False, "Dash left"),
    ActionSpec("right_dash", ("RightArrow", "C"), 50, False, "Dash right"),
    ActionSpec("attack", "X", 50, False, "Tap X for a normal attack"),
    ActionSpec("up_attack", ("UpArrow", "X"), 50, False, "Hold up and tap X"),
    ActionSpec(
        "down_attack", ("DownArrow", "X"), 50, False, "Hold down and tap X"
    ),
    ActionSpec(
        "attack_charge",
        "X",
        1350,
        False,
        "Hold X for at least 1.35 seconds; release by 3 seconds",
    ),
    ActionSpec("quick_cast", "LeftShift", 50, True, "Cast the equipped spell; consumes silk"),
    ActionSpec("harpoon_dash", "S", 50, False, "Harpoon dash / KeySupDash"),
)

ACTION_NAMES = tuple(spec.name for spec in ACTION_CATALOG)

# Compatibility mapping for single-action tools. Live training emits the full
# semantic vector directly; these entries select one equivalent field value.
ACTION_VECTORS: dict[str, tuple[int, int, int]] = {
    "left": (0, 1, 0),
    "right": (0, 2, 0),
    "wait": (0, 0, 0),
    "jump": (1, 0, 0),
    "jump_hold": (2, 0, 0),
    "left_dash": (0, 3, 0),
    "right_dash": (0, 4, 0),
    "attack": (0, 0, 1),
    "up_attack": (0, 0, 4),
    "down_attack": (0, 0, 5),
    "attack_charge": (0, 0, 2),
    "quick_cast": (0, 0, 3),
    "harpoon_dash": (0, 5, 0),
}


def get_action(name: str) -> ActionSpec:
    for spec in ACTION_CATALOG:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown final-project action: {name}")


def get_action_vector(name: str) -> tuple[int, int, int]:
    try:
        return ACTION_VECTORS[name]
    except KeyError:
        raise ValueError(f"unknown final-project action: {name}") from None
