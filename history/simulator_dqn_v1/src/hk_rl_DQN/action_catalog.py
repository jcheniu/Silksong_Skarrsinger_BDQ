"""Canonical Hornet action catalog shared by training and the real adapter.

The game-facing names describe input intents, not implementation details. The
current Python simulator implements the first six entries. The remaining
entries are deliberately marked ``runtime_only`` until their availability and
cooldowns are observed in Silksong telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    name: str
    key: str | None
    buttons: tuple[str, ...]
    simulator: bool
    runtime_only: bool
    description: str


ACTION_CATALOG: tuple[ActionSpec, ...] = (
    ActionSpec("left", "LeftArrow", ("move_left",), True, False, "Move left"),
    ActionSpec("right", "RightArrow", ("move_right",), True, False, "Move right"),
    ActionSpec("dash", "C", ("dash",), True, False, "Ground or air dash"),
    ActionSpec("attack", "X", ("attack",), True, False, "Needle attack; tap X"),
    ActionSpec("wait", None, (), True, False, "No intentional input"),
    ActionSpec("jump", "Z", ("jump",), True, False, "Jump; wall jump depends on contact"),
    ActionSpec("attack_charge", "X", ("attack",), False, True, "Hold X; charge attack duration is runtime-dependent"),
    ActionSpec("cast", "A", ("cast",), False, True, "Spell / cast; requires silk and an equipped spell"),
    ActionSpec("super_dash", "S", ("super_dash",), False, True, "Super dash; requires a valid charge state"),
    ActionSpec("quick_run", "C", ("dash",), False, True, "Hold C for fast run; duration is runtime-dependent"),
    ActionSpec("quick_cast", "LeftShift", ("quick_cast",), False, True, "Quick cast / quick skill"),
    ActionSpec("dreamnail", "D", ("dreamnail",), False, True, "Dreamnail-like interaction where available"),
    ActionSpec("taunt", "V", ("taunt",), False, True, "Taunt"),
    ActionSpec("wall_jump", "Z+direction", ("jump", "move_left_or_right"), False, True, "Wall jump/cling escape"),
)

ACTION_NAMES = tuple(spec.name for spec in ACTION_CATALOG)
SIMULATOR_ACTION_NAMES = tuple(spec.name for spec in ACTION_CATALOG if spec.simulator)


def action_index(name: str) -> int:
    try:
        return ACTION_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown Hornet action: {name}") from exc


def catalog_as_dicts() -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "name": spec.name,
            "key": spec.key,
            "buttons": list(spec.buttons),
            "simulator": spec.simulator,
            "runtime_only": spec.runtime_only,
            "description": spec.description,
        }
        for index, spec in enumerate(ACTION_CATALOG)
    ]
