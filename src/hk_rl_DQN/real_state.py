"""Encode live Silksong telemetry into a compact DQN observation.

This module deliberately contains no network, reward, action, or training code.
It only maps one JSON telemetry snapshot to a fixed-size numeric observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, TextIO


# These limits match the simulator's coordinate conventions. Real positions
# are normalized and clipped so a transient scene-loading value cannot explode
# the input range.
ARENA_WIDTH = 300.0
ARENA_HEIGHT = 300.0
VELOCITY_X_SCALE = 12.0
VELOCITY_Y_SCALE = 12.0
GROUND_Y = 20.0

ATTACK_TYPES = (
    "none",
    "slash",
    "cyclone",
    "ground_throw",
    "air_throw",
    "rethrow",
    "spin_attack",
    "dash_grind",
    "jump_attack",
    "spear_slam",
)
ATTACK_PHASES = ("idle", "anticipation", "active", "recovery")
REACTIONS = ("normal", "hit", "blocked", "evaded", "stunned", "dead")
PHASE_EVENTS = ("none", "roar", "phase_transition")
KINEMATIC_STATE_DIMENSIONS = 9
RESOURCE_STATE_DIMENSIONS = 1
BASE_STATE_DIMENSIONS = (
    KINEMATIC_STATE_DIMENSIONS
    + RESOURCE_STATE_DIMENSIONS
    + len(ATTACK_TYPES)
    + len(ATTACK_PHASES)
    + len(REACTIONS)
    + len(PHASE_EVENTS)
)
CONTROL_STATE_DIMENSIONS = 9
STATE_DIMENSIONS = BASE_STATE_DIMENSIONS + CONTROL_STATE_DIMENSIONS


@dataclass(frozen=True)
class KeyHoldState:
    """Persistent key state required by the Branching-DQN protocol."""

    attack_held: bool = False
    attack_hold_progress: float = 0.0
    dash_held: bool = False
    dash_hold_progress: float = 0.0
    skill_held: bool = False
    skill_hold_progress: float = 0.0
    interrupted: bool = False
    skill_available: bool = False
    spell_available: bool = False

    def as_tuple(self) -> tuple[float, ...]:
        return (
            float(self.attack_held),
            _clip(self.attack_hold_progress, 0.0, 1.0),
            float(self.dash_held),
            _clip(self.dash_hold_progress, 0.0, 1.0),
            float(self.skill_held),
            _clip(self.skill_hold_progress, 0.0, 1.0),
            float(self.interrupted),
            float(self.skill_available),
            float(self.spell_available),
        )


@dataclass(frozen=True)
class PlayerResources:
    """Validated player resource telemetry shared by state and action logic."""

    silk: int | None = None
    silk_max: int | None = None
    silk_parts: int | None = None
    skill_cost: int | None = None
    silk_abilities_disabled: bool | None = None
    skill_available: bool | None = None
    spell_available: bool | None = None

    @property
    def silk_normalized(self) -> float:
        if self.silk is None or self.silk_max is None or self.silk_max <= 0:
            return 0.0
        return _clip(self.silk / self.silk_max, 0.0, 1.0)

    def as_dict(self) -> dict[str, int | bool | None]:
        return {
            "silk": self.silk,
            "silk_max": self.silk_max,
            "silk_parts": self.silk_parts,
            "skill_cost": self.skill_cost,
            "silk_abilities_disabled": self.silk_abilities_disabled,
            "skill_available": self.skill_available,
            "spell_available": self.spell_available,
        }

    @property
    def can_harpoon_dash(self) -> bool:
        return self.skill_available is True

    @property
    def can_quick_cast(self) -> bool:
        return (
            self.spell_available is True
            and self.silk_abilities_disabled is False
            and self.silk is not None
            and self.skill_cost is not None
            and self.skill_cost >= 0
            and self.silk >= self.skill_cost
        )

    @property
    def is_complete(self) -> bool:
        return (
            self.silk is not None
            and self.silk >= 0
            and self.silk_max is not None
            and self.silk_max > 0
            and self.silk <= self.silk_max
            and self.silk_parts is not None
            and self.silk_parts >= 0
            and self.skill_cost is not None
            and self.skill_cost >= 0
            and self.silk_abilities_disabled is not None
            and self.skill_available is not None
            and self.spell_available is not None
        )


@dataclass(frozen=True)
class StateFrame:
    """Decoded state at one telemetry sampling instant."""

    observation: tuple[float, ...]
    timestamp: float
    frame: int
    scene: str
    player: Mapping[str, object] | None
    boss: Mapping[str, object] | None
    resources: PlayerResources
    control_state: str
    attack_type: str
    attack_phase: str
    reaction: str
    phase_event: str


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _one_hot(value: str, choices: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(1.0 if value == choice else 0.0 for choice in choices)


def _position(entity: Mapping[str, object] | None, key: str) -> float:
    return _number(entity.get(key)) if entity is not None else 0.0


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def decode_player_resources(snapshot: Mapping[str, object]) -> PlayerResources:
    """Decode the authoritative player_resources object without guessing."""

    resources = snapshot.get("player_resources")
    if not isinstance(resources, Mapping):
        return PlayerResources()
    return PlayerResources(
        silk=_optional_int(resources.get("silk")),
        silk_max=_optional_int(resources.get("silk_max")),
        silk_parts=_optional_int(resources.get("silk_parts")),
        skill_cost=_optional_int(resources.get("skill_cost")),
        silk_abilities_disabled=_optional_bool(
            resources.get("silk_abilities_disabled")
        ),
        skill_available=_optional_bool(resources.get("skill_available")),
        spell_available=_optional_bool(resources.get("spell_available")),
    )


def _state_name(item: Mapping[str, object]) -> str:
    return str(item.get("state") or "").strip()


def _fsm_items(snapshot: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = snapshot.get("fsm", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _find_fsm(items: Iterable[Mapping[str, object]], name: str, path: str | None = None) -> Mapping[str, object] | None:
    for item in items:
        if item.get("name") != name:
            continue
        if path is None or item.get("path") == path:
            return item
    return None


def _main_control_state(items: list[Mapping[str, object]]) -> str:
    item = _find_fsm(items, "Control", "Boss Scene/Hunter Queen Boss")
    return _state_name(item) if item is not None else ""


def _classify_attack(state: str) -> tuple[str, str]:
    lowered = state.lower()
    if not state:
        return "none", "idle"
    if "slash" in lowered and "spin" not in lowered:
        attack = "slash"
    elif "cyclone" in lowered:
        attack = "cyclone"
    elif "rethrow" in lowered or "re-throw" in lowered:
        attack = "rethrow"
    elif "air throw" in lowered or "airthrow" in lowered or "air sickle" in lowered:
        attack = "air_throw"
    elif lowered.startswith("throw") or "throw " in lowered:
        attack = "ground_throw"
    elif any(token in lowered for token in ("spin attack", "spin antic", "spin dash", "launch spin")):
        attack = "spin_attack"
    elif "dash grind" in lowered or lowered == "dash":
        attack = "dash_grind"
    elif "spear slam" in lowered:
        attack = "spear_slam"
    elif any(token in lowered for token in ("jump", "wall dive", "air dive")):
        attack = "jump_attack"
    else:
        return "none", "idle"

    if "antic" in lowered or "choice" in lowered or "set " in lowered:
        phase = "anticipation"
    elif any(token in lowered for token in ("end", "recover", "land", "recoil")):
        phase = "recovery"
    else:
        phase = "active"
    return attack, phase


def _classify_phase_event(state: str) -> str:
    lowered = state.lower()
    if "p2 roar" in lowered or "p3 roar" in lowered:
        return "phase_transition"
    if "roar" in lowered:
        return "roar"
    return "none"


def _classify_reaction(items: list[Mapping[str, object]], control_state: str) -> str:
    states = [_state_name(item) for item in items]
    lowered = [state.lower() for state in states]
    control_lower = control_state.lower()
    death_fsm_active = any(
        item.get("name") == "Death" and _state_name(item)
        for item in items
    )
    if death_fsm_active or control_lower in {"death", "die", "dead"}:
        return "dead"
    if any("stunned" in state or state in {"stun", "stun start", "stun damage"} for state in lowered):
        return "stunned"
    if any("block" in state for state in lowered):
        return "blocked"
    if any("evade" in state for state in lowered):
        return "evaded"
    if any(
        state in {"hit", "bell bind hit", "hazard hit", "stun damage", "damage recover"}
        or state.endswith(" hit")
        for state in lowered
    ):
        return "hit"
    return "normal"


def _grounded(snapshot: Mapping[str, object], player: Mapping[str, object] | None) -> bool:
    explicit = snapshot.get("player_grounded")
    if isinstance(explicit, bool):
        return explicit
    # Preserve compatibility with completed telemetry files from before the
    # explicit HeroController cState field was added.
    return abs(_position(player, "velocity_y")) < 0.1 and _position(player, "y") <= GROUND_Y


def encode_snapshot(
    snapshot: Mapping[str, object],
    key_state: KeyHoldState | None = None,
) -> StateFrame:
    """Convert one telemetry JSON object into a fixed live observation."""

    player = snapshot.get("player") if isinstance(snapshot.get("player"), Mapping) else None
    boss = snapshot.get("boss") if isinstance(snapshot.get("boss"), Mapping) else None
    player_x = _position(player, "x")
    player_y = _position(player, "y")
    boss_x = _position(boss, "x")
    boss_y = _position(boss, "y")
    items = _fsm_items(snapshot)
    control_state = _main_control_state(items)
    attack_type, attack_phase = _classify_attack(control_state)
    reaction = _classify_reaction(items, control_state)
    phase_event = _classify_phase_event(control_state)
    resources = decode_player_resources(snapshot)

    values = (
        _clip(player_x / ARENA_WIDTH),
        _clip(player_y / ARENA_HEIGHT),
        _clip(_position(player, "velocity_x") / VELOCITY_X_SCALE),
        _clip(_position(player, "velocity_y") / VELOCITY_Y_SCALE),
        _clip((boss_x - player_x) / ARENA_WIDTH),
        _clip((boss_y - player_y) / ARENA_HEIGHT),
        _clip(_position(boss, "velocity_x") / VELOCITY_X_SCALE),
        _clip(_position(boss, "velocity_y") / VELOCITY_Y_SCALE),
        1.0 if _grounded(snapshot, player) else 0.0,
        resources.silk_normalized,
        *_one_hot(attack_type, ATTACK_TYPES),
        *_one_hot(attack_phase, ATTACK_PHASES),
        *_one_hot(reaction, REACTIONS),
        *_one_hot(phase_event, PHASE_EVENTS),
        *(key_state or KeyHoldState()).as_tuple(),
    )
    if len(values) != STATE_DIMENSIONS:
        raise AssertionError(f"state encoder produced {len(values)} values, expected {STATE_DIMENSIONS}")
    return StateFrame(
        observation=values,
        timestamp=_number(snapshot.get("timestamp")),
        frame=int(_number(snapshot.get("frame"))),
        scene=str(snapshot.get("scene") or ""),
        player=player,
        boss=boss,
        resources=resources,
        control_state=control_state,
        attack_type=attack_type,
        attack_phase=attack_phase,
        reaction=reaction,
        phase_event=phase_event,
    )


def decode_jsonl(lines: Iterable[str]) -> Iterator[StateFrame]:
    """Decode valid snapshot lines, ignoring lifecycle and malformed lines."""

    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and value.get("type") == "snapshot":
            yield encode_snapshot(value)


def read_jsonl(path: str | Path) -> Iterator[StateFrame]:
    """Read a completed telemetry file in sampling order."""

    with Path(path).open("r", encoding="utf-8-sig") as stream:
        yield from decode_jsonl(stream)


def read_available(stream: TextIO) -> Iterator[StateFrame]:
    """Decode currently available lines from an already-open append-only file."""

    yield from decode_jsonl(stream)
