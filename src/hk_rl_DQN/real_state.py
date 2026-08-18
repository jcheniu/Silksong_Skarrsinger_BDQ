"""Encode live Silksong telemetry into a compact DQN observation.

This module deliberately contains no network, reward, action, or training code.
It only maps one JSON telemetry snapshot to a fixed-size numeric observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping, TextIO


# These limits match the simulator's coordinate conventions. Real positions
# are normalized and clipped so a transient scene-loading value cannot explode
# the input range.
ARENA_CENTER_X = 148.5
ARENA_HALF_WIDTH = 18.0
VERTICAL_POSITION_SCALE = 30.0
RELATIVE_POSITION_SCALE = 30.0
VELOCITY_X_SCALE = 8.0
VELOCITY_Y_SCALE = 8.0
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
BOSS_CONTROL_STATES = (
    "A Rethrow Antic",
    "A Rethrow Antic 2",
    "Air Rethrow",
    "Air Throw",
    "Air Throw Antic",
    "Air Throw Slash",
    "Air Throw Slash 2",
    "Approach Block",
    "Battle Roar",
    "Battle Roar End",
    "Battle Start",
    "Block",
    "Cyclone 1",
    "Cyclone 2",
    "Cyclone 3",
    "Cyclone 4",
    "Cyclone Antic",
    "Cyclone Multihit",
    "Cyclone Recoil",
    "Dash",
    "Entry Fall",
    "Entry Land",
    "Hornet Dead",
    "Jump Antic",
    "Jump Launch",
    "Launch Antic",
    "Launch In",
    "Launch In Antic",
    "Launch Spin",
    "Launch Up",
    "Long Approach",
    "Long Evade",
    "Movement 1",
    "Movement 2",
    "Movement 3",
    "Movement 4",
    "Movement 5",
    "P2 Roar",
    "P2 Roar Antic",
    "Roar",
    "Roar Antic",
    "Slash 1",
    "Slash 2",
    "Slash 3",
    "Slash 4",
    "Slash 5",
    "Slash 6",
    "Slash 7",
    "Slash 8",
    "Slash 9",
    "Slash Antic",
    "Slash End",
    "Spin Antic",
    "Spin Attack",
    "Spin Attack Land",
    "Spin Multihit",
    "Spin Recoil",
    "Stun Air",
    "Stun Recover",
    "Stunned",
    "Throw 1",
    "Throw 2",
    "Throw Antic",
    "Throw Fall",
    "Throw Land",
)
KINEMATIC_STATE_DIMENSIONS = 10
RESOURCE_STATE_DIMENSIONS = 1
BOSS_SEMANTIC_FEATURES = (
    "behavior_progress",
    "attack_category",
    "aerial",
    "collision_risk",
    "vertical_intent",
    "hit_pattern",
    "boss_status",
)
BOSS_SEMANTIC_DIMENSIONS = len(BOSS_SEMANTIC_FEATURES)
BASE_STATE_DIMENSIONS = (
    KINEMATIC_STATE_DIMENSIONS
    + RESOURCE_STATE_DIMENSIONS
    + BOSS_SEMANTIC_DIMENSIONS
)
CONTROL_STATE_DIMENSIONS = 6
STATE_DIMENSIONS = BASE_STATE_DIMENSIONS + CONTROL_STATE_DIMENSIONS
COLLISION_RISK_INDEX = (
    KINEMATIC_STATE_DIMENSIONS
    + RESOURCE_STATE_DIMENSIONS
    + BOSS_SEMANTIC_FEATURES.index("collision_risk")
)


@dataclass(frozen=True)
class KeyHoldState:
    """Compressed previous-action state required by the joint-action protocol."""

    jump_state: float = 0.0
    movement_direction: float = 0.0
    movement_mode: float = 0.0
    combat_action: float = 0.0
    attack_charge_progress: float = 0.0
    harpoon_phase: float = 0.0

    def as_tuple(self) -> tuple[float, ...]:
        return (
            _clip(self.jump_state, 0.0, 1.0),
            _clip(self.movement_direction),
            _clip(self.movement_mode, 0.0, 1.0),
            _clip(self.combat_action, 0.0, 1.0),
            _clip(self.attack_charge_progress, 0.0, 1.0),
            _clip(self.harpoon_phase, 0.0, 1.0),
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
    boss_vulnerable: bool | None
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


_SEQUENCE_LENGTHS = {
    "slash": 9,
    "movement": 5,
    "cyclone": 4,
    "throw": 2,
    "air throw slash": 2,
    "a rethrow antic": 2,
}


def _sequence_progress(state: str, phase: str) -> float:
    """Encode a named FSM step as a normalized behavior timeline."""

    lowered = state.lower()
    if phase == "anticipation" or "antic" in lowered:
        return 0.1
    if phase == "recovery" or any(
        token in lowered for token in ("end", "recover", "land", "recoil")
    ):
        return 1.0
    match = re.search(r"\s(\d+)$", lowered)
    if match is None:
        active_behavior = any(
            token in lowered
            for token in (
                "approach",
                "movement",
                "evade",
                "dash",
                "jump",
                "launch",
                "entry",
                "fall",
                "throw",
                "slash",
                "spin",
                "cyclone",
                "block",
                "roar",
                "stun",
            )
        )
        return 0.5 if phase == "active" or active_behavior else 0.0
    step = int(match.group(1))
    length = next(
        (value for prefix, value in _SEQUENCE_LENGTHS.items() if lowered.startswith(prefix)),
        step,
    )
    if length <= 1:
        return 0.5
    return _clip(0.2 + 0.6 * (step - 1) / (length - 1), 0.2, 0.8)


def _boss_semantics(
    control_state: str,
    attack_type: str,
    attack_phase: str,
    reaction: str,
    phase_event: str,
    collision_risk: float,
) -> tuple[float, ...]:
    """Compress raw Boss FSM labels into continuous, interpretable features."""

    lowered = control_state.lower()
    if "air throw slash" in lowered:
        attack_category = 7
    else:
        attack_category = {
            "none": 0,
            "slash": 1,
            "cyclone": 2,
            "spin_attack": 3,
            "ground_throw": 4,
            "air_throw": 4,
            "rethrow": 5,
            "dash_grind": 6,
            "jump_attack": 6,
            "spear_slam": 6,
        }[attack_type]
    aerial = any(
        token in lowered
        for token in ("air ", "jump", "launch", "fall", "spin attack")
    )
    if any(token in lowered for token in ("fall", "land", "slam", "dive")):
        vertical_intent = -1.0
    elif any(token in lowered for token in ("jump", "launch", "up")):
        vertical_intent = 1.0
    else:
        vertical_intent = 0.0
    if attack_type in {"cyclone", "spin_attack"}:
        hit_pattern = 1.0
    elif "multihit" in lowered:
        hit_pattern = 0.5
    else:
        hit_pattern = 0.0
    unknown_state = bool(control_state) and control_state not in BOSS_CONTROL_STATES
    if reaction == "dead":
        boss_status = 1.0
    elif phase_event == "phase_transition":
        boss_status = 0.9
    elif phase_event == "roar":
        boss_status = 0.75
    elif reaction == "stunned":
        boss_status = 0.6
    elif reaction == "hit":
        boss_status = 0.45
    elif reaction == "blocked":
        boss_status = 0.3
    elif reaction == "evaded":
        boss_status = 0.15
    elif unknown_state:
        boss_status = -1.0
    else:
        boss_status = 0.0
    return (
        _sequence_progress(control_state, attack_phase),
        attack_category / 7.0,
        float(aerial),
        _clip(collision_risk, 0.0, 1.0),
        vertical_intent,
        hit_pattern,
        boss_status,
    )


def _collision_risk(
    relative_x: float,
    relative_y: float,
    relative_velocity_x: float,
    relative_velocity_y: float,
) -> float:
    """Expose close-range collision danger without adding an observation."""

    horizontal = max(0.0, 1.0 - abs(relative_x) / 8.0)
    vertical = max(0.0, 1.0 - abs(relative_y) / 6.0)
    proximity = horizontal * vertical
    if proximity <= 0.0:
        return 0.0
    distance = math.hypot(relative_x, relative_y)
    if distance <= 1e-6:
        closing_speed = 0.0
    else:
        closing_speed = max(
            0.0,
            -(relative_x * relative_velocity_x + relative_y * relative_velocity_y)
            / distance,
        )
    closing = closing_speed / (closing_speed + VELOCITY_X_SCALE)
    return _clip(proximity * (0.75 + 0.25 * closing), 0.0, 1.0)


def _position(entity: Mapping[str, object] | None, key: str) -> float:
    return _number(entity.get(key)) if entity is not None else 0.0


def _signed_squash(value: float, scale: float) -> float:
    """Preserve high-speed ordering without hard clipping transient motion."""

    return value / (abs(value) + scale) if value else 0.0


def _facing(entity: Mapping[str, object] | None) -> float:
    value = _position(entity, "facing")
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


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
    if "cyclone" in lowered:
        attack = "cyclone"
    elif "rethrow" in lowered or "re-throw" in lowered:
        attack = "rethrow"
    elif "air throw" in lowered or "airthrow" in lowered or "air sickle" in lowered:
        attack = "air_throw"
    elif "slash" in lowered and "spin" not in lowered:
        attack = "slash"
    elif lowered.startswith("throw") or "throw " in lowered:
        attack = "ground_throw"
    elif any(
        token in lowered
        for token in (
            "spin attack",
            "spin antic",
            "spin dash",
            "spin multihit",
            "spin recoil",
            "launch spin",
        )
    ) or lowered.startswith("launch"):
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
    if lowered == "battle start":
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
    if any("stunned" in state or state.startswith("stun") for state in lowered):
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
    raw_boss_vulnerable = snapshot.get("boss_vulnerable")
    boss_vulnerable = (
        raw_boss_vulnerable if isinstance(raw_boss_vulnerable, bool) else None
    )

    relative_x = boss_x - player_x
    relative_y = boss_y - player_y
    player_velocity_x = _position(player, "velocity_x")
    player_velocity_y = _position(player, "velocity_y")
    relative_velocity_x = _position(boss, "velocity_x") - player_velocity_x
    relative_velocity_y = _position(boss, "velocity_y") - player_velocity_y
    values = (
        _clip((player_x - ARENA_CENTER_X) / ARENA_HALF_WIDTH),
        _clip((player_y - GROUND_Y) / VERTICAL_POSITION_SCALE),
        _signed_squash(player_velocity_x, VELOCITY_X_SCALE),
        _signed_squash(player_velocity_y, VELOCITY_Y_SCALE),
        _clip(relative_x / RELATIVE_POSITION_SCALE),
        _clip(relative_y / RELATIVE_POSITION_SCALE),
        _signed_squash(relative_velocity_x, VELOCITY_X_SCALE),
        _signed_squash(relative_velocity_y, VELOCITY_Y_SCALE),
        1.0 if _grounded(snapshot, player) else 0.0,
        _facing(player),
        resources.silk_normalized,
        *_boss_semantics(
            control_state,
            attack_type,
            attack_phase,
            reaction,
            phase_event,
            _collision_risk(
                relative_x,
                relative_y,
                relative_velocity_x,
                relative_velocity_y,
            ),
        ),
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
        boss_vulnerable=boss_vulnerable,
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
