"""Decode Branching-DQN actions and apply simultaneous keyboard state."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Mapping, Sequence

from .action_recorder import ActionRecorder, ChargeState
from ..real_state import KeyHoldState, PlayerResources, decode_player_resources


BRANCH_NAMES = (
    "horizontal",
    "jump_z",
    "dash_c",
    "attack_x",
    "skill_s",
    "spell_shift",
    "dream_d",
    "taunt_v",
)
BRANCH_SIZES = (3, 2, 2, 2, 2, 2, 2, 2)
ACTION_PROTOCOL = "branching-key-state-v2-harpoon-silk"
BranchMasks = tuple[tuple[bool, ...], ...]

KEYS = {
    "LeftArrow": (0x4B, True),
    "RightArrow": (0x4D, True),
    "Z": (0x2C, False),
    "C": (0x2E, False),
    "X": (0x2D, False),
    "S": (0x1F, False),
    "LeftShift": (0x2A, False),
    "D": (0x20, False),
    "V": (0x2F, False),
}
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
SW_RESTORE = 9


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]


def validate_action(action: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in action)
    if len(values) != len(BRANCH_SIZES):
        raise ValueError(f"expected {len(BRANCH_SIZES)} action branches, got {len(values)}")
    for name, value, size in zip(BRANCH_NAMES, values, BRANCH_SIZES):
        if not 0 <= value < size:
            raise ValueError(f"{name} must be in [0, {size - 1}], got {value}")
    return values


def validate_masks(masks: Sequence[Sequence[bool]]) -> BranchMasks:
    values = tuple(tuple(bool(allowed) for allowed in branch) for branch in masks)
    if len(values) != len(BRANCH_SIZES):
        raise ValueError(f"expected {len(BRANCH_SIZES)} branch masks")
    for name, branch, size in zip(BRANCH_NAMES, values, BRANCH_SIZES):
        if len(branch) != size or not any(branch):
            raise ValueError(f"invalid {name} mask: {branch}")
    return values


def branch_availability(snapshot: Mapping[str, object]) -> tuple[BranchMasks, tuple[str, ...]]:
    """Build conservative branch masks from optional resource/control telemetry."""

    masks = [[True] * size for size in BRANCH_SIZES]
    reasons: list[str] = []
    resources = decode_player_resources(snapshot)
    if not resources.can_harpoon_dash:
        masks[4][1] = False
        reasons.append("skill_s held masked: CanHarpoonDash is false or unavailable")
    if not resources.can_quick_cast:
        masks[5][1] = False
        if not resources.is_complete:
            reasons.append("spell_shift held masked: player resource telemetry is incomplete")
        elif resources.silk_abilities_disabled:
            reasons.append("spell_shift held masked: silk abilities are disabled")
        elif (
            resources.silk is not None
            and resources.skill_cost is not None
            and resources.silk < resources.skill_cost
        ):
            reasons.append(
                f"spell_shift held masked: silk {resources.silk} < cost {resources.skill_cost}"
            )
        else:
            reasons.append("spell_shift held masked: quick cast control/cooldown is unavailable")

    controls = snapshot.get("player_control")
    control_values = controls if isinstance(controls, Mapping) else {}
    for branch_index, field in ((1, "jump_available"), (2, "dash_available"), (3, "attack_available")):
        if control_values.get(field) is False:
            masks[branch_index][1] = False
            reasons.append(f"{BRANCH_NAMES[branch_index]} held masked: {field}=false")
    return validate_masks(masks), tuple(reasons)


def action_keys(action: Sequence[int]) -> tuple[str, ...]:
    values = validate_action(action)
    keys: list[str] = []
    if values[0] == 1:
        keys.append("LeftArrow")
    elif values[0] == 2:
        keys.append("RightArrow")
    for enabled, key in zip(values[1:], ("Z", "C", "X", "S", "LeftShift", "D", "V")):
        if enabled:
            keys.append(key)
    return tuple(keys)


def decode_actions(action: Sequence[int]) -> tuple[str, ...]:
    values = validate_action(action)
    names: list[str] = []
    if values[0] == 1:
        names.append("left")
    elif values[0] == 2:
        names.append("right")
    for enabled, name in zip(
        values[1:],
        ("jump", "dash", "attack", "harpoon_dash", "quick_cast", "dreamnail", "taunt"),
    ):
        if enabled:
            names.append(name)
    return tuple(names or ("wait",))


def _send_key(key: str, key_up: bool) -> None:
    scan_code, extended = KEYS[key]
    flags = KEYEVENTF_SCANCODE
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if key_up:
        flags |= KEYEVENTF_KEYUP
    item = INPUT(
        type=INPUT_KEYBOARD,
        union=INPUT_UNION(ki=KEYBDINPUT(0, scan_code, flags, 0, 0)),
    )
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT))
    if sent != 1:
        raise ctypes.WinError()


def find_game_window() -> int:
    user32 = ctypes.windll.user32
    matches: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        if "Hollow Knight Silksong" in title.value:
            matches.append(hwnd)
        return True

    user32.EnumWindows(callback_type(visit), 0)
    if not matches:
        raise RuntimeError("Silksong window was not found")
    return matches[0]


def focus_game_window() -> int:
    hwnd = find_game_window()
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    return hwnd


@dataclass
class KeyboardActionExecutor:
    """Maintain desired key state across fixed-duration control ticks."""

    recorder: ActionRecorder
    tick_ms: int = 100
    send_input: bool = False

    def __post_init__(self) -> None:
        if self.tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        self._held_keys: set[str] = set()
        self._charge = ChargeState()
        self._attack_hold_ms = 0
        self._dash_hold_ms = 0
        self._skill_hold_ms = 0
        self._interrupted = False
        if self.send_input:
            focus_game_window()

    def control_state(self, snapshot: Mapping[str, object]) -> KeyHoldState:
        masks, _reasons = branch_availability(snapshot)
        return KeyHoldState(
            attack_held="X" in self._held_keys,
            attack_hold_progress=self._attack_hold_ms / 1350.0,
            dash_held="C" in self._held_keys,
            dash_hold_progress=self._dash_hold_ms / 300.0,
            skill_held="S" in self._held_keys,
            skill_hold_progress=self._skill_hold_ms / 900.0,
            interrupted=self._interrupted,
            skill_available=masks[4][1],
            spell_available=masks[5][1],
        )

    def apply(
        self,
        action: Sequence[int],
        interrupted: bool = False,
        branch_masks: Sequence[Sequence[bool]] | None = None,
        masked_reasons: Sequence[str] = (),
        player_resources: PlayerResources | None = None,
    ) -> dict[str, object]:
        values = validate_action(action)
        masks = validate_masks(branch_masks) if branch_masks is not None else None
        if masks is not None:
            values = tuple(
                value if masks[index][value] else 0
                for index, value in enumerate(values)
            )
        desired = set(action_keys(values))
        if interrupted:
            desired.clear()
        if self.send_input:
            for key in sorted(self._held_keys - desired):
                _send_key(key, True)
            for key in sorted(desired - self._held_keys):
                _send_key(key, False)
        self._held_keys = desired
        self._interrupted = interrupted
        self._attack_hold_ms = self._attack_hold_ms + self.tick_ms if "X" in desired else 0
        self._dash_hold_ms = self._dash_hold_ms + self.tick_ms if "C" in desired else 0
        self._skill_hold_ms = self._skill_hold_ms + self.tick_ms if "S" in desired else 0
        item = self.recorder.record_frame(
            decode_actions(values),
            self.tick_ms,
            note=f"policy vector={list(values)} protocol={ACTION_PROTOCOL}",
            interrupted=interrupted,
            charge=self._charge,
            action_vector=values,
            charge_pressed=bool(values[3]),
            branch_masks=masks,
            masked_reasons=masked_reasons,
            player_resources=(
                player_resources.as_dict() if player_resources is not None else None
            ),
        )
        return item

    def release_all(self) -> None:
        if self.send_input:
            for key in sorted(self._held_keys):
                _send_key(key, True)
        self._held_keys.clear()
        self._attack_hold_ms = 0
        self._dash_hold_ms = 0
        self._skill_hold_ms = 0
        self._interrupted = True
        self._charge.step(False, self.tick_ms, interrupted=True)

    def close(self) -> None:
        self.release_all()
        self.recorder.close()
