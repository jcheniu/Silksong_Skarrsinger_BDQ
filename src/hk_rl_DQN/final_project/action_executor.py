"""Decode Branching-DQN actions and apply simultaneous keyboard state."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import time
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
BRANCH_SIZES = (3, 4, 3, 3, 2, 2, 2, 2)
ACTION_PROTOCOL = "branching-key-state-v5-executed-fragments"
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
    masks[6][1] = False
    reasons.append("dream_d disabled by policy")
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
    mobility_controls = (
        (1, (1, 2), "jump_available", "jump"),
        (1, (3,), "double_jump_available", "double_jump"),
        (2, (1,), "dash_available", "dash"),
        (3, (1, 2), "attack_available", "attack"),
    )
    for branch_index, values, field, action_name in mobility_controls:
        if control_values.get(field) is not True:
            for value in values:
                masks[branch_index][value] = False
            reasons.append(f"{action_name} masked: {field} is false or unavailable")
    sprint_available = control_values.get("sprint_available") is True
    sprinting = control_values.get("sprinting") is True
    if snapshot.get("player_grounded") is not True or not (sprint_available or sprinting):
        masks[2][2] = False
        reasons.append("sprint masked: grounded sprint is unavailable")
    return validate_masks(masks), tuple(reasons)


def action_keys(action: Sequence[int]) -> tuple[str, ...]:
    values = validate_action(action)
    keys: list[str] = []
    if values[0] == 1:
        keys.append("LeftArrow")
    elif values[0] == 2:
        keys.append("RightArrow")
    if values[1]:
        keys.append("Z")
    if values[2]:
        keys.append("C")
    if values[3]:
        keys.append("X")
    for enabled, key in zip(values[4:], ("S", "LeftShift", None, "V")):
        if enabled and key is not None:
            keys.append(key)
    return tuple(keys)


def decode_actions(action: Sequence[int]) -> tuple[str, ...]:
    values = validate_action(action)
    names: list[str] = []
    if values[0] == 1:
        names.append("left")
    elif values[0] == 2:
        names.append("right")
    if values[1] == 1:
        names.append("jump")
    elif values[1] == 2:
        names.append("jump_hold")
    elif values[1] == 3:
        names.append("double_jump")
    if values[2] == 1:
        names.append("dash")
    elif values[2] == 2:
        names.append("quick_run")
    if values[3] == 1:
        names.append("attack")
    elif values[3] == 2:
        names.append("attack_charge")
    for enabled, name in zip(
        values[4:],
        ("harpoon_dash", "quick_cast", None, "taunt"),
    ):
        if enabled and name is not None:
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


def focus_game_window(timeout_s: float = 60.0) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            hwnd = find_game_window()
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Silksong window was not found within {timeout_s:g} seconds"
                ) from None
            time.sleep(0.25)
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
    minimum_direction_hold_ms: int = 300
    minimum_run_hold_ms: int = 300

    def __post_init__(self) -> None:
        if self.tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        self._held_keys: set[str] = set()
        self._charge = ChargeState()
        self._attack_hold_ms = 0
        self._jump_hold_ms = 0
        self._dash_hold_ms = 0
        self._skill_hold_ms = 0
        self._horizontal_value = 0
        self._horizontal_hold_ms = 0
        self._run_hold_ms = 0
        self._jump_hold_remaining_ms = 0
        self._jump_mode = "released"
        self._interrupted = False
        if self.send_input:
            focus_game_window()

    def control_state(self, snapshot: Mapping[str, object]) -> KeyHoldState:
        masks, _reasons = branch_availability(snapshot)
        return KeyHoldState(
            jump_held="Z" in self._held_keys,
            jump_hold_progress=self._jump_hold_ms / 350.0,
            jump_available=masks[1][1] or masks[1][2],
            double_jump_available=masks[1][3],
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
        attempted = validate_action(action)
        masks = validate_masks(branch_masks) if branch_masks is not None else None
        illegal_branches: tuple[str, ...] = ()
        if masks is not None:
            illegal_branches = tuple(
                BRANCH_NAMES[index]
                for index, value in enumerate(attempted)
                if not masks[index][value]
            )
            values = tuple(
                value if masks[index][value] else 0
                for index, value in enumerate(attempted)
            )
        else:
            values = attempted
        values = self._smooth_mobility(values, masks, interrupted)
        desired = set(action_keys(values))
        pulse_keys: set[str] = set()

        # Jump intents are temporal fragments. A short jump and a double jump
        # are pulses; a hold jump keeps Z down for a minimum launch window and
        # can be renewed by selecting the intent on later ticks.
        if interrupted:
            self._jump_hold_remaining_ms = 0
            self._jump_mode = "released"
        elif values[1] == 1:
            self._jump_hold_remaining_ms = 0
            self._jump_mode = "short"
            pulse_keys.add("Z")
        elif values[1] == 2:
            self._jump_hold_remaining_ms = max(self._jump_hold_remaining_ms, 350)
            self._jump_mode = "hold"
        elif values[1] == 3:
            self._jump_hold_remaining_ms = 0
            self._jump_mode = "double"
            pulse_keys.add("Z")
        elif self._jump_hold_remaining_ms > 0:
            self._jump_mode = "hold"
        else:
            self._jump_mode = "released"

        if self._jump_hold_remaining_ms > 0:
            desired.add("Z")
            self._jump_hold_remaining_ms = max(0, self._jump_hold_remaining_ms - self.tick_ms)

        # A tap must be a real key pulse even when the policy repeats it on
        # consecutive ticks. Charge attacks deliberately remain held and are
        # tracked by ChargeState below.
        if values[3] == 1:
            pulse_keys.add("X")
        if interrupted:
            desired.clear()
        held_before = set(self._held_keys)
        if self.send_input:
            if values[1] in (1, 3) and "Z" in self._held_keys:
                _send_key("Z", True)
                self._held_keys.remove("Z")
            if "X" in pulse_keys and "X" in self._held_keys:
                _send_key("X", True)
                self._held_keys.remove("X")
            for key in sorted(self._held_keys - desired):
                _send_key(key, True)
            for key in sorted(desired - self._held_keys):
                _send_key(key, False)
        self._held_keys = desired
        self._interrupted = interrupted
        self._jump_hold_ms = self._jump_hold_ms + self.tick_ms if "Z" in desired else 0
        self._attack_hold_ms = self._attack_hold_ms + self.tick_ms if "X" in desired else 0
        self._dash_hold_ms = self._dash_hold_ms + self.tick_ms if "C" in desired else 0
        self._skill_hold_ms = self._skill_hold_ms + self.tick_ms if "S" in desired else 0
        executed_values = list(values)
        if interrupted:
            executed_values = [0] * len(BRANCH_SIZES)
        elif self._jump_mode == "hold" and "Z" in desired and values[1] == 0:
            executed_values[1] = 2
        executed = tuple(executed_values)
        recorded_actions = list(decode_actions(executed))
        newly_pressed_keys = tuple(sorted(desired - held_before))
        item = self.recorder.record_frame(
            recorded_actions,
            self.tick_ms,
            note=(
                f"policy vector={list(attempted)} executed={list(values)} "
                f"protocol={ACTION_PROTOCOL}"
            ),
            interrupted=interrupted,
            charge=self._charge,
            action_vector=executed,
            attempted_action_vector=attempted,
            illegal_branches=illegal_branches,
            newly_pressed_keys=newly_pressed_keys,
            charge_pressed=values[3] == 2,
            branch_masks=masks,
            masked_reasons=masked_reasons,
            player_resources=(
                player_resources.as_dict() if player_resources is not None else None
            ),
        )
        return item

    def _smooth_mobility(
        self,
        values: tuple[int, ...],
        masks: BranchMasks | None,
        interrupted: bool,
    ) -> tuple[int, ...]:
        smoothed = list(values)
        if interrupted:
            self._horizontal_value = 0
            self._horizontal_hold_ms = 0
            self._run_hold_ms = 0
            return tuple(smoothed)

        requested_horizontal = smoothed[0]
        if self._horizontal_value == 0:
            self._horizontal_value = requested_horizontal
            self._horizontal_hold_ms = self.tick_ms if requested_horizontal else 0
        elif requested_horizontal == self._horizontal_value:
            self._horizontal_hold_ms += self.tick_ms
        elif self._horizontal_hold_ms < self.minimum_direction_hold_ms:
            smoothed[0] = self._horizontal_value
            self._horizontal_hold_ms += self.tick_ms
        else:
            self._horizontal_value = requested_horizontal
            self._horizontal_hold_ms = self.tick_ms if requested_horizontal else 0

        run_allowed = masks is None or masks[2][2]
        if self._run_hold_ms and run_allowed and self._run_hold_ms < self.minimum_run_hold_ms:
            smoothed[2] = 2
            self._run_hold_ms += self.tick_ms
        elif smoothed[2] == 2 and run_allowed:
            self._run_hold_ms = self._run_hold_ms + self.tick_ms if self._run_hold_ms else self.tick_ms
        else:
            self._run_hold_ms = 0
        return tuple(smoothed)

    def release_all(self) -> None:
        if self.send_input:
            for key in sorted(self._held_keys):
                _send_key(key, True)
        self._held_keys.clear()
        self._attack_hold_ms = 0
        self._jump_hold_ms = 0
        self._dash_hold_ms = 0
        self._skill_hold_ms = 0
        self._horizontal_value = 0
        self._horizontal_hold_ms = 0
        self._run_hold_ms = 0
        self._jump_hold_remaining_ms = 0
        self._jump_mode = "released"
        self._interrupted = True
        self._charge.step(False, self.tick_ms, interrupted=True)

    def close(self) -> None:
        self.release_all()
        self.recorder.close()
