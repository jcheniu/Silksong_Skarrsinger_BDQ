"""Decode joint-action vectors and apply simultaneous keyboard state."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import time
from typing import Mapping, Sequence

from .action_recorder import ActionRecorder, ChargeState
from ..real_state import KeyHoldState, PlayerResources, decode_player_resources


BRANCH_NAMES = (
    "jump_z",
    "movement",
    "combat",
)
BRANCH_SIZES = (3, 6, 6)
ACTION_PROTOCOL = "semantic-joint-v19-curated-53"
BranchMasks = tuple[tuple[bool, ...], ...]

KEYS = {
    "LeftArrow": (0x4B, True),
    "RightArrow": (0x4D, True),
    "UpArrow": (0x48, True),
    "DownArrow": (0x50, True),
    "Z": (0x2C, False),
    "C": (0x2E, False),
    "X": (0x2D, False),
    "S": (0x1F, False),
    "LeftShift": (0x2A, False),
    "D": (0x20, False),
}
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
SW_RESTORE = 9
SPI_GETWORKAREA = 0x0030


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


def branch_availability(
    snapshot: Mapping[str, object],
    continuing_action: Sequence[int] | None = None,
    harpoon_locked: bool = False,
    charge_protected: bool = False,
    charge_must_hold: bool = False,
) -> tuple[BranchMasks, tuple[str, ...]]:
    """Build conservative branch masks from optional resource/control telemetry."""

    masks = [[True] * size for size in BRANCH_SIZES]
    continuing = (
        validate_action(continuing_action)
        if continuing_action is not None
        else (0,) * len(BRANCH_SIZES)
    )
    reasons: list[str] = []
    resources = decode_player_resources(snapshot)
    if harpoon_locked:
        masks = [[index == 0 for index in range(size)] for size in BRANCH_SIZES]
        reasons.append("all branches masked: harpoon active/recovery lock")
        return validate_masks(masks), tuple(reasons)
    if not resources.can_harpoon_dash:
        masks[1][5] = False
        reasons.append("movement harpoon masked: CanHarpoonDash is false or unavailable")
    if charge_protected:
        masks[1][5] = False
        reasons.append("movement harpoon masked: charge hold/release is protected")
    if not resources.can_quick_cast:
        masks[2][3] = False
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
    dash_available = control_values.get("dash_available") is True
    for value in (3, 4):
        masks[1][value] = dash_available
    if not dash_available:
        reasons.append("dash masked: dash_available is false or unavailable")
    attack_available = control_values.get("attack_available") is True
    for value in (1, 4, 5):
        masks[2][value] = attack_available
    masks[2][2] = attack_available or continuing[2] == 2
    if not attack_available:
        reasons.append("attack start masked: attack_available is false or unavailable")
    if charge_must_hold:
        masks[2] = [index == 2 for index in range(BRANCH_SIZES[2])]
        reasons.append("combat branch locked: incomplete charge must keep holding X")
    return validate_masks(masks), tuple(reasons)


def action_keys(action: Sequence[int]) -> tuple[str, ...]:
    values = validate_action(action)
    keys: list[str] = []
    if values[0]:
        keys.append("Z")
    if values[1] in (1, 3):
        keys.append("LeftArrow")
    elif values[1] in (2, 4):
        keys.append("RightArrow")
    if values[1] in (3, 4):
        keys.append("C")
    elif values[1] == 5:
        keys.append("S")
    if values[2] in (1, 2):
        keys.append("X")
    elif values[2] == 4:
        keys.extend(("UpArrow", "X"))
    elif values[2] == 5:
        keys.extend(("DownArrow", "X"))
    elif values[2] == 3:
        keys.append("LeftShift")
    return tuple(keys)


def decode_actions(action: Sequence[int]) -> tuple[str, ...]:
    values = validate_action(action)
    names: list[str] = []
    if values[0] == 1:
        names.append("jump")
    elif values[0] == 2:
        names.append("jump_hold")
    if values[1] == 1:
        names.append("left")
    elif values[1] == 2:
        names.append("right")
    elif values[1] == 3:
        names.append("left_dash")
    elif values[1] == 4:
        names.append("right_dash")
    elif values[1] == 5:
        names.append("harpoon_dash")
    if values[2] == 1:
        names.append("attack")
    elif values[2] == 2:
        names.append("attack_charge")
    elif values[2] == 3:
        names.append("quick_cast")
    elif values[2] == 4:
        names.append("up_attack")
    elif values[2] == 5:
        names.append("down_attack")
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


def place_game_window_top_left_quarter(timeout_s: float = 60.0) -> int:
    """Place the restored game window in the top-left quarter of the work area."""

    hwnd = focus_game_window(timeout_s)
    user32 = ctypes.windll.user32
    work_area = wintypes.RECT()
    if not user32.SystemParametersInfoW(
        SPI_GETWORKAREA,
        0,
        ctypes.byref(work_area),
        0,
    ):
        raise ctypes.WinError()
    width = max(1, (work_area.right - work_area.left) // 2)
    height = max(1, (work_area.bottom - work_area.top) // 2)
    if not user32.MoveWindow(
        hwnd,
        work_area.left,
        work_area.top,
        width,
        height,
        True,
    ):
        raise ctypes.WinError()
    return hwnd


@dataclass
class KeyboardActionExecutor:
    """Maintain desired key state across fixed-duration control ticks."""

    recorder: ActionRecorder
    tick_ms: int = 50
    send_input: bool = False
    harpoon_active_ms: int = 300
    harpoon_lock_ms: int = 900
    charge_release_protection_ms: int = 500

    def __post_init__(self) -> None:
        if self.tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        if not 0 < self.harpoon_active_ms <= self.harpoon_lock_ms:
            raise ValueError("harpoon timing must satisfy 0 < active <= lock")
        self._held_keys: set[str] = set()
        self._charge = ChargeState()
        self._harpoon_lock_remaining_ms = 0
        self._charge_release_protection_remaining_ms = 0
        self._last_executed = (0,) * len(BRANCH_SIZES)
        self._interrupted = False
        if self.send_input:
            focus_game_window()

    def control_state(self, snapshot: Mapping[str, object]) -> KeyHoldState:
        del snapshot
        jump_value, movement_value, combat_value = self._last_executed
        movement_direction = (
            -1.0 if movement_value in (1, 3) else 1.0 if movement_value in (2, 4) else 0.0
        )
        movement_mode = (
            1.0 if movement_value == 5 else 0.5 if movement_value in (3, 4) else 0.0
        )
        harpoon_elapsed_ms = (
            self.harpoon_lock_ms - self._harpoon_lock_remaining_ms
            if self._harpoon_lock_remaining_ms > 0
            else 0
        )
        recovery_duration_ms = self.harpoon_lock_ms - self.harpoon_active_ms
        if self._harpoon_lock_remaining_ms <= 0:
            harpoon_phase = 0.0
        elif harpoon_elapsed_ms < self.harpoon_active_ms:
            harpoon_phase = 0.25
        elif recovery_duration_ms > 0:
            recovery_progress = (
                harpoon_elapsed_ms - self.harpoon_active_ms
            ) / recovery_duration_ms
            harpoon_phase = 0.25 + 0.75 * recovery_progress
        else:
            harpoon_phase = 1.0
        return KeyHoldState(
            jump_state=jump_value / 2.0,
            movement_direction=movement_direction,
            movement_mode=movement_mode,
            combat_action=combat_value / 5.0,
            attack_charge_progress=(
                self._charge.elapsed_ms / self._charge.max_hold_ms
            ),
            harpoon_phase=harpoon_phase,
        )

    @property
    def continuing_action(self) -> tuple[int, ...]:
        return self._last_executed

    @property
    def harpoon_locked(self) -> bool:
        return self._harpoon_lock_remaining_ms > 0

    @property
    def charge_protected(self) -> bool:
        return (
            self._charge.active
            or self._charge_release_protection_remaining_ms > 0
        )

    @property
    def charge_must_hold(self) -> bool:
        return (
            self._charge.active
            and self._charge.elapsed_ms < self._charge.required_ms
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
        charge_was_active = self._charge.active
        charge_was_complete = self._charge.elapsed_ms >= self._charge.required_ms
        charge_was_at_max = self._charge.elapsed_ms >= self._charge.max_hold_ms
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
        adjusted_reasons: list[str] = []
        harpoon_lock_was_active = self.harpoon_locked
        harpoon_started = False
        if harpoon_lock_was_active:
            if any(values):
                adjusted_reasons.append("harpoon lock forced all branches neutral")
            values = (0,) * len(BRANCH_SIZES)
        else:
            adjusted = list(values)
            if charge_was_active and not charge_was_complete and adjusted[2] != 2:
                adjusted[2] = 2
                adjusted_reasons.append(
                    "incomplete charge forced combat branch to keep holding X"
                )
            if adjusted[1] == 5 and (
                charge_was_active
                or adjusted[2] == 2
                or self._charge_release_protection_remaining_ms > 0
            ):
                adjusted[1] = 0
                adjusted_reasons.append(
                    "charge hold/release protection suppressed harpoon"
                )
            if adjusted[2] == 2 and charge_was_at_max:
                adjusted[2] = 0
                adjusted_reasons.append("maximum charge duration released combat branch")
            if adjusted[1] == 5:
                harpoon_started = True
                if adjusted[0] != 0 or adjusted[2] != 0:
                    adjusted_reasons.append(
                        "harpoon launch forced jump and combat branches neutral"
                    )
                adjusted[0] = 0
                adjusted[2] = 0
            values = tuple(adjusted)
        if interrupted:
            if any(values):
                adjusted_reasons.append("interruption forced all branches neutral")
            values = (0,) * len(BRANCH_SIZES)
            harpoon_started = False
        desired = set(action_keys(values))
        pulse_keys: set[str] = set()

        # The policy controls only generic Z key semantics. The same press and
        # hold sequence may become a ground jump, double jump, or cloak hover
        # depending entirely on the game's current state.
        if values[0] == 1:
            pulse_keys.add("Z")

        # A tap must be a real key pulse even when the policy repeats it on
        # consecutive ticks. Charge attacks deliberately remain held and are
        # tracked by ChargeState below.
        if values[2] in (1, 4, 5):
            pulse_keys.add("X")
        if values[1] in (3, 4):
            pulse_keys.add("C")
        if values[1] == 5:
            pulse_keys.add("S")
        if values[2] == 3:
            pulse_keys.add("LeftShift")
        held_before = set(self._held_keys)
        if self.send_input:
            for key in sorted(pulse_keys & self._held_keys):
                _send_key(key, True)
                self._held_keys.remove(key)
            for key in sorted(self._held_keys - desired):
                _send_key(key, True)
            for key in sorted(desired - self._held_keys):
                _send_key(key, False)
        self._held_keys = desired
        self._interrupted = interrupted
        if interrupted:
            self._harpoon_lock_remaining_ms = 0
            self._charge_release_protection_remaining_ms = 0
        elif harpoon_started:
            self._harpoon_lock_remaining_ms = max(0, self.harpoon_lock_ms - self.tick_ms)
        elif harpoon_lock_was_active:
            self._harpoon_lock_remaining_ms = max(
                0, self._harpoon_lock_remaining_ms - self.tick_ms
            )
        executed = tuple(values)
        recorded_actions = list(decode_actions(executed))
        newly_pressed_keys = tuple(sorted((desired - held_before) | pulse_keys))
        started_branches: list[str] = []
        charge_released = (
            not interrupted and charge_was_complete and values[2] != 2
        )
        if charge_released:
            self._charge_release_protection_remaining_ms = (
                self.charge_release_protection_ms
            )
        elif self._charge_release_protection_remaining_ms > 0:
            self._charge_release_protection_remaining_ms = max(
                0,
                self._charge_release_protection_remaining_ms - self.tick_ms,
            )
        if values[2] in (1, 4, 5) or charge_released:
            started_branches.append("attack_x")
        if harpoon_started:
            started_branches.append("skill_s")
        if "LeftShift" in newly_pressed_keys:
            started_branches.append("spell_shift")
        item = self.recorder.record_frame(
            recorded_actions,
            self.tick_ms,
            note=(
                f"policy vector={list(attempted)} executed={list(executed)} "
                f"protocol={ACTION_PROTOCOL}"
            ),
            interrupted=interrupted,
            charge=self._charge,
            action_vector=executed,
            attempted_action_vector=attempted,
            illegal_branches=illegal_branches,
            newly_pressed_keys=newly_pressed_keys,
            started_branches=started_branches,
            adjusted_reasons=adjusted_reasons,
            charge_released=charge_released,
            charge_pressed=values[2] == 2,
            branch_masks=masks,
            masked_reasons=masked_reasons,
            player_resources=(
                player_resources.as_dict() if player_resources is not None else None
            ),
        )
        item["temporal_owner"] = (
            "skill_s" if harpoon_started or harpoon_lock_was_active else None
        )
        self._last_executed = executed
        return item

    def release_all(self) -> None:
        if self.send_input:
            for key in sorted(self._held_keys):
                _send_key(key, True)
        self._held_keys.clear()
        self._harpoon_lock_remaining_ms = 0
        self._charge_release_protection_remaining_ms = 0
        self._last_executed = (0,) * len(BRANCH_SIZES)
        self._interrupted = True
        self._charge.step(False, self.tick_ms, interrupted=True)

    def close(self) -> None:
        self.release_all()
        self.recorder.close()
