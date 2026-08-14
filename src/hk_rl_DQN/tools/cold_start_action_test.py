"""Cold-start one-action game test with live BepInEx log synchronization.

This is deliberately a manual-observation tool: one process is started, one
action is sent after the configured challenge marker, and the process remains
open so the operator can observe the result.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import subprocess
import time
from pathlib import Path

from ..final_project.action_catalog import get_action
from ..final_project.action_recorder import ActionRecorder


KEYS = {
    "LeftArrow": (0x25, 0x4B, True),
    "RightArrow": (0x27, 0x4D, True),
    "Z": (0x5A, 0x2C, False),
    "X": (0x58, 0x2D, False),
    "C": (0x43, 0x2E, False),
    "S": (0x53, 0x1F, False),
    "D": (0x44, 0x20, False),
    "V": (0x56, 0x2F, False),
    "LeftShift": (0xA0, 0x2A, False),
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


def send_scan_code(scan_code: int, extended: bool, key_up: bool) -> None:
    flags = KEYEVENTF_SCANCODE
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if key_up:
        flags |= KEYEVENTF_KEYUP
    item = INPUT(type=INPUT_KEYBOARD, union=INPUT_UNION(ki=KEYBDINPUT(0, scan_code, flags, 0, 0)))
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
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        if "Hollow Knight Silksong" in title.value:
            matches.append(hwnd)
        return True

    callback = callback_type(visit)
    user32.EnumWindows(callback, 0)
    if not matches:
        raise RuntimeError("Silksong window was not found")
    return matches[0]


def focus_game_window(timeout_s: float = 5.0) -> int:
    user32 = ctypes.windll.user32
    hwnd = find_game_window()
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if user32.GetForegroundWindow() == hwnd:
            time.sleep(0.25)
            return hwnd
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.1)
    raise RuntimeError("Silksong window could not be focused; refusing to send input")


def send_key(key: str, duration_ms: int) -> int:
    if key == "Z+direction":
        hwnd = focus_game_window()
        _send_key_down("RightArrow")
        try:
            _send_key_down("Z")
            time.sleep(duration_ms / 1000.0)
        finally:
            _send_key_up("Z")
            _send_key_up("RightArrow")
        return hwnd
    if key not in KEYS:
        raise ValueError(f"unsupported test key: {key}")
    user32 = ctypes.windll.user32
    hwnd = focus_game_window()
    _vk, scan_code, extended = KEYS[key]
    _send_key_down(key)
    try:
        time.sleep(duration_ms / 1000.0)
    finally:
        _send_key_up(key)
    return hwnd


def _send_key_down(key: str) -> None:
    _vk, scan_code, extended = KEYS[key]
    send_scan_code(scan_code, extended, False)


def _send_key_up(key: str) -> None:
    _vk, scan_code, extended = KEYS[key]
    send_scan_code(scan_code, extended, True)


def wait_after_hit_settle(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def wait_for_marker(log_path: Path, marker: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    stream = None
    position = log_path.stat().st_size if log_path.exists() else 0
    try:
        while time.monotonic() < deadline:
            if not log_path.exists():
                time.sleep(0.05)
                continue
            size = log_path.stat().st_size
            if stream is None or size < position:
                if stream is not None:
                    stream.close()
                stream = log_path.open("r", encoding="utf-8", errors="replace")
                position = 0 if size < position else size
                stream.seek(position)
            line = stream.readline()
            if line:
                position = stream.tell()
                print(line.rstrip())
                if marker in line:
                    return line.rstrip()
            else:
                time.sleep(0.05)
    finally:
        if stream is not None:
            stream.close()
    raise TimeoutError(f"marker not found within {timeout_s}s: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("--duration-ms", type=int)
    parser.add_argument("--game-exe", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--marker", default="Karmelita challenge state: Challenge Complete")
    parser.add_argument("--intent-marker", default="Karmelita boss intent:")
    parser.add_argument("--hit-marker", default="Karmelita hero health changed:")
    parser.add_argument("--trigger", choices=("intent", "first_hit"), default="intent")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--observe-s", type=float, default=12.0)
    parser.add_argument("--output", type=Path, default=Path("runs/final_actions.jsonl"))
    parser.add_argument("--no-launch", action="store_true", help="Reuse an already running game instance")
    parser.add_argument("--reset-marker", default="Hornet died; restarting Karmelita encounter")
    parser.add_argument("--wait-reset", action="store_true", help="Wait for a death reset before the challenge marker")
    parser.add_argument("--post-hit-settle-s", type=float, default=0.0)
    args = parser.parse_args()

    spec = get_action(args.action)
    duration_ms = spec.min_hold_ms if args.duration_ms is None else args.duration_ms
    recorder = ActionRecorder(args.output)
    try:
        if not args.no_launch:
            subprocess.Popen([str(args.game_exe)], cwd=str(args.game_exe.parent))
        if args.wait_reset:
            reset_line = wait_for_marker(args.log, args.reset_marker, args.timeout_s)
            print(f"Matched reset marker: {reset_line}")
        matched = wait_for_marker(args.log, args.marker, args.timeout_s)
        print(f"Matched marker: {matched}")
        if args.trigger == "first_hit":
            trigger_line = wait_for_marker(args.log, args.hit_marker, args.timeout_s)
            print(f"Matched first-hit trigger: {trigger_line}")
            wait_after_hit_settle(args.post_hit_settle_s)
        else:
            trigger_line = wait_for_marker(args.log, args.intent_marker, args.timeout_s)
            print(f"Matched boss intent: {trigger_line}")
        if spec.key is None:
            command = recorder.record(args.action, duration_ms, note=f"after marker: {args.marker}")
            print(json.dumps(command, ensure_ascii=True))
            print("wait action: no key sent")
        else:
            sent_hwnd = send_key(spec.key, duration_ms)
            command = recorder.record(
                args.action,
                duration_ms,
                note=f"sent successfully after marker: {args.marker}; focused_hwnd={sent_hwnd}",
            )
            print(json.dumps(command, ensure_ascii=True))
            print(f"Sent {spec.key} for {duration_ms} ms to Silksong hwnd={sent_hwnd}; observe the game now.")
        time.sleep(args.observe_s)
        print(f"Observation window complete: {args.observe_s:.1f}s")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
