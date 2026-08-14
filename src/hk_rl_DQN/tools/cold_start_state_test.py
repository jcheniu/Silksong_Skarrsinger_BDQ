"""Cold-start Silksong and validate the live DQN state stream.

This is an explicit CLI acceptance test, not a pytest test. It launches the
game, reads only telemetry appended during this run, validates every arena
snapshot, writes a report, and closes the launched process unless requested
otherwise.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ..real_state import (
    ATTACK_PHASES,
    ATTACK_TYPES,
    KINEMATIC_STATE_DIMENSIONS,
    RESOURCE_STATE_DIMENSIONS,
    PHASE_EVENTS,
    REACTIONS,
    STATE_DIMENSIONS,
    encode_snapshot,
)


DEFAULT_GAME_EXE = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\Hollow Knight Silksong\Hollow Knight Silksong.exe"
)
DEFAULT_TELEMETRY = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight Silksong"
    r"\BepInEx\plugins\hollow-knight-rl-KarmelitaPractice\telemetry.jsonl"
)
ARENA_SCENE = "Memory_Ant_Queen"
WM_CLOSE = 0x0010
ATTACK_HINTS = ("slash", "cyclone", "throw", "spin", "grind", "spear", "jump", "dive")


def _close_process_window(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _lparam: int) -> bool:
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if process_id.value == process.pid:
            user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        return True

    user32.EnumWindows(callback_type(visit), 0)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


class AppendedJsonl:
    def __init__(self, path: Path, start_at_end: bool) -> None:
        self.path = path
        self.position = path.stat().st_size if start_at_end and path.exists() else 0
        self.pending = b""

    def read(self) -> list[tuple[str, dict[str, Any] | None]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.position:
            self.position = 0
            self.pending = b""
        with self.path.open("rb") as stream:
            stream.seek(self.position)
            chunk = stream.read()
            self.position = stream.tell()
        if not chunk:
            return []
        parts = (self.pending + chunk).split(b"\n")
        self.pending = parts.pop()
        rows: list[tuple[str, dict[str, Any] | None]] = []
        for raw in parts:
            text = raw.decode("utf-8-sig", errors="replace").rstrip("\r")
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                rows.append((text, None))
                continue
            rows.append((text, value if isinstance(value, dict) else None))
        return rows


def _one_hot_slices(observation: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    offset = KINEMATIC_STATE_DIMENSIONS + RESOURCE_STATE_DIMENSIONS
    groups = []
    for size in (len(ATTACK_TYPES), len(ATTACK_PHASES), len(REACTIONS), len(PHASE_EVENTS)):
        groups.append(observation[offset : offset + size])
        offset += size
    return tuple(groups)


def _validate_snapshot(
    raw: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    unmapped: set[str],
) -> Any | None:
    for field in ("player", "boss", "fsm"):
        if raw.get(field) is None:
            errors.append(f"frame {raw.get('frame')}: missing {field}")
            return None
    resources = raw.get("player_resources")
    if not isinstance(resources, dict):
        errors.append(f"frame {raw.get('frame')}: missing player_resources")
        return None
    for field in ("silk", "silk_max", "silk_parts", "skill_cost"):
        value = resources.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"frame {raw.get('frame')}: invalid player_resources.{field}")
            return None
    for field in ("silk_abilities_disabled", "skill_available", "spell_available"):
        if not isinstance(resources.get(field), bool):
            errors.append(f"frame {raw.get('frame')}: invalid player_resources.{field}")
            return None
    if resources["silk"] < 0 or resources["silk_max"] <= 0:
        errors.append(f"frame {raw.get('frame')}: invalid silk range")
        return None
    if resources["silk"] > resources["silk_max"]:
        errors.append(f"frame {raw.get('frame')}: silk exceeds effective maximum")
        return None
    if resources["spell_available"] and (
        resources["silk_abilities_disabled"]
        or resources["silk"] < resources["skill_cost"]
    ):
        errors.append(f"frame {raw.get('frame')}: inconsistent spell availability")
        return None
    controls = raw.get("player_control")
    if not isinstance(controls, dict):
        errors.append(f"frame {raw.get('frame')}: missing player_control")
        return None
    for field in ("jump_available", "dash_available", "attack_available"):
        if not isinstance(controls.get(field), bool):
            errors.append(f"frame {raw.get('frame')}: invalid player_control.{field}")
            return None
    try:
        state = encode_snapshot(raw)
    except Exception as error:
        errors.append(f"frame {raw.get('frame')}: encoder failed: {error}")
        return None
    if len(state.observation) != STATE_DIMENSIONS:
        errors.append(f"frame {state.frame}: dimension {len(state.observation)} != {STATE_DIMENSIONS}")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in state.observation):
        errors.append(f"frame {state.frame}: observation contains non-finite or out-of-range values")
    for group in _one_hot_slices(state.observation):
        if not math.isclose(sum(group), 1.0, abs_tol=1e-7):
            errors.append(f"frame {state.frame}: invalid one-hot group {group}")
    if not state.control_state:
        errors.append(f"frame {state.frame}: Boss Control FSM is missing or inactive")
    lowered = state.control_state.lower()
    if state.attack_type == "none" and any(hint in lowered for hint in ATTACK_HINTS):
        unmapped.add(state.control_state)
    if raw.get("player_grounded") is None:
        warning = "player_grounded is inferred from position/velocity, not read directly"
        if warning not in warnings:
            warnings.append(warning)
    if not state.resources.is_complete:
        errors.append(f"frame {state.frame}: decoded player resources are incomplete")
    return state


def run_test(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if not args.no_launch and not args.game_exe.exists():
        raise FileNotFoundError(args.game_exe)
    reader = AppendedJsonl(args.telemetry, start_at_end=True)
    process: subprocess.Popen[bytes] | None = None
    errors: list[str] = []
    warnings: list[str] = []
    states = []
    malformed_lines = 0
    session_started = False
    arena_started_at: float | None = None
    wall_deadline = time.monotonic() + args.timeout_s
    try:
        if not args.no_launch:
            process = subprocess.Popen([str(args.game_exe)], cwd=str(args.game_exe.parent))
            print(f"Started Silksong pid={process.pid}", flush=True)
        else:
            print("Reusing the running Silksong process", flush=True)

        while time.monotonic() < wall_deadline:
            for _text, raw in reader.read():
                if raw is None:
                    malformed_lines += 1
                    continue
                if raw.get("type") == "telemetry_start":
                    session_started = True
                    continue
                if raw.get("type") != "snapshot":
                    continue
                if not session_started and not args.no_launch:
                    continue
                if raw.get("scene") != ARENA_SCENE or not raw.get("encounter_active"):
                    continue
                if arena_started_at is None:
                    arena_started_at = time.monotonic()
                    print(
                        f"Entered {ARENA_SCENE}; lure the Boss during the "
                        f"{args.observe_s:.1f}s observation window.",
                        flush=True,
                    )
                state = _validate_snapshot(raw, errors, warnings, set())
                if state is not None:
                    states.append(state)

            if arena_started_at is not None and time.monotonic() - arena_started_at >= args.observe_s:
                break
            if process is not None and process.poll() is not None:
                errors.append(f"game exited early with code {process.returncode}")
                break
            time.sleep(0.05)
        else:
            errors.append(f"arena was not observed within {args.timeout_s:.1f}s")
    finally:
        if process is not None and not args.keep_game:
            _close_process_window(process)

    # Re-run the mapping check with a persistent set for the final report.
    unmapped: set[str] = set()
    attack_counts = Counter(state.attack_type for state in states)
    phase_counts = Counter(state.attack_phase for state in states)
    reaction_counts = Counter(state.reaction for state in states)
    event_counts = Counter(state.phase_event for state in states)
    control_states = Counter(state.control_state for state in states)
    for state in states:
        lowered = state.control_state.lower()
        if state.attack_type == "none" and any(hint in lowered for hint in ATTACK_HINTS):
            unmapped.add(state.control_state)

    gaps = [
        current.timestamp - previous.timestamp
        for previous, current in zip(states, states[1:])
        if current.timestamp >= previous.timestamp
    ]
    backward_frames = sum(
        current.frame < previous.frame for previous, current in zip(states, states[1:])
    )
    if malformed_lines:
        errors.append(f"{malformed_lines} malformed JSONL lines")
    if len(states) < args.min_snapshots:
        errors.append(f"only {len(states)} valid arena snapshots; expected at least {args.min_snapshots}")
    if backward_frames:
        errors.append(f"{backward_frames} frame-number regressions")
    if gaps and max(gaps) > args.max_gap_s:
        errors.append(f"maximum sampling gap {max(gaps):.3f}s exceeds {args.max_gap_s:.3f}s")
    if not any(name != "none" for name in attack_counts):
        errors.append("no Boss attack was classified")
    if len(control_states) < 2:
        errors.append("Boss Control FSM did not change state")
    if unmapped:
        errors.append(f"attack-like states were not mapped: {sorted(unmapped)}")

    report = {
        "passed": not errors,
        "state_dimensions": STATE_DIMENSIONS,
        "arena_snapshots": len(states),
        "silk_observed": {
            "minimum": min(
                (state.resources.silk for state in states if state.resources.silk is not None),
                default=None,
            ),
            "maximum": max(
                (state.resources.silk for state in states if state.resources.silk is not None),
                default=None,
            ),
        },
        "duration_s": round(states[-1].timestamp - states[0].timestamp, 3) if len(states) >= 2 else 0.0,
        "max_sampling_gap_s": round(max(gaps), 4) if gaps else None,
        "distinct_control_states": sorted(control_states),
        "attack_counts": dict(sorted(attack_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "reaction_counts": dict(sorted(reaction_counts.items())),
        "phase_event_counts": dict(sorted(event_counts.items())),
        "unmapped_attack_states": sorted(unmapped),
        "warnings": warnings,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, 0 if report["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start and validate live Silksong DQN state telemetry")
    parser.add_argument("--game-exe", type=Path, default=DEFAULT_GAME_EXE)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--report", type=Path, default=Path("runs/state_detection_cold_start_report.json"))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--observe-s", type=float, default=45.0)
    parser.add_argument("--min-snapshots", type=int, default=100)
    parser.add_argument("--max-gap-s", type=float, default=0.5)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--keep-game", action="store_true")
    args = parser.parse_args()
    report, exit_code = run_test(args)
    print(json.dumps(report, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
