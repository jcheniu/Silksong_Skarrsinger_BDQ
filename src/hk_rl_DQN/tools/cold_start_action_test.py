"""Cold-start BDQ action test with live BepInEx log synchronization.

The tool accepts either a compatibility atomic action name or the exact
[jump_z, movement, combat] vector used by live training.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from ..final_project.action_catalog import get_action, get_action_vector
from ..final_project.action_executor import (
    BRANCH_SIZES,
    KeyboardActionExecutor,
    validate_action,
)
from ..final_project.action_recorder import ActionRecorder

ALL_ACTIONS_AVAILABLE = tuple(
    tuple(True for _ in range(size)) for size in BRANCH_SIZES
)


def execute_action_vector(
    executor: KeyboardActionExecutor,
    action: Sequence[int],
    ticks: int,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, object]]:
    """Execute exactly the same fixed-tick action path used by live training."""

    values = validate_action(action)
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    frames: list[dict[str, object]] = []
    try:
        for _ in range(ticks):
            frames.append(
                executor.apply(values, branch_masks=ALL_ACTIONS_AVAILABLE)
            )
            sleep(executor.tick_ms / 1000.0)
    finally:
        frames.append(
            executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
        )
    return frames


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
    parser.add_argument("action", nargs="?")
    parser.add_argument(
        "--action-vector",
        nargs=3,
        type=int,
        metavar=("JUMP_Z", "MOVEMENT", "COMBAT"),
    )
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--tick-ms", type=int, default=100)
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

    if (args.action is None) == (args.action_vector is None):
        parser.error("provide either an atomic action or --action-vector")
    if args.tick_ms <= 0:
        parser.error("--tick-ms must be positive")
    if args.ticks is not None and args.ticks <= 0:
        parser.error("--ticks must be positive")
    if args.action_vector is not None:
        if args.duration_ms is not None:
            parser.error("--duration-ms is only valid with an atomic action")
        action_vector = validate_action(args.action_vector)
        ticks = args.ticks or 1
        label = "vector_" + "_".join(str(value) for value in action_vector)
    else:
        assert args.action is not None
        if args.ticks is not None and args.duration_ms is not None:
            parser.error("use either --ticks or --duration-ms, not both")
        spec = get_action(args.action)
        action_vector = get_action_vector(args.action)
        duration_ms = spec.min_hold_ms if args.duration_ms is None else args.duration_ms
        if duration_ms < spec.min_hold_ms:
            parser.error(f"{args.action} requires at least {spec.min_hold_ms} ms")
        ticks = args.ticks or max(1, math.ceil(duration_ms / args.tick_ms))
        label = args.action
    recorder = ActionRecorder(args.output)
    executor: KeyboardActionExecutor | None = None
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
        executor = KeyboardActionExecutor(
            recorder,
            tick_ms=args.tick_ms,
            send_input=True,
        )
        frames = execute_action_vector(executor, action_vector, ticks)
        print(json.dumps(frames, ensure_ascii=True))
        print(
            f"Executed {label}: vector={list(action_vector)}, ticks={ticks}, "
            f"tick_ms={args.tick_ms}; observe the game now."
        )
        time.sleep(args.observe_s)
        print(f"Observation window complete: {args.observe_s:.1f}s")
    finally:
        if executor is not None:
            executor.close()
        else:
            recorder.close()


if __name__ == "__main__":
    main()
