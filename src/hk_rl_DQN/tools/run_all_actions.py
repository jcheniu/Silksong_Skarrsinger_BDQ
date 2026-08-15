"""Run one cold-start test per action, in a deterministic order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ..final_project.action_executor import validate_action


# Every non-neutral value in the three semantic fields, plus one simultaneous
# joint-action smoke test. Ticks describe repeated policy selections, not new
# semantic actions invented by the test tool.
BDQ_ACTION_CASES: dict[str, tuple[tuple[int, int, int], int]] = {
    "jump_press": ((1, 0, 0), 1),
    "jump_hold": ((2, 0, 0), 8),
    "move_left": ((0, 1, 0), 1),
    "move_right": ((0, 2, 0), 1),
    "dash": ((0, 3, 0), 1),
    "left_dash": ((0, 4, 0), 1),
    "right_dash": ((0, 5, 0), 1),
    "attack": ((0, 0, 1), 1),
    "up_attack": ((0, 0, 5), 1),
    "down_attack": ((0, 0, 6), 1),
    "attack_charge": ((0, 0, 2), 14),
    "harpoon_dash": ((0, 6, 0), 1),
    "quick_cast": ((0, 0, 3), 1),
    "taunt_hold": ((0, 0, 4), 2),
    "combined_jump_right_dash_attack": ((2, 5, 1), 1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-exe", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/final_actions"))
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--observe-s", type=float, default=5)
    parser.add_argument("--actions", nargs="*", default=list(BDQ_ACTION_CASES))
    parser.add_argument("--tick-ms", type=int, default=100)
    parser.add_argument("--keep-game", action="store_true", help="Do not close the game between actions")
    parser.add_argument("--repeats", type=int, default=10, help="Cold starts per action")
    parser.add_argument("--reuse-game", action="store_true", help="Start once and reuse the game after death resets")
    parser.add_argument("--interval-s", type=float, default=1.0, help="Delay between repeated tests")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.interval_s < 0:
        raise ValueError("--interval-s must be non-negative")
    if args.tick_ms <= 0:
        raise ValueError("--tick-ms must be positive")
    unknown = sorted(set(args.actions) - set(BDQ_ACTION_CASES))
    if unknown:
        raise ValueError(f"unknown BDQ action cases: {unknown}")
    launched = False
    for action_name in args.actions:
        action_vector, ticks = BDQ_ACTION_CASES[action_name]
        validate_action(action_vector)
        for repeat in range(1, args.repeats + 1):
            output = args.output_dir / f"{action_name}_{repeat:02d}.jsonl"
            command = [
                sys.executable,
                "-u",
                "-m",
                "hk_rl_DQN.tools.cold_start_action_test",
                "--action-vector",
                *(str(value) for value in action_vector),
                "--ticks",
                str(ticks),
                "--tick-ms",
                str(args.tick_ms),
                "--trigger",
                "first_hit",
                "--timeout-s",
                str(args.timeout_s),
                "--observe-s",
                str(args.observe_s),
                "--game-exe",
                str(args.game_exe),
                "--log",
                str(args.log),
                "--output",
                str(output),
            ]
            if args.reuse_game and launched:
                command.extend(["--no-launch", "--wait-reset"])
            print(
                f"=== TEST {action_name} repeat {repeat}/{args.repeats} ===",
                flush=True,
            )
            completed = subprocess.run(command, check=False)
            result = {
                "action": action_name,
                "action_vector": list(action_vector),
                "ticks": ticks,
                "repeat": repeat,
                "exit_code": completed.returncode,
                "output": str(output),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
            launched = True
            time.sleep(args.interval_s)
            if not args.keep_game and not args.reuse_game:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", "Hollow Knight Silksong.exe"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(2)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
