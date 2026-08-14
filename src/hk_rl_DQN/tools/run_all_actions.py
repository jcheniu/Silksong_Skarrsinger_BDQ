"""Run one cold-start test per action, in a deterministic order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ..final_project.action_catalog import ACTION_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-exe", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/final_actions"))
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--observe-s", type=float, default=5)
    parser.add_argument("--actions", nargs="*", default=list(ACTION_NAMES))
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
    launched = False
    for action in args.actions:
      for repeat in range(1, args.repeats + 1):
        output = args.output_dir / f"{action}_{repeat:02d}.jsonl"
        command = [
            sys.executable, "-u", "-m",
            "hk_rl_DQN.tools.cold_start_action_test", action,
            "--trigger", "first_hit", "--timeout-s", str(args.timeout_s),
            "--observe-s", str(args.observe_s), "--game-exe", str(args.game_exe),
            "--log", str(args.log), "--output", str(output),
        ]
        if args.reuse_game and launched:
            command.extend(["--no-launch", "--wait-reset"])
        print(f"=== TEST {action} repeat {repeat}/{args.repeats} ===", flush=True)
        completed = subprocess.run(command, check=False)
        result = {"action": action, "repeat": repeat, "exit_code": completed.returncode, "output": str(output)}
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
