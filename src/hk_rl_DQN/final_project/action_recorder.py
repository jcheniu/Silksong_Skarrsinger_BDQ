"""Record action commands for later real-game playback and DQN alignment."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path

from .action_catalog import ACTION_NAMES, get_action


@dataclass
class ChargeState:
    """Tracks X held across control frames; omission or interruption releases it."""
    required_ms: int = 1350
    elapsed_ms: int = 0
    active: bool = False

    def step(self, pressed: bool, duration_ms: int, interrupted: bool = False) -> dict[str, object]:
        if not pressed or interrupted:
            self.elapsed_ms = 0
            self.active = False
        else:
            self.elapsed_ms += max(0, duration_ms)
            self.active = True
        return {"charge_elapsed_ms": self.elapsed_ms, "charge_required_ms": self.required_ms,
                "charge_completed": self.elapsed_ms >= self.required_ms, "interrupted": interrupted}


class ActionRecorder:
    # Policy boundary: DQN emits [jump_z, movement, combat] once per control
    # tick. The keyboard adapter executes all three selected branch values
    # together. This recorder stores both the vector and its decoded actions.
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        self.sequence = 0

    def record_frame(
        self,
        actions: Sequence[str],
        duration_ms: int = 50,
        note: str = "",
        interrupted: bool = False,
        charge: ChargeState | None = None,
        action_vector: Sequence[int] | None = None,
        charge_pressed: bool | None = None,
        branch_masks: Sequence[Sequence[bool]] | None = None,
        masked_reasons: Sequence[str] = (),
        player_resources: Mapping[str, object] | None = None,
        attempted_action_vector: Sequence[int] | None = None,
        illegal_branches: Sequence[str] = (),
        newly_pressed_keys: Sequence[str] = (),
        started_branches: Sequence[str] = (),
        adjusted_reasons: Sequence[str] = (),
    ) -> dict[str, object]:
        names = tuple(dict.fromkeys(actions))
        if not names:
            names = ("wait",)
        specs = [get_action(action) for action in names]
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        charge_meta = None
        if charge is not None:
            pressed = "attack_charge" in names if charge_pressed is None else charge_pressed
            charge_meta = charge.step(pressed, duration_ms, interrupted)
        self.sequence += 1
        item = {
            "sequence": self.sequence,
            "timestamp": time.time(),
            "actions": list(names),
            "keys": [
                key
                for spec in specs
                for key in (
                    ()
                    if spec.key is None
                    else spec.key
                    if isinstance(spec.key, tuple)
                    else (spec.key,)
                )
            ],
            "duration_ms": duration_ms,
            "hold": duration_ms > 75,
            "consumes_silk": any(spec.consumes_silk for spec in specs),
            "interrupted": interrupted,
            "note": note,
        }
        if charge_meta is not None:
            item.update(charge_meta)
        if action_vector is not None:
            item["action_vector"] = [int(value) for value in action_vector]
        if attempted_action_vector is not None:
            item["attempted_action_vector"] = [
                int(value) for value in attempted_action_vector
            ]
        item["illegal_branches"] = list(illegal_branches)
        item["newly_pressed_keys"] = list(newly_pressed_keys)
        item["started_branches"] = list(started_branches)
        item["adjusted_reasons"] = list(adjusted_reasons)
        if branch_masks is not None:
            item["branch_masks"] = [
                [bool(allowed) for allowed in branch] for branch in branch_masks
            ]
        if masked_reasons:
            item["masked_reasons"] = list(masked_reasons)
        if player_resources is not None:
            item["player_resources"] = dict(player_resources)
        self.stream.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")
        return item

    def record(self, action: str, duration_ms: int | None = None, note: str = "") -> dict[str, object]:
        """Compatibility wrapper for recording one atomic action."""
        spec = get_action(action)
        duration = spec.min_hold_ms if duration_ms is None else duration_ms
        if duration < spec.min_hold_ms:
            raise ValueError(f"{action} requires at least {spec.min_hold_ms} ms")
        return self.record_frame([action], duration, note)

    def close(self) -> None:
        self.stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one combat action as JSONL")
    parser.add_argument("action", choices=ACTION_NAMES)
    parser.add_argument("--duration-ms", type=int)
    parser.add_argument("--note", default="")
    parser.add_argument("--output", default="runs/final_actions.jsonl")
    args = parser.parse_args()
    recorder = ActionRecorder(args.output)
    try:
        print(json.dumps(recorder.record(args.action, args.duration_ms, args.note), ensure_ascii=True))
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
