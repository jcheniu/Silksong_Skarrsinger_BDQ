"""Write action commands for the BepInEx real-game adapter.

The adapter consumes one JSON object per line. This keeps action transport
independent from the DQN and makes every command auditable and replayable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .action_catalog import ACTION_NAMES, action_index


class ActionCommandWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def send(self, action: int | str, duration_ms: int = 50) -> dict[str, object]:
        name = ACTION_NAMES[action] if isinstance(action, int) else action
        action_index(name)
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        command = {
            "timestamp": time.time(),
            "action": name,
            "action_id": action_index(name),
            "duration_ms": duration_ms,
            "hold": duration_ms > 75,
        }
        self._stream.write(json.dumps(command, separators=(",", ":")) + "\n")
        return command

    def close(self) -> None:
        self._stream.close()
