"""Callable combat actions for the final real-game test adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .action_catalog import ACTION_NAMES
from .action_recorder import ActionRecorder


ActionFunction = Callable[[ActionRecorder, int | None], dict[str, Any]]


def _send(recorder: ActionRecorder, name: str, duration_ms: int | None = None) -> dict[str, Any]:
    return recorder.record(name, duration_ms, note="function action")


def frame(recorder: ActionRecorder, actions: list[str], duration_ms: int = 50,
          interrupted: bool = False) -> dict[str, Any]:
    """Emit one decoded tensor/list frame; each listed intent is independent."""
    return recorder.record_frame(actions, duration_ms, note="composed control frame",
                                 interrupted=interrupted)


def left(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "left", duration_ms or 100)


def right(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "right", duration_ms or 100)


def wait(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "wait", duration_ms or 100)


def jump(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "jump", duration_ms or 80)


def dash(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "dash", duration_ms or 80)


def quick_run(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "quick_run", duration_ms or 600)


def attack(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "attack", duration_ms or 80)


def attack_charge(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "attack_charge", duration_ms or 1350)


def quick_cast(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "quick_cast", duration_ms or 80)


def harpoon_dash(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "harpoon_dash", duration_ms or 80)


def dreamnail(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "dreamnail", duration_ms or 80)


def taunt(recorder: ActionRecorder, duration_ms: int | None = None) -> dict[str, Any]:
    return _send(recorder, "taunt", duration_ms or 80)


ACTION_FUNCTIONS: dict[str, ActionFunction] = {
    name: globals()[name] for name in ACTION_NAMES
}
