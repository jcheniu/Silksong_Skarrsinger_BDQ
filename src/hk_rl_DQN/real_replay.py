"""Replay transitions, symmetry augmentation, and replay serialization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Mapping, Sequence

import torch

from .final_project.action_executor import validate_action
from .real_actions import (
    JOINT_ACTION_COUNT,
    decode_joint_action,
    joint_action_id,
)
from .real_state import STATE_DIMENSIONS


REPLAY_CHECKPOINT_VERSION = 3
REPLAY_CAPACITY = 50_000


@dataclass(frozen=True)
class Transition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    next_action_mask: tuple[bool, ...]

    @property
    def action_vector(self) -> tuple[int, ...]:
        return decode_joint_action(self.action)


_MIRRORED_STATE_SIGN_INDICES = (0, 2, 4, 6, 9, 19)


def mirror_observation(observation: Sequence[float]) -> tuple[float, ...]:
    """Reflect an observation across the arena center line."""

    if len(observation) != STATE_DIMENSIONS:
        raise ValueError(f"expected {STATE_DIMENSIONS} state values")
    values = [float(value) for value in observation]
    for index in _MIRRORED_STATE_SIGN_INDICES:
        values[index] = -values[index]
    return tuple(values)


def mirror_action(action: Sequence[int]) -> tuple[int, ...]:
    jump, movement, combat = validate_action(action)
    movement = {1: 2, 2: 1, 3: 4, 4: 3}.get(movement, movement)
    return jump, movement, combat


def mirror_transition(transition: Transition) -> Transition:
    mirrored_mask = [False] * JOINT_ACTION_COUNT
    for action_id, allowed in enumerate(transition.next_action_mask):
        if allowed:
            mirrored_mask[joint_action_id(mirror_action(decode_joint_action(action_id)))] = True
    return Transition(
        state=mirror_observation(transition.state),
        action=joint_action_id(mirror_action(transition.action_vector)),
        reward=transition.reward,
        next_state=mirror_observation(transition.next_state),
        done=transition.done,
        next_action_mask=tuple(mirrored_mask),
    )


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        if len(transition.state) != STATE_DIMENSIONS:
            raise ValueError("invalid transition state dimensions")
        if len(transition.next_state) != STATE_DIMENSIONS:
            raise ValueError("invalid transition next-state dimensions")
        decode_joint_action(transition.action)
        if len(transition.next_action_mask) != JOINT_ACTION_COUNT:
            raise ValueError("invalid joint action mask dimensions")
        if not any(transition.next_action_mask):
            raise ValueError("joint action mask must allow at least one action")
        self._items.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        if batch_size > len(self._items):
            raise ValueError("batch size exceeds replay size")
        return rng.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def state_dict(self) -> dict[str, object]:
        items = list(self._items)
        return {
            "version": REPLAY_CHECKPOINT_VERSION,
            "capacity": self._items.maxlen,
            "states": torch.tensor(
                [item.state for item in items], dtype=torch.float32
            ).reshape(-1, STATE_DIMENSIONS),
            "actions": torch.tensor(
                [item.action for item in items], dtype=torch.int16
            ).reshape(-1),
            "rewards": torch.tensor(
                [item.reward for item in items], dtype=torch.float32
            ),
            "next_states": torch.tensor(
                [item.next_state for item in items], dtype=torch.float32
            ).reshape(-1, STATE_DIMENSIONS),
            "dones": torch.tensor([item.done for item in items], dtype=torch.bool),
            "next_action_masks": torch.tensor(
                [item.next_action_mask for item in items], dtype=torch.bool
            ).reshape(-1, JOINT_ACTION_COUNT),
        }

    def load_state_dict(self, data: Mapping[str, object]) -> None:
        if data.get("version") != REPLAY_CHECKPOINT_VERSION:
            raise ValueError("checkpoint replay version mismatch")
        if data.get("capacity") != self._items.maxlen:
            raise ValueError("checkpoint replay capacity mismatch")
        required = (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
            "next_action_masks",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"checkpoint replay fields are missing: {missing}")
        states = torch.as_tensor(data.get("states")).cpu()
        actions = torch.as_tensor(data.get("actions")).cpu()
        rewards = torch.as_tensor(data.get("rewards")).cpu()
        next_states = torch.as_tensor(data.get("next_states")).cpu()
        dones = torch.as_tensor(data.get("dones")).cpu()
        raw_masks = data.get("next_action_masks")
        if raw_masks is None:
            raise ValueError("checkpoint replay masks are missing")
        masks = torch.as_tensor(raw_masks).cpu()
        size = int(states.shape[0]) if states.ndim == 2 else -1
        expected_shapes = (
            states.shape == (size, STATE_DIMENSIONS),
            actions.shape == (size,),
            rewards.shape == (size,),
            next_states.shape == (size, STATE_DIMENSIONS),
            dones.shape == (size,),
            masks.shape == (size, JOINT_ACTION_COUNT),
        )
        if not all(expected_shapes):
            raise ValueError("checkpoint replay tensor dimensions are invalid")
        if size > int(self._items.maxlen or 0):
            raise ValueError("checkpoint replay exceeds configured capacity")
        if not all(
            torch.isfinite(values).all().item()
            for values in (states, rewards, next_states)
        ):
            raise ValueError("checkpoint replay contains non-finite values")
        self.clear()
        for index in range(size):
            self.append(
                Transition(
                    state=tuple(float(value) for value in states[index].tolist()),
                    action=int(actions[index].item()),
                    reward=float(rewards[index].item()),
                    next_state=tuple(
                        float(value) for value in next_states[index].tolist()
                    ),
                    done=bool(dones[index].item()),
                    next_action_mask=tuple(
                        bool(value) for value in masks[index].tolist()
                    ),
                )
            )
