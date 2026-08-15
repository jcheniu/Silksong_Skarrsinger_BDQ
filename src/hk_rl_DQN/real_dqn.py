"""Live joint-action Double DQN for the Silksong telemetry/action pipeline."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .final_project.action_executor import (
    ACTION_PROTOCOL,
    BRANCH_NAMES,
    BRANCH_SIZES,
    BranchMasks,
    KeyboardActionExecutor,
    branch_availability,
    find_game_window,
    validate_action,
    validate_masks,
)
from .final_project.action_recorder import ActionRecorder
from .real_reward import DODGE_REWARD, ILLEGAL_ACTION_PENALTY, RewardFrame, RewardTracker
from .real_state import STATE_DIMENSIONS, StateFrame, encode_snapshot


STATE_ENCODING = "real-telemetry-state-v12-charge-window-24"
ALGORITHM = "joint-dueling-double-dqn"
CHECKPOINT_VERSION = 22
REPLAY_CHECKPOINT_VERSION = 2
REWARD_PROTOCOL = "event-ledger-joint-action-v16"
HIDDEN_DIMENSIONS = (96, 96)
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
REPLAY_WARMUP = 1_000
PURE_EXPLORATION_STEPS = 2_000
TARGET_UPDATE_INTERVAL = 500
EPSILON_START = 0.60
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 15_000
EXPLORATION_ACTIVATION_RATES = (0.45, 0.85, 0.30)
MOVEMENT_EXPLORATION_WEIGHTS = (0.0, 32.0, 32.0, 12.0, 8.0, 8.0, 8.0)
COMBAT_EXPLORATION_WEIGHTS = (0.0, 30.0, 8.0, 8.0, 1.0, 20.0, 20.0)
ACTION_LABELS = (
    ("released", "press_z", "hold_z"),
    (
        "neutral",
        "hold_left",
        "hold_right",
        "dash",
        "left_dash",
        "right_dash",
        "harpoon_s",
    ),
    ("neutral", "tap_x", "hold_x", "shift", "taunt_v", "up_x", "down_x"),
)
COMBAT_ACTION_HURT_PENALTY_PER_HP = -3.0
EVADE_SUCCESS_REWARD = 0.8
EVADE_FAILURE_PENALTY = -1.0
COMBAT_HURT_CREDIT_WINDOW_STEPS = 8
CREDIT_FINALIZATION_STEPS = DAMAGE_CREDIT_WINDOW_STEPS = 20
TAUNT_OUTCOME_WINDOW_STEPS = 6
TAUNT_STEP_PENALTY = -0.02
TAUNT_MISS_PENALTY = 0.0
TAUNT_HURT_PENALTY = -1.0
OFFENSIVE_MISS_PENALTY = -0.5
GRADIENT_CLIP_NORM = 10.0
BRANCH_INDEX = {name: index for index, name in enumerate(BRANCH_NAMES)}
JOINT_ACTIONS = tuple(product(*(range(size) for size in BRANCH_SIZES)))
JOINT_ACTION_INDEX = {action: index for index, action in enumerate(JOINT_ACTIONS)}
JOINT_ACTION_COUNT = len(JOINT_ACTIONS)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "real_dqn.pt"
DEFAULT_METRICS = PROJECT_ROOT / "runs" / "real_dqn.jsonl"
DEFAULT_ACTION_LOG = PROJECT_ROOT / "runs" / "real_dqn_actions.jsonl"
DEFAULT_CONTROL_TICK_MS = 100
GAME_RELAUNCH_DELAY_SECONDS = 2.0
GAME_RELAUNCH_WINDOW_SECONDS = 60.0
MAX_GAME_RELAUNCHES_PER_WINDOW = 3
DEFAULT_GAME_EXE = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\Hollow Knight Silksong\Hollow Knight Silksong.exe"
)
DEFAULT_TELEMETRY = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight Silksong"
    r"\BepInEx\plugins\hollow-knight-rl-KarmelitaPractice\telemetry.jsonl"
)
ARENA_SCENE = "Memory_Ant_Queen"


class JointDQN(nn.Module):
    """Dueling Q network over all legal simultaneous action combinations."""

    def __init__(
        self,
        state_dimensions: int = STATE_DIMENSIONS,
        action_count: int = JOINT_ACTION_COUNT,
        hidden_dimensions: Sequence[int] = HIDDEN_DIMENSIONS,
    ) -> None:
        super().__init__()
        dimensions = (state_dimensions, *hidden_dimensions)
        layers: list[nn.Module] = []
        for input_size, output_size in zip(dimensions, dimensions[1:]):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        self.shared = nn.Sequential(*layers)
        feature_size = dimensions[-1]
        self.value = nn.Linear(feature_size, 1)
        self.advantage = nn.Linear(feature_size, action_count)
        self.action_count = int(action_count)

    def forward(self, states: Tensor) -> Tensor:
        features = self.shared(states)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


BranchingDQN = JointDQN


def joint_action_id(action: Sequence[int]) -> int:
    return JOINT_ACTION_INDEX[validate_action(action)]


def decode_joint_action(action_id: int) -> tuple[int, ...]:
    if not 0 <= int(action_id) < JOINT_ACTION_COUNT:
        raise ValueError(f"joint action must be in [0, {JOINT_ACTION_COUNT - 1}]")
    return JOINT_ACTIONS[int(action_id)]


def joint_action_mask(branch_masks: BranchMasks) -> tuple[bool, ...]:
    masks = validate_masks(branch_masks)
    return tuple(
        all(masks[index][value] for index, value in enumerate(action))
        for action in JOINT_ACTIONS
    )


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


@dataclass
class TauntTrial:
    pending: "PendingTransition"
    remaining_steps: int = TAUNT_OUTCOME_WINDOW_STEPS


@dataclass
class ActionOutcomeTrial:
    pending: "PendingTransition"
    action_kind: str
    penalize_miss: bool = True
    remaining_steps: int = DAMAGE_CREDIT_WINDOW_STEPS
    hit: bool = False


@dataclass
class CombatRiskTrial:
    pending: "PendingTransition"
    remaining_steps: int = COMBAT_HURT_CREDIT_WINDOW_STEPS


@dataclass
class PendingTransition:
    transition: Transition
    created_step: int
    attack_ids: set[int] = field(default_factory=set)
    delayed_reward: float = 0.0

    def add_reward(self, value: float) -> None:
        self.delayed_reward += float(value)

    def finalize(self) -> Transition:
        return replace(
            self.transition,
            reward=self.transition.reward + self.delayed_reward,
        )


@dataclass
class BossAttackCreditWindow:
    attack_id: int
    attack_type: str = "unknown"
    active_seen: bool = False
    hurt_player: bool = False
    transitions: list[PendingTransition] = field(default_factory=list)


@dataclass
class ActionExplorationState:
    """Short movement commitment used only for exploratory left/right actions."""

    sticky_movement: int = 0
    remaining_ticks: int = 0

    def consume(self, mask: Sequence[bool]) -> int | None:
        if self.remaining_ticks <= 0 or self.sticky_movement not in (1, 2):
            self.clear()
            return None
        if not mask[self.sticky_movement]:
            self.clear()
            return None
        value = self.sticky_movement
        self.remaining_ticks -= 1
        if self.remaining_ticks == 0:
            self.sticky_movement = 0
        return value

    def start(self, movement: int, additional_ticks: int) -> None:
        self.sticky_movement = movement
        self.remaining_ticks = additional_ticks

    def reconcile(self, executed_movement: int) -> None:
        if self.sticky_movement and executed_movement != self.sticky_movement:
            self.clear()

    def clear(self) -> None:
        self.sticky_movement = 0
        self.remaining_ticks = 0


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


def epsilon_for_step(step: int) -> float:
    fraction = min(1.0, max(0, step) / EPSILON_DECAY_STEPS)
    if fraction >= 1.0:
        return EPSILON_END
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def _weighted_exploration_choice(
    candidates: Sequence[int],
    weights: Sequence[float],
    rng: random.Random,
) -> int:
    total_weight = sum(weights[index] for index in candidates)
    if total_weight <= 0:
        return rng.choice(list(candidates))
    threshold = rng.random() * total_weight
    selected = candidates[-1]
    for candidate in candidates:
        threshold -= weights[candidate]
        if threshold < 0:
            selected = candidate
            break
    return selected


def select_action(
    network: JointDQN,
    observation: Sequence[float],
    epsilon: float,
    rng: random.Random,
    device: torch.device,
    branch_masks: BranchMasks | None = None,
    exploration_state: ActionExplorationState | None = None,
    selected_q_values: list[float] | None = None,
) -> tuple[int, ...]:
    if len(observation) != STATE_DIMENSIONS:
        raise ValueError(f"expected {STATE_DIMENSIONS} state values")
    with torch.no_grad():
        values = network(
            torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        )
    masks = (
        validate_masks(branch_masks)
        if branch_masks is not None
        else tuple(
            tuple(True for _ in range(size)) for size in BRANCH_SIZES
        )
    )
    joint_mask = joint_action_mask(masks)
    sticky_movement = (
        exploration_state.consume(masks[BRANCH_INDEX["movement"]])
        if exploration_state is not None
        else None
    )
    explore = rng.random() < epsilon
    if explore:
        action = []
        for branch_index, (size, mask) in enumerate(zip(BRANCH_SIZES, masks)):
            available = [index for index, allowed in enumerate(mask) if allowed]
            if not available:
                raise ValueError("every action branch must have an available value")
            if branch_index == BRANCH_INDEX["movement"] and sticky_movement is not None:
                action.append(sticky_movement)
                continue
            non_neutral = [index for index in available if index != 0]
            activation_rate = EXPLORATION_ACTIVATION_RATES[branch_index]
            if non_neutral and rng.random() < activation_rate:
                if branch_index == BRANCH_INDEX["movement"]:
                    selected = _weighted_exploration_choice(
                        non_neutral, MOVEMENT_EXPLORATION_WEIGHTS, rng
                    )
                    action.append(selected)
                    if exploration_state is not None and selected in (1, 2):
                        exploration_state.start(selected, rng.choice([1, 2]))
                elif branch_index == BRANCH_INDEX["combat"]:
                    action.append(
                        _weighted_exploration_choice(
                            non_neutral, COMBAT_EXPLORATION_WEIGHTS, rng
                        )
                    )
                else:
                    action.append(rng.choice(non_neutral))
            else:
                action.append(0 if 0 in available else rng.choice(available))
        selected_action = tuple(action)
        selected_id = joint_action_id(selected_action)
    else:
        scores = values.squeeze(0).clone()
        legal = [
            allowed
            and (sticky_movement is None or action[BRANCH_INDEX["movement"]] == sticky_movement)
            for action, allowed in zip(JOINT_ACTIONS, joint_mask)
        ]
        scores[torch.tensor([not allowed for allowed in legal], device=device)] = -torch.inf
        selected_id = int(scores.argmax().item())
        selected_action = decode_joint_action(selected_id)
    if selected_q_values is not None:
        selected_q_values.append(
            float(values[0, selected_id].detach().cpu().item())
        )
    return selected_action


def optimize_model(
    online: JointDQN,
    target: JointDQN,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
) -> float:
    if not transitions:
        raise ValueError("transitions must not be empty")
    states = torch.tensor([item.state for item in transitions], dtype=torch.float32, device=device)
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long, device=device)
    rewards = torch.tensor(
        [item.reward for item in transitions], dtype=torch.float32, device=device
    )
    next_states = torch.tensor(
        [item.next_state for item in transitions], dtype=torch.float32, device=device
    )
    dones = torch.tensor([item.done for item in transitions], dtype=torch.bool, device=device)

    online_values = online(states)
    selected = online_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        online_next = online(next_states)
        next_masks = torch.tensor(
            [item.next_action_mask for item in transitions],
            dtype=torch.bool,
            device=device,
        )
        next_actions = online_next.masked_fill(~next_masks, -torch.inf).argmax(
            dim=1, keepdim=True
        )
        target_next = target(next_states)
        next_values = target_next.gather(1, next_actions).squeeze(1)
        expected = rewards + GAMMA * next_values * (~dones)

    loss = F.smooth_l1_loss(selected, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    return float(loss.detach().cpu().item())


def checkpoint_metadata(
    global_step: int,
    episodes: int,
    control_tick_ms: int = DEFAULT_CONTROL_TICK_MS,
) -> dict[str, object]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "reward_protocol": REWARD_PROTOCOL,
        "algorithm": ALGORITHM,
        "state_encoding": STATE_ENCODING,
        "state_dimensions": STATE_DIMENSIONS,
        "action_protocol": ACTION_PROTOCOL,
        "branch_names": list(BRANCH_NAMES),
        "branch_sizes": list(BRANCH_SIZES),
        "joint_action_count": JOINT_ACTION_COUNT,
        "hidden_dimensions": list(HIDDEN_DIMENSIONS),
        "control_tick_ms": control_tick_ms,
        "replay_checkpoint_version": REPLAY_CHECKPOINT_VERSION,
        "replay_capacity": REPLAY_CAPACITY,
        "global_step": global_step,
        "episodes": episodes,
    }


def validate_checkpoint(
    checkpoint: Mapping[str, object],
    control_tick_ms: int = DEFAULT_CONTROL_TICK_MS,
) -> None:
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "reward_protocol": REWARD_PROTOCOL,
        "algorithm": ALGORITHM,
        "state_encoding": STATE_ENCODING,
        "state_dimensions": STATE_DIMENSIONS,
        "action_protocol": ACTION_PROTOCOL,
        "branch_names": list(BRANCH_NAMES),
        "branch_sizes": list(BRANCH_SIZES),
        "joint_action_count": JOINT_ACTION_COUNT,
        "control_tick_ms": control_tick_ms,
        "replay_checkpoint_version": REPLAY_CHECKPOINT_VERSION,
        "replay_capacity": REPLAY_CAPACITY,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch: {checkpoint.get(key)!r} != {value!r}")
    if "online_state_dict" not in checkpoint:
        raise ValueError("checkpoint is missing online_state_dict")


def save_checkpoint(
    path: Path,
    online: BranchingDQN,
    target: BranchingDQN,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    episodes: int,
    control_tick_ms: int = DEFAULT_CONTROL_TICK_MS,
    replay: ReplayBuffer | None = None,
) -> None:
    item = checkpoint_metadata(global_step, episodes, control_tick_ms)
    item.update(
        {
            "online_state_dict": online.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_state_dict": (
                replay if replay is not None else ReplayBuffer()
            ).state_dict(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(item, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    online: BranchingDQN,
    target: BranchingDQN,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reset: bool,
    control_tick_ms: int = DEFAULT_CONTROL_TICK_MS,
    replay: ReplayBuffer | None = None,
) -> tuple[int, int]:
    if reset:
        target.load_state_dict(online.state_dict())
        if replay is not None:
            replay.clear()
        return 0, 0
    if not path.exists():
        target.load_state_dict(online.state_dict())
        if replay is not None:
            replay.clear()
        return 0, 0
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, control_tick_ms)
    online.load_state_dict(checkpoint["online_state_dict"])
    target.load_state_dict(checkpoint.get("target_state_dict", checkpoint["online_state_dict"]))
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if replay is not None:
        replay_state = checkpoint.get("replay_state_dict")
        if not isinstance(replay_state, Mapping):
            raise ValueError("checkpoint is missing replay_state_dict")
        replay.load_state_dict(replay_state)
    return int(checkpoint.get("global_step", 0)), int(checkpoint.get("episodes", 0))


class TelemetryTail:
    """Read complete JSON objects appended after this process starts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.position = path.stat().st_size if path.exists() else 0
        self.pending = b""

    def read(self) -> Iterator[dict[str, object]]:
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        if size < self.position:
            self.position = 0
            self.pending = b""
        with self.path.open("rb") as stream:
            stream.seek(self.position)
            chunk = stream.read()
            self.position = stream.tell()
        if not chunk:
            return
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        for raw in lines:
            try:
                item = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                yield item


@dataclass
class EpisodeMetrics:
    episode: int
    steps: int = 0
    reward: float = 0.0
    gradient_updates: int = 0
    loss_total: float = 0.0
    actual_action_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    policy_q_total: float = 0.0
    policy_q_samples: int = 0
    damage_deal: int = 0
    player_hurts: int = 0
    player_damage_taken: int = 0
    dodges: int = 0
    won: bool = False
    damage_reward: float = 0.0
    dodge_reward: float = 0.0
    attack_range_reward: float = 0.0
    player_hurt_penalty: float = 0.0
    victory_reward: float = 0.0
    silk_spent: int = 0
    silk_penalty: float = 0.0
    dodge_backfill_reward: float = 0.0
    movement_dodge_reward: float = 0.0
    evade_failure_backfill_penalty: float = 0.0
    combat_hurt_penalty: float = 0.0
    failed_dodges: int = 0
    damage_credit_reward: float = 0.0
    unattributed_damage_reward: float = 0.0
    player_parries: int = 0
    parry_reward: float = 0.0
    parry_credit_reward: float = 0.0
    unattributed_parry_reward: float = 0.0
    offensive_misses: int = 0
    attack_misses: int = 0
    spell_misses: int = 0
    harpoon_damage_reward: float = 0.0
    offensive_miss_penalty: float = 0.0
    taunt_misses: int = 0
    taunt_hurts: int = 0
    taunt_penalty: float = 0.0
    dodges_by_attack: dict[str, int] = field(default_factory=dict)
    failed_dodges_by_attack: dict[str, int] = field(default_factory=dict)
    illegal_actions: int = 0
    illegal_action_penalty: float = 0.0

    def as_dict(
        self,
        epsilon: float,
        replay_size: int,
        global_step: int,
    ) -> dict[str, object]:
        item = asdict(self)
        item["mean_loss"] = (
            self.loss_total / self.gradient_updates
            if self.gradient_updates
            else None
        )
        item["mean_policy_q"] = (
            self.policy_q_total / self.policy_q_samples
            if self.policy_q_samples
            else None
        )
        item["epsilon"] = epsilon
        item["replay_size"] = replay_size
        item["global_step"] = global_step
        del item["loss_total"]
        del item["policy_q_total"]
        del item["policy_q_samples"]
        return item


@dataclass
class ArenaResetGate:
    """Ignore lingering terminal snapshots until the encounter actually exits."""

    awaiting_exit: bool = False
    encounter_id: int | None = None

    def allow_snapshot(self, is_arena: bool, encounter_id: int | None = None) -> bool:
        if not is_arena:
            self.awaiting_exit = False
            if encounter_id is not None:
                self.encounter_id = encounter_id
            return False
        if encounter_id is not None and encounter_id != self.encounter_id:
            self.encounter_id = encounter_id
            self.awaiting_exit = False
        return not self.awaiting_exit

    def mark_episode_finished(self) -> None:
        self.awaiting_exit = True

    def force_resume(self, encounter_id: int | None = None) -> None:
        if encounter_id is not None:
            self.encounter_id = encounter_id
        self.awaiting_exit = False


@dataclass
class ArenaActionWatchdog:
    timeout_seconds: float = 1.0
    last_accepted_timestamp: float | None = None

    def record(self, timestamp: float) -> None:
        self.last_accepted_timestamp = timestamp

    def reset(self) -> None:
        self.last_accepted_timestamp = None

    def stalled(self, timestamp: float) -> bool:
        if self.last_accepted_timestamp is None:
            return False
        if timestamp < self.last_accepted_timestamp:
            self.record(timestamp)
            return False
        return timestamp - self.last_accepted_timestamp >= self.timeout_seconds


class LiveTrainer:
    """Stateful bridge joining telemetry, reward, joint DQN, replay, and actions."""

    def __init__(
        self,
        online: BranchingDQN,
        target: BranchingDQN,
        optimizer: torch.optim.Optimizer,
        executor: KeyboardActionExecutor,
        device: torch.device,
        rng: random.Random,
        global_step: int = 0,
        episodes: int = 0,
        learning_enabled: bool = True,
        replay: ReplayBuffer | None = None,
    ) -> None:
        self.online = online
        self.target = target
        self.optimizer = optimizer
        self.executor = executor
        self.device = device
        self.rng = rng
        self.global_step = global_step
        self.completed_episodes = episodes
        self.learning_enabled = learning_enabled
        self.replay = replay if replay is not None else ReplayBuffer()
        self.reward_tracker = RewardTracker()
        self.previous_state: StateFrame | None = None
        self.previous_action: tuple[int, ...] | None = None
        self.previous_illegal_penalty = 0.0
        self.previous_illegal_branches: tuple[str, ...] = ()
        self.previous_taunt_started = False
        self.previous_started_branches: tuple[str, ...] = ()
        self.pending_credit_transitions: deque[PendingTransition] = deque()
        self.taunt_trials: list[TauntTrial] = []
        self.action_outcome_trials: list[ActionOutcomeTrial] = []
        self.combat_risk_trials: list[CombatRiskTrial] = []
        self.attack_windows: dict[int, BossAttackCreditWindow] = {}
        self.active_attack_id = 0
        self.synthetic_attack_id = -1
        self.attack_event_counters = {
            "started_events": 0,
            "active_events": 0,
            "finished_events": 0,
            "player_hit_events": 0,
        }
        self.attack_events_initialized = False
        self.action_exploration_state = ActionExplorationState()
        self.metrics = EpisodeMetrics(episode=episodes + 1)

    def current_epsilon(self) -> float:
        if self.global_step < PURE_EXPLORATION_STEPS:
            return 1.0
        return epsilon_for_step(self.global_step - PURE_EXPLORATION_STEPS)

    def _apply_taunt_outcomes(self, reward: RewardFrame) -> None:
        if not self.taunt_trials:
            return
        remaining: list[TauntTrial] = []
        for trial in self.taunt_trials:
            if reward.player_damage_taken > 0:
                penalty = TAUNT_HURT_PENALTY
                self.metrics.taunt_hurts += 1
            else:
                trial.remaining_steps -= 1
                if trial.remaining_steps > 0:
                    remaining.append(trial)
                    continue
                penalty = TAUNT_MISS_PENALTY
                self.metrics.taunt_misses += 1
            trial.pending.add_reward(penalty)
            self.metrics.reward += penalty
            self.metrics.taunt_penalty += penalty
        self.taunt_trials = remaining

    def _register_action_outcome(self, pending: PendingTransition) -> None:
        action_kinds = {
            "attack_x": ("attack", True),
            "skill_s": ("harpoon", False),
            "spell_shift": ("spell", True),
        }
        for event_name, (action_kind, penalize_miss) in action_kinds.items():
            if event_name in self.previous_started_branches:
                self.action_outcome_trials.append(
                    ActionOutcomeTrial(
                        pending,
                        action_kind,
                        penalize_miss,
                    )
                )

    def _penalize_offensive_miss(self, trial: ActionOutcomeTrial) -> None:
        trial.pending.add_reward(OFFENSIVE_MISS_PENALTY)
        self.metrics.reward += OFFENSIVE_MISS_PENALTY
        self.metrics.offensive_misses += 1
        self.metrics.offensive_miss_penalty += OFFENSIVE_MISS_PENALTY
        if trial.action_kind == "attack":
            self.metrics.attack_misses += 1
        elif trial.action_kind == "spell":
            self.metrics.spell_misses += 1

    def _apply_action_outcomes(
        self,
        reward: RewardFrame,
    ) -> None:
        if reward.damage_reward > 0:
            if self.action_outcome_trials:
                newest_remaining = max(
                    trial.remaining_steps for trial in self.action_outcome_trials
                )
                candidates = [
                    trial
                    for trial in self.action_outcome_trials
                    if trial.remaining_steps == newest_remaining
                ]
                share = reward.damage_reward / max(1, len(candidates))
                for trial in candidates:
                    trial.hit = True
                    trial.pending.add_reward(share)
                    if trial.action_kind == "harpoon":
                        self.metrics.harpoon_damage_reward += share
                self.metrics.damage_credit_reward += reward.damage_reward
            else:
                self.metrics.unattributed_damage_reward += reward.damage_reward

        if reward.parry_reward > 0:
            attack_trials = [
                trial
                for trial in self.action_outcome_trials
                if trial.action_kind == "attack"
            ]
            if attack_trials:
                newest_remaining = max(
                    trial.remaining_steps for trial in attack_trials
                )
                candidates = [
                    trial
                    for trial in attack_trials
                    if trial.remaining_steps == newest_remaining
                ]
                share = reward.parry_reward / max(1, len(candidates))
                for trial in candidates:
                    trial.hit = True
                    trial.pending.add_reward(share)
                self.metrics.parry_credit_reward += reward.parry_reward
            else:
                self.metrics.unattributed_parry_reward += reward.parry_reward

        remaining: list[ActionOutcomeTrial] = []
        for trial in self.action_outcome_trials:
            trial.remaining_steps -= 1
            if trial.remaining_steps > 0:
                remaining.append(trial)
                continue
            if not trial.hit and trial.penalize_miss:
                self._penalize_offensive_miss(trial)
        self.action_outcome_trials = remaining

    @staticmethod
    def _attack_credit_weight(action: tuple[int, ...]) -> float:
        jump, movement, _combat = action
        weight = 0.2
        if jump != 0:
            weight += 1.0
        if movement in (3, 4, 5, 6):
            weight += 1.0
        elif movement in (1, 2):
            weight += 0.7
        return weight

    def _window(self, attack_id: int, attack_type: str = "unknown") -> BossAttackCreditWindow:
        window = self.attack_windows.get(attack_id)
        if window is None:
            window = BossAttackCreditWindow(attack_id, attack_type)
            self.attack_windows[attack_id] = window
        elif window.attack_type == "unknown" and attack_type != "unknown":
            window.attack_type = attack_type
        return window

    def _attach_to_attack(
        self,
        pending: PendingTransition,
        attack_id: int,
        attack_type: str = "unknown",
    ) -> None:
        if attack_id == 0 or attack_id in pending.attack_ids:
            return
        window = self._window(attack_id, attack_type)
        window.transitions.append(pending)
        pending.attack_ids.add(attack_id)

    def _resolve_attack_window(self, attack_id: int) -> None:
        window = self.attack_windows.pop(attack_id, None)
        if window is None:
            return
        unique = []
        seen: set[int] = set()
        for item in window.transitions:
            marker = id(item)
            if marker not in seen:
                seen.add(marker)
                unique.append(item)
            item.attack_ids.discard(attack_id)
        if not unique or not window.active_seen:
            return
        budget = (
            EVADE_FAILURE_PENALTY if window.hurt_player else EVADE_SUCCESS_REWARD
        )
        weights = [
            self._attack_credit_weight(item.transition.action_vector)
            for item in unique
        ]
        total_weight = sum(weights)
        for item, weight in zip(unique, weights):
            item.add_reward(budget * weight / total_weight)
        if window.hurt_player:
            self.metrics.failed_dodges += 1
            self.metrics.evade_failure_backfill_penalty += budget
            self.metrics.failed_dodges_by_attack[window.attack_type] = (
                self.metrics.failed_dodges_by_attack.get(window.attack_type, 0) + 1
            )
        else:
            self.metrics.dodges += 1
            self.metrics.dodge_reward += DODGE_REWARD
            self.metrics.dodge_backfill_reward += budget
            self.metrics.movement_dodge_reward += budget
            self.metrics.dodges_by_attack[window.attack_type] = (
                self.metrics.dodges_by_attack.get(window.attack_type, 0) + 1
            )
        self.metrics.reward += budget

    @staticmethod
    def _counter(mapping: Mapping[str, object], name: str) -> int:
        value = mapping.get(name, 0)
        if isinstance(value, bool):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _apply_attack_events(
        self,
        snapshot: Mapping[str, object],
        state: StateFrame,
        reward: RewardFrame,
        pending: PendingTransition,
    ) -> None:
        raw = snapshot.get("boss_attack")
        if isinstance(raw, Mapping):
            attack_id = self._counter(raw, "id")
            attack_type = str(raw.get("type", "unknown"))
            phase = str(raw.get("phase", "idle"))
            if not self.attack_events_initialized:
                for name in self.attack_event_counters:
                    self.attack_event_counters[name] = self._counter(raw, name)
                self.attack_events_initialized = True
                self.active_attack_id = attack_id
                if attack_id:
                    window = self._window(attack_id, attack_type)
                    window.active_seen |= phase == "active"
                    window.hurt_player |= reward.player_damage_taken > 0
                    self._attach_to_attack(pending, attack_id, attack_type)
                return
            if self.active_attack_id:
                self._attach_to_attack(pending, self.active_attack_id)
            if attack_id:
                window = self._window(attack_id, attack_type)
                window.active_seen |= phase == "active"
                window.hurt_player |= reward.player_damage_taken > 0
                self._attach_to_attack(pending, attack_id, attack_type)
            hit_count = self._counter(raw, "player_hit_events")
            if hit_count > self.attack_event_counters["player_hit_events"]:
                hit_id = self._counter(raw, "last_player_hit_id")
                if hit_id:
                    self._window(hit_id).hurt_player = True
            active_count = self._counter(raw, "active_events")
            if active_count > self.attack_event_counters["active_events"]:
                if attack_id:
                    self._window(attack_id, attack_type).active_seen = True
            finished_count = self._counter(raw, "finished_events")
            if finished_count > self.attack_event_counters["finished_events"]:
                finished_id = self._counter(raw, "last_finished_id")
                if finished_id:
                    self._resolve_attack_window(finished_id)
            for name in self.attack_event_counters:
                self.attack_event_counters[name] = self._counter(raw, name)
            self.active_attack_id = attack_id
            return

        if state.attack_type != "none":
            if self.active_attack_id == 0:
                self.synthetic_attack_id -= 1
                self.active_attack_id = self.synthetic_attack_id
            window = self._window(self.active_attack_id, state.attack_type)
            window.active_seen |= state.attack_phase == "active"
            window.hurt_player |= reward.player_damage_taken > 0
            self._attach_to_attack(pending, self.active_attack_id, state.attack_type)
        elif self.active_attack_id and reward.attack_finished is not None:
            window = self._window(self.active_attack_id, reward.attack_finished)
            window.hurt_player |= reward.attack_hurt_player
            self._attach_to_attack(pending, self.active_attack_id)
            self._resolve_attack_window(self.active_attack_id)
            self.active_attack_id = 0

    def _register_combat_risk(self, pending: PendingTransition) -> None:
        if pending.transition.action_vector[BRANCH_INDEX["combat"]] != 0:
            self.combat_risk_trials.append(CombatRiskTrial(pending))

    def _apply_combat_hurt(self, player_damage_taken: int) -> None:
        if player_damage_taken <= 0 or not self.combat_risk_trials:
            return
        weights = [
            0.85 ** (COMBAT_HURT_CREDIT_WINDOW_STEPS - trial.remaining_steps)
            for trial in self.combat_risk_trials
        ]
        penalty = COMBAT_ACTION_HURT_PENALTY_PER_HP * player_damage_taken
        total_weight = sum(weights)
        for trial, weight in zip(self.combat_risk_trials, weights):
            trial.pending.add_reward(penalty * weight / total_weight)
        self.metrics.combat_hurt_penalty += penalty
        self.metrics.reward += penalty

    def _age_combat_risks(self) -> None:
        remaining = []
        for trial in self.combat_risk_trials:
            trial.remaining_steps -= 1
            if trial.remaining_steps > 0:
                remaining.append(trial)
        self.combat_risk_trials = remaining

    def _finalize_pending(self, force: bool = False) -> None:
        retained: deque[PendingTransition] = deque()
        for pending in self.pending_credit_transitions:
            age = self.global_step - pending.created_step
            if force or (age >= CREDIT_FINALIZATION_STEPS and not pending.attack_ids):
                self.replay.append(pending.finalize())
            else:
                retained.append(pending)
        self.pending_credit_transitions = retained

    def _expire_taunt_trials(self) -> None:
        for trial in self.taunt_trials:
            trial.pending.add_reward(TAUNT_MISS_PENALTY)
            self.metrics.reward += TAUNT_MISS_PENALTY
            self.metrics.taunt_penalty += TAUNT_MISS_PENALTY
            self.metrics.taunt_misses += 1
        self.taunt_trials.clear()

    def _expire_action_outcomes(self) -> None:
        for trial in self.action_outcome_trials:
            if trial.hit or not trial.penalize_miss:
                continue
            self._penalize_offensive_miss(trial)
        self.action_outcome_trials.clear()

    def observe(
        self,
        snapshot: Mapping[str, object],
        force_terminal: bool = False,
    ) -> RewardFrame:
        reward = self.reward_tracker.step(snapshot)
        if force_terminal and not reward.terminated:
            reward = replace(reward, terminated=True)
        if reward.player_hurt < 0:
            self.executor.release_all()
            self.action_exploration_state.clear()
        masks, mask_reasons = branch_availability(
            snapshot,
            self.executor.continuing_action,
            harpoon_locked=self.executor.harpoon_locked,
            charge_protected=self.executor.charge_protected,
            charge_must_hold=self.executor.charge_must_hold,
        )
        state = encode_snapshot(snapshot, self.executor.control_state(snapshot))
        if (
            self.learning_enabled
            and self.previous_state is not None
            and self.previous_action is not None
        ):
            taunt_step_penalty = (
                TAUNT_STEP_PENALTY if self.previous_action[2] == 4 else 0.0
            )
            transition_reward = (
                reward.total
                - reward.damage_reward
                - reward.dodge
                - reward.parry_reward
                + self.previous_illegal_penalty
                + taunt_step_penalty
            )
            transition = Transition(
                state=self.previous_state.observation,
                action=joint_action_id(self.previous_action),
                reward=transition_reward,
                next_state=state.observation,
                done=reward.terminated,
                next_action_mask=joint_action_mask(masks),
            )
            pending = PendingTransition(transition, self.global_step)
            self.pending_credit_transitions.append(pending)
            if self.previous_taunt_started:
                self.taunt_trials.append(TauntTrial(pending))
            self._register_action_outcome(pending)
            self._register_combat_risk(pending)
            self.metrics.steps += 1
            self.metrics.reward += transition_reward
            self.metrics.taunt_penalty += taunt_step_penalty
            self.metrics.damage_deal += reward.damage_deal
            self.metrics.player_hurts += int(reward.player_hurt < 0)
            self.metrics.player_damage_taken += reward.player_damage_taken
            self.metrics.damage_reward += reward.damage_reward
            self.metrics.player_parries += reward.player_parries
            self.metrics.parry_reward += reward.parry_reward
            self.metrics.attack_range_reward += reward.entered_attack_range
            self.metrics.player_hurt_penalty += reward.player_hurt
            self.metrics.victory_reward += reward.victory
            self.metrics.silk_spent += reward.silk_spent
            self.metrics.silk_penalty += reward.silk_penalty
            if self.previous_illegal_penalty < 0:
                self.metrics.illegal_actions += 1
                self.metrics.illegal_action_penalty += self.previous_illegal_penalty
            self._apply_taunt_outcomes(reward)
            self._apply_action_outcomes(reward)
            self._apply_combat_hurt(reward.player_damage_taken)
            self._apply_attack_events(snapshot, state, reward, pending)
            self._age_combat_risks()
            self.global_step += 1
            self._finalize_pending()
            if len(self.replay) >= REPLAY_WARMUP:
                loss = optimize_model(
                    self.online,
                    self.target,
                    self.optimizer,
                    self.replay.sample(BATCH_SIZE, self.rng),
                    self.device,
                )
                self.metrics.gradient_updates += 1
                self.metrics.loss_total += loss
            if self.global_step % TARGET_UPDATE_INTERVAL == 0:
                self.target.load_state_dict(self.online.state_dict())

        if reward.terminated:
            self.metrics.won = reward.boss_dead
            self.executor.release_all()
            return reward

        selected_q_values: list[float] = []
        action = select_action(
            self.online,
            state.observation,
            self.current_epsilon(),
            self.rng,
            self.device,
            masks,
            self.action_exploration_state,
            selected_q_values,
        )
        action_result = self.executor.apply(
            action,
            branch_masks=masks,
            masked_reasons=mask_reasons,
            player_resources=state.resources,
        )
        self.previous_state = state
        executed_action = action_result.get("action_vector", action)
        self.previous_action = validate_action(executed_action)
        for branch_index, (branch_name, action_value) in enumerate(
            zip(BRANCH_NAMES, self.previous_action)
        ):
            branch_counts = self.metrics.actual_action_counts.setdefault(
                branch_name, {}
            )
            action_name = ACTION_LABELS[branch_index][action_value]
            branch_counts[action_name] = branch_counts.get(action_name, 0) + 1
        if len(selected_q_values) == 1:
            self.metrics.policy_q_total += selected_q_values[0]
            self.metrics.policy_q_samples += 1
        self.action_exploration_state.reconcile(self.previous_action[1])
        newly_pressed = action_result.get("newly_pressed_keys", ())
        self.previous_taunt_started = "V" in newly_pressed
        raw_started = action_result.get("started_branches", ())
        self.previous_started_branches = tuple(str(value) for value in raw_started)
        raw_illegal_branches = action_result.get("illegal_branches", ())
        self.previous_illegal_branches = tuple(
            str(value) for value in raw_illegal_branches
        )
        self.previous_illegal_penalty = (
            ILLEGAL_ACTION_PENALTY if self.previous_illegal_branches else 0.0
        )
        return reward

    def finish_episode(self) -> dict[str, object]:
        self._expire_taunt_trials()
        self._expire_action_outcomes()
        for attack_id in list(self.attack_windows):
            self._resolve_attack_window(attack_id)
        self._finalize_pending(force=True)
        result = self.metrics.as_dict(
            self.current_epsilon(),
            len(self.replay),
            self.global_step,
        )
        self.completed_episodes += 1
        self.executor.release_all()
        self.reward_tracker.reset()
        self.previous_state = None
        self.previous_action = None
        self.previous_illegal_penalty = 0.0
        self.previous_illegal_branches = ()
        self.previous_taunt_started = False
        self.previous_started_branches = ()
        self.action_exploration_state.clear()
        self.combat_risk_trials.clear()
        self.attack_windows.clear()
        self.active_attack_id = 0
        self.metrics = EpisodeMetrics(episode=self.completed_episodes + 1)
        return result

    def reset_interrupted_episode(self) -> None:
        """Keep completed replay entries but break any cross-process transition."""

        for attack_id in list(self.attack_windows):
            self._resolve_attack_window(attack_id)
        self._finalize_pending(force=True)
        self.taunt_trials.clear()
        self.action_outcome_trials.clear()
        self.combat_risk_trials.clear()
        self.attack_windows.clear()
        self.active_attack_id = 0
        self.executor.release_all()
        self.reward_tracker.reset()
        self.previous_state = None
        self.previous_action = None
        self.previous_illegal_penalty = 0.0
        self.previous_illegal_branches = ()
        self.previous_taunt_started = False
        self.previous_started_branches = ()
        self.action_exploration_state.clear()
        self.metrics = EpisodeMetrics(episode=self.completed_episodes + 1)


def append_metric(path: Path, metric: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(dict(metric), separators=(",", ":")) + "\n")


def _encounter_id(snapshot: Mapping[str, object]) -> int | None:
    value = snapshot.get("encounter_id")
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snapshot_timestamp(snapshot: Mapping[str, object]) -> float:
    try:
        return float(snapshot.get("timestamp", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _player_alive(snapshot: Mapping[str, object]) -> bool:
    health = snapshot.get("player_health")
    if not isinstance(health, Mapping):
        return False
    value = health.get("health")
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def train_live(args: argparse.Namespace) -> None:
    if args.launch and not args.game_exe.exists():
        raise FileNotFoundError(args.game_exe)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    online = BranchingDQN().to(device)
    target = BranchingDQN().to(device)
    optimizer = torch.optim.AdamW(online.parameters(), lr=LEARNING_RATE)
    checkpoint_exists = args.checkpoint.exists()
    if args.reset:
        print(f"explicit reset requested; starting new training: {args.checkpoint}", flush=True)
    elif checkpoint_exists:
        print(f"resuming checkpoint without reset: {args.checkpoint}", flush=True)
    else:
        print(f"checkpoint not found; starting new training: {args.checkpoint}", flush=True)
    replay = ReplayBuffer()
    global_step, episodes = load_checkpoint(
        args.checkpoint,
        online,
        target,
        optimizer,
        device,
        args.reset,
        args.tick_ms,
        replay,
    )
    if replay:
        print(f"restored replay transitions: {len(replay)}", flush=True)
    online.train()
    target.eval()
    process: subprocess.Popen[bytes] | None = None
    relaunch_times: deque[float] = deque()
    if args.launch:
        process = subprocess.Popen([str(args.game_exe)], cwd=str(args.game_exe.parent))
        print(f"started Silksong pid={process.pid}; waiting for game window", flush=True)
    recorder = ActionRecorder(args.action_log)
    try:
        executor = KeyboardActionExecutor(
            recorder,
            tick_ms=args.tick_ms,
            send_input=args.execute_actions,
        )
    except Exception:
        if process is not None and process.poll() is None and not args.keep_game:
            process.terminate()
        raise
    trainer = LiveTrainer(
        online,
        target,
        optimizer,
        executor,
        device,
        rng,
        global_step,
        episodes,
        learning_enabled=args.execute_actions,
        replay=replay,
    )
    tail = TelemetryTail(args.telemetry)
    in_arena = False
    reset_gate = ArenaResetGate()
    action_watchdog = ArenaActionWatchdog()
    try:
        if not args.execute_actions:
            print(
                "dry-run: actions are logged; keyboard input and learning are disabled",
                flush=True,
            )
        while trainer.completed_episodes < args.episodes:
            had_data = False
            for snapshot in tail.read():
                had_data = True
                is_arena = (
                    snapshot.get("type") == "snapshot"
                    and snapshot.get("scene") == ARENA_SCENE
                    and bool(snapshot.get("encounter_active"))
                    and snapshot.get("player") is not None
                    and snapshot.get("boss") is not None
                )
                encounter_id = _encounter_id(snapshot)
                snapshot_timestamp = _snapshot_timestamp(snapshot)
                if not is_arena:
                    if in_arena and trainer.previous_state is not None:
                        trainer.observe(snapshot, force_terminal=True)
                        metric = trainer.finish_episode()
                        append_metric(args.metrics, metric)
                        save_checkpoint(
                            args.checkpoint,
                            online,
                            target,
                            optimizer,
                            trainer.global_step,
                            trainer.completed_episodes,
                            args.tick_ms,
                            trainer.replay,
                        )
                        print(json.dumps(metric), flush=True)
                    in_arena = False
                    reset_gate.allow_snapshot(False, encounter_id)
                    action_watchdog.reset()
                    continue
                if not reset_gate.allow_snapshot(True, encounter_id):
                    if not (
                        _player_alive(snapshot)
                        and action_watchdog.stalled(snapshot_timestamp)
                    ):
                        continue
                    print(
                        "action watchdog recovered a stalled arena gate: "
                        f"encounter_id={encounter_id}",
                        flush=True,
                    )
                    executor.release_all()
                    reset_gate.force_resume(encounter_id)
                in_arena = True
                action_watchdog.record(snapshot_timestamp)
                reward = trainer.observe(snapshot)
                if reward.terminated or trainer.metrics.steps >= args.max_episode_steps:
                    metric = trainer.finish_episode()
                    append_metric(args.metrics, metric)
                    save_checkpoint(
                        args.checkpoint,
                        online,
                        target,
                        optimizer,
                        trainer.global_step,
                        trainer.completed_episodes,
                        args.tick_ms,
                        trainer.replay,
                    )
                    print(json.dumps(metric), flush=True)
                    in_arena = False
                    reset_gate.mark_episode_finished()
            if process is not None and process.poll() is not None:
                return_code = process.returncode
                if return_code == 0:
                    try:
                        find_game_window()
                    except RuntimeError:
                        now = time.monotonic()
                        while (
                            relaunch_times
                            and now - relaunch_times[0]
                            > GAME_RELAUNCH_WINDOW_SECONDS
                        ):
                            relaunch_times.popleft()
                        if len(relaunch_times) >= MAX_GAME_RELAUNCHES_PER_WINDOW:
                            raise RuntimeError(
                                "Silksong repeatedly closed normally; refusing "
                                "to relaunch more than "
                                f"{MAX_GAME_RELAUNCHES_PER_WINDOW} times in "
                                f"{GAME_RELAUNCH_WINDOW_SECONDS:.0f} seconds"
                            ) from None
                        relaunch_times.append(now)
                        if trainer.previous_state is not None:
                            print(
                                "discarding the interrupted episode boundary "
                                "before relaunch; completed replay is retained",
                                flush=True,
                            )
                            trainer.reset_interrupted_episode()
                        save_checkpoint(
                            args.checkpoint,
                            online,
                            target,
                            optimizer,
                            trainer.global_step,
                            trainer.completed_episodes,
                            args.tick_ms,
                            trainer.replay,
                        )
                        print(
                            "Silksong closed normally before training finished; "
                            "relaunching",
                            flush=True,
                        )
                        time.sleep(GAME_RELAUNCH_DELAY_SECONDS)
                        process = subprocess.Popen(
                            [str(args.game_exe)], cwd=str(args.game_exe.parent)
                        )
                        print(
                            f"restarted Silksong pid={process.pid}; waiting for "
                            "game window",
                            flush=True,
                        )
                        in_arena = False
                        reset_gate = ArenaResetGate()
                        action_watchdog.reset()
                        continue
                    print(
                        "the launched process exited with code 0, but the "
                        "Silksong window is still running; continuing training",
                        flush=True,
                    )
                    process = None
                else:
                    raise RuntimeError(f"Silksong exited with code {return_code}")
            if not had_data:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("training interrupted; saving checkpoint", flush=True)
    finally:
        executor.close()
        save_checkpoint(
            args.checkpoint,
            online,
            target,
            optimizer,
            trainer.global_step,
            trainer.completed_episodes,
            args.tick_ms,
            trainer.replay,
        )
        if process is not None and process.poll() is None and not args.keep_game:
            process.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a live joint-action Double DQN agent")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--action-log", type=Path, default=DEFAULT_ACTION_LOG)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--game-exe", type=Path, default=DEFAULT_GAME_EXE)
    parser.add_argument("--tick-ms", type=int, default=DEFAULT_CONTROL_TICK_MS)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="explicitly discard checkpoint state; never enabled implicitly",
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--execute-actions", action="store_true")
    parser.add_argument("--keep-game", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    train_live(args)


if __name__ == "__main__":
    main()
