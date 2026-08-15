"""Live Branching Double DQN for the Silksong telemetry/action pipeline."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
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
from .real_reward import ILLEGAL_ACTION_PENALTY, RewardFrame, RewardTracker
from .real_state import STATE_DIMENSIONS, StateFrame, encode_snapshot


STATE_ENCODING = "real-telemetry-state-v11-semantic-24"
ALGORITHM = "branching-dueling-double-dqn"
CHECKPOINT_VERSION = 20
REPLAY_CHECKPOINT_VERSION = 1
REWARD_PROTOCOL = "three-head-harpoon-movement-v14"
HIDDEN_DIMENSIONS = (96, 96)
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
REPLAY_WARMUP = 1_000
TARGET_UPDATE_INTERVAL = 500
EPSILON_START = 0.60
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 15_000
EXPLORATION_ACTIVATION_RATES = (0.45, 0.85, 0.30)
MOVEMENT_EXPLORATION_WEIGHTS = (0.0, 32.0, 32.0, 12.0, 8.0, 8.0, 8.0)
DODGE_BACKFILL_DISCOUNT = 0.9
EVADE_FAILURE_BACKFILL_PENALTY = -0.5
TAUNT_OUTCOME_WINDOW_STEPS = 6
TAUNT_STEP_PENALTY = -0.02
TAUNT_MISS_PENALTY = 0.0
TAUNT_HURT_PENALTY = -1.0
DAMAGE_CREDIT_WINDOW_STEPS = 20
OFFENSIVE_MISS_PENALTY = -0.5
GRADIENT_CLIP_NORM = 10.0
BRANCH_INDEX = {name: index for index, name in enumerate(BRANCH_NAMES)}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "real_dqn.pt"
DEFAULT_METRICS = PROJECT_ROOT / "runs" / "real_dqn.jsonl"
DEFAULT_ACTION_LOG = PROJECT_ROOT / "runs" / "real_dqn_actions.jsonl"
DEFAULT_CONTROL_TICK_MS = 100
DEFAULT_GAME_EXE = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\Hollow Knight Silksong\Hollow Knight Silksong.exe"
)
DEFAULT_TELEMETRY = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight Silksong"
    r"\BepInEx\plugins\hollow-knight-rl-KarmelitaPractice\telemetry.jsonl"
)
ARENA_SCENE = "Memory_Ant_Queen"


class BranchingDQN(nn.Module):
    """Dueling shared encoder with one discrete Q head per key branch."""

    def __init__(
        self,
        state_dimensions: int = STATE_DIMENSIONS,
        branch_sizes: Sequence[int] = BRANCH_SIZES,
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
        self.advantages = nn.ModuleList(
            nn.Linear(feature_size, int(size)) for size in branch_sizes
        )
        self.branch_sizes = tuple(int(size) for size in branch_sizes)

    def forward(self, states: Tensor) -> tuple[Tensor, ...]:
        features = self.shared(states)
        value = self.value(features)
        outputs = []
        for head in self.advantages:
            advantage = head(features)
            outputs.append(value + advantage - advantage.mean(dim=-1, keepdim=True))
        return tuple(outputs)


@dataclass
class Transition:
    state: tuple[float, ...]
    action: tuple[int, ...]
    reward: float
    next_state: tuple[float, ...]
    done: bool
    next_action_masks: BranchMasks
    branch_rewards: list[float] | None = None

    def __post_init__(self) -> None:
        if self.branch_rewards is None:
            self.branch_rewards = [float(self.reward)] * len(BRANCH_SIZES)
        elif len(self.branch_rewards) != len(BRANCH_SIZES):
            raise ValueError("invalid branch reward dimensions")

    def add_branch_reward(self, branch_index: int, value: float) -> None:
        assert self.branch_rewards is not None
        self.branch_rewards[branch_index] += value


@dataclass
class TauntTrial:
    transition: Transition
    remaining_steps: int = TAUNT_OUTCOME_WINDOW_STEPS


@dataclass
class ActionOutcomeTrial:
    transition: Transition
    action_kind: str
    branch_index: int
    penalize_miss: bool = True
    remaining_steps: int = DAMAGE_CREDIT_WINDOW_STEPS
    hit: bool = False


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
        validate_action(transition.action)
        validate_masks(transition.next_action_masks)
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
            ).reshape(-1, len(BRANCH_SIZES)),
            "rewards": torch.tensor(
                [item.reward for item in items], dtype=torch.float32
            ),
            "next_states": torch.tensor(
                [item.next_state for item in items], dtype=torch.float32
            ).reshape(-1, STATE_DIMENSIONS),
            "dones": torch.tensor([item.done for item in items], dtype=torch.bool),
            "next_action_masks": tuple(
                torch.tensor(
                    [item.next_action_masks[index] for item in items],
                    dtype=torch.bool,
                ).reshape(-1, size)
                for index, size in enumerate(BRANCH_SIZES)
            ),
            "branch_rewards": torch.tensor(
                [item.branch_rewards for item in items], dtype=torch.float32
            ).reshape(-1, len(BRANCH_SIZES)),
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
            "branch_rewards",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"checkpoint replay fields are missing: {missing}")
        states = torch.as_tensor(data.get("states")).cpu()
        actions = torch.as_tensor(data.get("actions")).cpu()
        rewards = torch.as_tensor(data.get("rewards")).cpu()
        next_states = torch.as_tensor(data.get("next_states")).cpu()
        dones = torch.as_tensor(data.get("dones")).cpu()
        branch_rewards = torch.as_tensor(data.get("branch_rewards")).cpu()
        raw_masks = data.get("next_action_masks")
        if not isinstance(raw_masks, (tuple, list)):
            raise ValueError("checkpoint replay masks are missing")
        masks = tuple(torch.as_tensor(mask).cpu() for mask in raw_masks)
        size = int(states.shape[0]) if states.ndim == 2 else -1
        expected_shapes = (
            states.shape == (size, STATE_DIMENSIONS),
            actions.shape == (size, len(BRANCH_SIZES)),
            rewards.shape == (size,),
            next_states.shape == (size, STATE_DIMENSIONS),
            dones.shape == (size,),
            branch_rewards.shape == (size, len(BRANCH_SIZES)),
            len(masks) == len(BRANCH_SIZES),
            all(
                mask.shape == (size, branch_size)
                for mask, branch_size in zip(masks, BRANCH_SIZES)
            ),
        )
        if not all(expected_shapes):
            raise ValueError("checkpoint replay tensor dimensions are invalid")
        if size > int(self._items.maxlen or 0):
            raise ValueError("checkpoint replay exceeds configured capacity")
        if not all(
            torch.isfinite(values).all().item()
            for values in (states, rewards, next_states, branch_rewards)
        ):
            raise ValueError("checkpoint replay contains non-finite values")
        self.clear()
        for index in range(size):
            self.append(
                Transition(
                    state=tuple(float(value) for value in states[index].tolist()),
                    action=tuple(int(value) for value in actions[index].tolist()),
                    reward=float(rewards[index].item()),
                    next_state=tuple(
                        float(value) for value in next_states[index].tolist()
                    ),
                    done=bool(dones[index].item()),
                    next_action_masks=tuple(
                        tuple(bool(value) for value in mask[index].tolist())
                        for mask in masks
                    ),
                    branch_rewards=[
                        float(value) for value in branch_rewards[index].tolist()
                    ],
                )
            )


def epsilon_for_step(step: int) -> float:
    fraction = min(1.0, max(0, step) / EPSILON_DECAY_STEPS)
    if fraction >= 1.0:
        return EPSILON_END
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def select_action(
    network: BranchingDQN,
    observation: Sequence[float],
    epsilon: float,
    rng: random.Random,
    device: torch.device,
    branch_masks: BranchMasks | None = None,
    exploration_state: ActionExplorationState | None = None,
) -> tuple[int, ...]:
    if len(observation) != STATE_DIMENSIONS:
        raise ValueError(f"expected {STATE_DIMENSIONS} state values")
    with torch.no_grad():
        values = network(
            torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        )
    action = []
    masks = (
        validate_masks(branch_masks)
        if branch_masks is not None
        else tuple(
            tuple(True for _ in range(size)) for size in BRANCH_SIZES
        )
    )
    explore = rng.random() < epsilon
    for branch_index, (branch_values, size, mask) in enumerate(
        zip(values, BRANCH_SIZES, masks)
    ):
        available = [index for index, allowed in enumerate(mask) if allowed]
        if not available:
            raise ValueError("every action branch must have an available value")
        sticky_movement = (
            exploration_state.consume(mask)
            if branch_index == BRANCH_INDEX["movement"] and exploration_state is not None
            else None
        )
        if sticky_movement is not None:
            action.append(sticky_movement)
        elif explore:
            non_neutral = [index for index in available if index != 0]
            activation_rate = EXPLORATION_ACTIVATION_RATES[branch_index]
            if non_neutral and rng.random() < activation_rate:
                if branch_index == BRANCH_INDEX["movement"]:
                    total_weight = sum(
                        MOVEMENT_EXPLORATION_WEIGHTS[index] for index in non_neutral
                    )
                    threshold = rng.random() * total_weight
                    selected = non_neutral[-1]
                    for candidate in non_neutral:
                        threshold -= MOVEMENT_EXPLORATION_WEIGHTS[candidate]
                        if threshold < 0:
                            selected = candidate
                            break
                    action.append(selected)
                    if exploration_state is not None and selected in (1, 2):
                        exploration_state.start(selected, rng.choice([1, 2]))
                else:
                    action.append(rng.choice(non_neutral))
            else:
                action.append(0 if 0 in available else rng.choice(available))
        else:
            scores = branch_values.squeeze(0).clone()
            scores[torch.tensor([not allowed for allowed in mask], device=device)] = -torch.inf
            action.append(int(scores.argmax().item()))
    return tuple(action)


def optimize_model(
    online: BranchingDQN,
    target: BranchingDQN,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
) -> float:
    if not transitions:
        raise ValueError("transitions must not be empty")
    states = torch.tensor([item.state for item in transitions], dtype=torch.float32, device=device)
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long, device=device)
    branch_rewards = torch.tensor(
        [item.branch_rewards for item in transitions], dtype=torch.float32, device=device
    )
    next_states = torch.tensor(
        [item.next_state for item in transitions], dtype=torch.float32, device=device
    )
    dones = torch.tensor([item.done for item in transitions], dtype=torch.bool, device=device)

    online_values = online(states)
    selected = torch.stack(
        [
            branch.gather(1, actions[:, index : index + 1]).squeeze(1)
            for index, branch in enumerate(online_values)
        ],
        dim=1,
    )

    with torch.no_grad():
        online_next = online(next_states)
        next_masks = [
            torch.tensor(
                [item.next_action_masks[index] for item in transitions],
                dtype=torch.bool,
                device=device,
            )
            for index in range(len(BRANCH_SIZES))
        ]
        next_actions = [
            branch.masked_fill(~mask, -torch.inf).argmax(dim=1, keepdim=True)
            for branch, mask in zip(online_next, next_masks)
        ]
        target_next = target(next_states)
        next_values = torch.stack(
            [
                branch.gather(1, action).squeeze(1)
                for branch, action in zip(target_next, next_actions)
            ],
            dim=1,
        ).mean(dim=1)
        immediate = branch_rewards
        expected = immediate + GAMMA * next_values.unsqueeze(1) * (~dones).unsqueeze(1)

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
    losses: int = 0
    loss_total: float = 0.0
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
    evade_failure_backfill_penalty: float = 0.0
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

    def as_dict(self, epsilon: float) -> dict[str, object]:
        item = asdict(self)
        item["mean_loss"] = self.loss_total / self.losses if self.losses else None
        item["epsilon"] = epsilon
        del item["loss_total"]
        del item["losses"]
        return item


@dataclass
class ArenaResetGate:
    """Ignore lingering terminal snapshots until the encounter actually exits."""

    awaiting_exit: bool = False

    def allow_snapshot(self, is_arena: bool) -> bool:
        if not is_arena:
            self.awaiting_exit = False
            return False
        return not self.awaiting_exit

    def mark_episode_finished(self) -> None:
        self.awaiting_exit = True


class LiveTrainer:
    """Stateful bridge joining telemetry, reward, BDQ, replay, and actions."""

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
        self.pending_attack_transitions: list[Transition] = []
        self.taunt_trials: list[TauntTrial] = []
        self.action_outcome_trials: list[ActionOutcomeTrial] = []
        self.action_exploration_state = ActionExplorationState()
        self.metrics = EpisodeMetrics(episode=episodes + 1)

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
            trial.transition.reward += penalty
            trial.transition.add_branch_reward(BRANCH_INDEX["combat"], penalty)
            self.metrics.reward += penalty
            self.metrics.taunt_penalty += penalty
        self.taunt_trials = remaining

    def _register_action_outcome(self, transition: Transition) -> None:
        action_kinds = {
            "attack_x": ("attack", BRANCH_INDEX["combat"], True),
            "skill_s": ("harpoon", BRANCH_INDEX["movement"], False),
            "spell_shift": ("spell", BRANCH_INDEX["combat"], True),
        }
        for event_name, (action_kind, branch_index, penalize_miss) in action_kinds.items():
            if event_name in self.previous_started_branches:
                self.action_outcome_trials.append(
                    ActionOutcomeTrial(
                        transition,
                        action_kind,
                        branch_index,
                        penalize_miss,
                    )
                )

    def _penalize_offensive_miss(self, trial: ActionOutcomeTrial) -> None:
        trial.transition.reward += OFFENSIVE_MISS_PENALTY
        trial.transition.add_branch_reward(
            trial.branch_index, OFFENSIVE_MISS_PENALTY
        )
        self.metrics.reward += OFFENSIVE_MISS_PENALTY
        self.metrics.offensive_misses += 1
        self.metrics.offensive_miss_penalty += OFFENSIVE_MISS_PENALTY
        if trial.action_kind == "attack":
            self.metrics.attack_misses += 1
        elif trial.action_kind == "spell":
            self.metrics.spell_misses += 1

    def _apply_action_outcomes(self, transition: Transition, reward: RewardFrame) -> None:
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
                    trial.transition.add_branch_reward(trial.branch_index, share)
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
                    trial.transition.add_branch_reward(BRANCH_INDEX["combat"], share)
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

    def _store_transition(
        self,
        transition: Transition,
        reward: RewardFrame,
        state: StateFrame,
    ) -> None:
        attack_related = (
            bool(self.pending_attack_transitions)
            or self.previous_state is not None
            and self.previous_state.attack_type != "none"
            or state.attack_type != "none"
            or reward.attack_finished is not None
        )
        if not attack_related:
            self.replay.append(transition)
            return
        self.pending_attack_transitions.append(transition)
        if reward.attack_finished is None and not reward.terminated:
            return
        if reward.dodge > 0:
            for distance, item in enumerate(
                reversed(self.pending_attack_transitions)
            ):
                bonus = reward.dodge * (DODGE_BACKFILL_DISCOUNT ** distance)
                for branch_index in (
                    BRANCH_INDEX["jump_z"],
                    BRANCH_INDEX["movement"],
                ):
                    item.add_branch_reward(branch_index, bonus)
                if distance > 0:
                    item.reward += bonus
                    self.metrics.reward += bonus
                    self.metrics.dodge_backfill_reward += bonus
        elif reward.attack_hurt_player:
            self.metrics.failed_dodges += 1
            if reward.attack_finished is not None:
                self.metrics.failed_dodges_by_attack[reward.attack_finished] = (
                    self.metrics.failed_dodges_by_attack.get(
                        reward.attack_finished, 0
                    )
                    + 1
                )
            for distance, item in enumerate(
                reversed(self.pending_attack_transitions)
            ):
                penalty = EVADE_FAILURE_BACKFILL_PENALTY * (
                    DODGE_BACKFILL_DISCOUNT ** distance
                )
                for branch_index in (
                    BRANCH_INDEX["jump_z"],
                    BRANCH_INDEX["movement"],
                ):
                    item.add_branch_reward(branch_index, penalty)
                item.reward += penalty
                self.metrics.reward += penalty
                self.metrics.evade_failure_backfill_penalty += penalty
        for item in self.pending_attack_transitions:
            self.replay.append(item)
        self.pending_attack_transitions.clear()

    def _flush_pending_transitions(self) -> None:
        for item in self.pending_attack_transitions:
            self.replay.append(item)
        self.pending_attack_transitions.clear()

    def _expire_taunt_trials(self) -> None:
        for trial in self.taunt_trials:
            trial.transition.reward += TAUNT_MISS_PENALTY
            trial.transition.add_branch_reward(
                BRANCH_INDEX["combat"], TAUNT_MISS_PENALTY
            )
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

    def observe(self, snapshot: Mapping[str, object]) -> RewardFrame:
        reward = self.reward_tracker.step(snapshot)
        if reward.player_hurt < 0:
            self.executor.release_all()
            self.action_exploration_state.clear()
        masks, mask_reasons = branch_availability(
            snapshot,
            self.executor.continuing_action,
            harpoon_locked=self.executor.harpoon_locked,
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
                reward.total + self.previous_illegal_penalty + taunt_step_penalty
            )
            global_reward = (
                reward.total
                - reward.damage_reward
                - reward.dodge
                - reward.parry_reward
                - reward.silk_penalty
            )
            branch_rewards = [global_reward] * len(BRANCH_SIZES)
            branch_rewards[BRANCH_INDEX["combat"]] += (
                reward.silk_penalty + taunt_step_penalty
            )
            if self.previous_illegal_penalty < 0:
                for name in self.previous_illegal_branches:
                    branch_index = BRANCH_INDEX.get(name)
                    if branch_index is not None:
                        branch_rewards[branch_index] += self.previous_illegal_penalty
            transition = Transition(
                state=self.previous_state.observation,
                action=self.previous_action,
                reward=transition_reward,
                next_state=state.observation,
                done=reward.terminated,
                next_action_masks=masks,
                branch_rewards=branch_rewards,
            )
            if self.previous_taunt_started and not self.taunt_trials:
                self.taunt_trials.append(TauntTrial(transition))
            self._register_action_outcome(transition)
            self.metrics.steps += 1
            self.metrics.reward += transition_reward
            self.metrics.taunt_penalty += taunt_step_penalty
            self.metrics.damage_deal += reward.damage_deal
            self.metrics.player_hurts += int(reward.player_hurt < 0)
            self.metrics.player_damage_taken += reward.player_damage_taken
            self.metrics.dodges += int(reward.dodge > 0)
            self.metrics.damage_reward += reward.damage_reward
            self.metrics.dodge_reward += reward.dodge
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
            if reward.dodge > 0 and reward.attack_finished is not None:
                self.metrics.dodges_by_attack[reward.attack_finished] = (
                    self.metrics.dodges_by_attack.get(reward.attack_finished, 0) + 1
                )
            self._apply_taunt_outcomes(reward)
            self._apply_action_outcomes(transition, reward)
            self._store_transition(transition, reward, state)
            self.global_step += 1
            if len(self.replay) >= REPLAY_WARMUP:
                loss = optimize_model(
                    self.online,
                    self.target,
                    self.optimizer,
                    self.replay.sample(BATCH_SIZE, self.rng),
                    self.device,
                )
                self.metrics.losses += 1
                self.metrics.loss_total += loss
            if self.global_step % TARGET_UPDATE_INTERVAL == 0:
                self.target.load_state_dict(self.online.state_dict())

        if reward.terminated:
            self.metrics.won = reward.boss_dead
            self.executor.release_all()
            return reward

        action = select_action(
            self.online,
            state.observation,
            epsilon_for_step(self.global_step),
            self.rng,
            self.device,
            masks,
            self.action_exploration_state,
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
        self._flush_pending_transitions()
        result = self.metrics.as_dict(epsilon_for_step(self.global_step))
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
        self.metrics = EpisodeMetrics(episode=self.completed_episodes + 1)
        return result


def append_metric(path: Path, metric: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(dict(metric), separators=(",", ":")) + "\n")


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
                if not is_arena:
                    if in_arena and trainer.previous_state is not None:
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
                    reset_gate.allow_snapshot(False)
                    continue
                if not reset_gate.allow_snapshot(True):
                    continue
                in_arena = True
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
                        raise RuntimeError(
                            "Silksong closed normally before the requested "
                            f"{args.episodes} episodes completed"
                        ) from None
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
    parser = argparse.ArgumentParser(description="Train a live Branching Double DQN agent")
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
