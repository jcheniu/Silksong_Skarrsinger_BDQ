"""Live Branching Double DQN for the Silksong telemetry/action pipeline."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
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
    validate_action,
    validate_masks,
)
from .final_project.action_recorder import ActionRecorder
from .real_reward import RewardFrame, RewardTracker
from .real_state import STATE_DIMENSIONS, StateFrame, encode_snapshot


STATE_ENCODING = "real-telemetry-state-v2-silk"
ALGORITHM = "branching-dueling-double-dqn"
CHECKPOINT_VERSION = 2
HIDDEN_DIMENSIONS = (128, 128)
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
REPLAY_WARMUP = 1_000
TARGET_UPDATE_INTERVAL = 500
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 40_000
GRADIENT_CLIP_NORM = 10.0
DEFAULT_CHECKPOINT = Path("checkpoints/real_dqn.pt")
DEFAULT_METRICS = Path("runs/real_dqn.jsonl")
DEFAULT_ACTION_LOG = Path("runs/real_dqn_actions.jsonl")
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


@dataclass(frozen=True)
class Transition:
    state: tuple[float, ...]
    action: tuple[int, ...]
    reward: float
    next_state: tuple[float, ...]
    done: bool
    next_action_masks: BranchMasks


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
        else tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
    )
    for branch_values, size, mask in zip(values, BRANCH_SIZES, masks):
        available = [index for index, allowed in enumerate(mask) if allowed]
        if not available:
            raise ValueError("every action branch must have an available value")
        if rng.random() < epsilon:
            action.append(rng.choice(available))
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
    rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32, device=device)
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
    ).mean(dim=1)

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
        "algorithm": ALGORITHM,
        "state_encoding": STATE_ENCODING,
        "state_dimensions": STATE_DIMENSIONS,
        "action_protocol": ACTION_PROTOCOL,
        "branch_names": list(BRANCH_NAMES),
        "branch_sizes": list(BRANCH_SIZES),
        "hidden_dimensions": list(HIDDEN_DIMENSIONS),
        "control_tick_ms": control_tick_ms,
        "global_step": global_step,
        "episodes": episodes,
    }


def validate_checkpoint(
    checkpoint: Mapping[str, object],
    control_tick_ms: int = DEFAULT_CONTROL_TICK_MS,
) -> None:
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "algorithm": ALGORITHM,
        "state_encoding": STATE_ENCODING,
        "state_dimensions": STATE_DIMENSIONS,
        "action_protocol": ACTION_PROTOCOL,
        "branch_names": list(BRANCH_NAMES),
        "branch_sizes": list(BRANCH_SIZES),
        "control_tick_ms": control_tick_ms,
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
) -> None:
    item = checkpoint_metadata(global_step, episodes, control_tick_ms)
    item.update(
        {
            "online_state_dict": online.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(item, path)


def load_checkpoint(
    path: Path,
    online: BranchingDQN,
    target: BranchingDQN,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reset: bool,
    control_tick_ms: int = DEFAULT_CONTROL_TICK_MS,
) -> tuple[int, int]:
    if reset:
        target.load_state_dict(online.state_dict())
        return 0, 0
    if not path.exists():
        target.load_state_dict(online.state_dict())
        return 0, 0
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, control_tick_ms)
    online.load_state_dict(checkpoint["online_state_dict"])
    target.load_state_dict(checkpoint.get("target_state_dict", checkpoint["online_state_dict"]))
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
    boss_hits: int = 0
    player_hurts: int = 0
    dodges: int = 0
    won: bool = False

    def as_dict(self, epsilon: float) -> dict[str, object]:
        item = asdict(self)
        item["mean_loss"] = self.loss_total / self.losses if self.losses else None
        item["epsilon"] = epsilon
        del item["loss_total"]
        del item["losses"]
        return item


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
        self.replay = ReplayBuffer()
        self.reward_tracker = RewardTracker()
        self.previous_state: StateFrame | None = None
        self.previous_action: tuple[int, ...] | None = None
        self.metrics = EpisodeMetrics(episode=episodes + 1)

    def observe(self, snapshot: Mapping[str, object]) -> RewardFrame:
        reward = self.reward_tracker.step(snapshot)
        if reward.player_hurt < 0:
            self.executor.release_all()
        masks, mask_reasons = branch_availability(snapshot)
        state = encode_snapshot(snapshot, self.executor.control_state(snapshot))
        if (
            self.learning_enabled
            and self.previous_state is not None
            and self.previous_action is not None
        ):
            self.replay.append(
                Transition(
                    state=self.previous_state.observation,
                    action=self.previous_action,
                    reward=reward.total,
                    next_state=state.observation,
                    done=reward.terminated,
                    next_action_masks=masks,
                )
            )
            self.metrics.steps += 1
            self.metrics.reward += reward.total
            self.metrics.boss_hits += int(reward.boss_hit > 0)
            self.metrics.player_hurts += int(reward.player_hurt < 0)
            self.metrics.dodges += int(reward.dodge > 0)
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
        )
        self.executor.apply(
            action,
            branch_masks=masks,
            masked_reasons=mask_reasons,
            player_resources=state.resources,
        )
        self.previous_state = state
        self.previous_action = action
        return reward

    def finish_episode(self) -> dict[str, object]:
        result = self.metrics.as_dict(epsilon_for_step(self.global_step))
        self.completed_episodes += 1
        self.executor.release_all()
        self.reward_tracker.reset()
        self.previous_state = None
        self.previous_action = None
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
    global_step, episodes = load_checkpoint(
        args.checkpoint, online, target, optimizer, device, args.reset, args.tick_ms
    )
    online.train()
    target.eval()
    recorder = ActionRecorder(args.action_log)
    executor = KeyboardActionExecutor(
        recorder,
        tick_ms=args.tick_ms,
        send_input=args.execute_actions,
    )
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
    )
    tail = TelemetryTail(args.telemetry)
    process: subprocess.Popen[bytes] | None = None
    in_arena = False
    try:
        if args.launch:
            process = subprocess.Popen([str(args.game_exe)], cwd=str(args.game_exe.parent))
            print(f"started Silksong pid={process.pid}", flush=True)
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
                        )
                        print(json.dumps(metric), flush=True)
                    in_arena = False
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
                    )
                    print(json.dumps(metric), flush=True)
                    in_arena = False
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Silksong exited with code {process.returncode}")
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
