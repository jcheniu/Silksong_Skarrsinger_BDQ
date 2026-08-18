"""Live joint-action Double DQN for the Silksong telemetry/action pipeline."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import json
import math
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
    place_game_window_top_left_quarter,
    validate_action,
    validate_masks,
)
from .final_project.action_recorder import ActionRecorder
from .real_actions import (
    ACTION_LABELS,
    BRANCH_INDEX,
    COMBAT_EXPLORATION_WEIGHTS,
    EPSILON_DECAY_TRANSITIONS,
    EPSILON_END,
    EPSILON_RECIPROCAL_SHAPE,
    EPSILON_START,
    EXPLORATION_ACTIVATION_RATES,
    JOINT_ACTIONS,
    JOINT_ACTION_COUNT,
    JOINT_ACTION_INDEX,
    MOVEMENT_EXPLORATION_WEIGHTS,
    AttackOpportunity,
    ActionExplorationState,
    apply_attack_opportunity_mask,
    attack_opportunity,
    coordinate_temporal_action,
    danger_requires_commitment_break,
    decode_joint_action,
    epsilon_for_transition,
    _entity_value,
    joint_action_id,
    joint_action_mask,
    select_action,
)
from .real_replay import (
    REPLAY_CAPACITY,
    REPLAY_CHECKPOINT_VERSION,
    ReplayBuffer,
    Transition,
    mirror_action,
    mirror_observation,
    mirror_transition,
)
from .real_reward import (
    ARENA_BOUNDARY_PENALTY,
    ARENA_BOUNDARY_FRACTION,
    ATTACK_ANIMATION_COMMITMENT_STEPS,
    ATTACK_END_GRACE_SECONDS,
    ATTACK_MISS_PENALTY,
    BOSS_PROXIMITY_REWARD,
    BOSS_PROXIMITY_X,
    BOSS_PROXIMITY_Y,
    CHARGE_RELEASE_COMMITMENT_STEPS,
    COMBAT_HURT_EVENT_PENALTY,
    COMBAT_HURT_CREDIT_DECAY,
    COMBAT_HURT_CREDIT_WINDOW_STEPS,
    CONTACT_RISK_INCREASE_SCALE,
    CREDIT_FINALIZATION_STEPS,
    DAMAGE_CREDIT_WINDOW_STEPS,
    DODGE_REWARD,
    EVADE_FAILURE_PENALTY,
    EVADE_SUCCESS_REWARD,
    HARPOON_SUCCESS_BONUS_FRACTION,
    ILLEGAL_ACTION_PENALTY,
    LONG_NO_DAMAGE_PENALTY,
    LONG_NO_DAMAGE_STEPS,
    SPELL_MISS_PENALTY,
    SPELL_ANIMATION_COMMITMENT_STEPS,
    STAGNATION_REGION_FRACTION,
    STAGNATION_PENALTY,
    STAGNATION_WINDOW_STEPS,
    ZERO_SPACE_HURT_WINDOW_SECONDS,
    ZERO_SPACE_ENTRY_PENALTY,
    ZERO_SPACE_HOLD_PENALTY,
    ZERO_SPACE_HURT_PENALTY,
    ZERO_SPACE_OFFENSIVE_CLAWBACK_FRACTION,
    ZERO_SPACE_RADIUS_X,
    ZERO_SPACE_RADIUS_Y,
    ActionOutcomeTrial,
    BossAttackCreditWindow,
    CombatRiskTrial,
    EpisodeMetrics,
    PendingTransition,
    RewardCreditMixin,
    RewardFrame,
    RewardTracker,
)
from .real_state import (
    STATE_DIMENSIONS,
    StateFrame,
    encode_snapshot,
)


STATE_ENCODING = "real-telemetry-state-v15-collision-risk-split-spin-24"
ALGORITHM = "joint-dueling-double-dqn"
CHECKPOINT_VERSION = 32
REWARD_PROTOCOL = "normalized-evade-budget-v25-zero-space-curated-53"
HIDDEN_DIMENSIONS = (96, 96)
LEARNING_RATE = 1e-4
GAMMA = 0.995
BATCH_SIZE = 128
REPLAY_WARMUP = 2_000
PURE_EXPLORATION_STEPS = 0  # Compatibility only; exploration now decays by episode.
TARGET_UPDATE_INTERVAL = 1_000
EVALUATION_INTERVAL_EPISODES = 10
GRADIENT_CLIP_NORM = 10.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "real_dqn.pt"
DEFAULT_METRICS = PROJECT_ROOT / "runs" / "real_dqn.jsonl"
DEFAULT_ACTION_LOG = PROJECT_ROOT / "runs" / "real_dqn_actions.jsonl"
DEFAULT_CONTROL_TICK_MS = 50
GAME_RELAUNCH_DELAY_SECONDS = 1.0
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


# Compatibility name retained for checkpoints/tests; the model has one
# coordinated 53-action advantage output, not three independent Q heads.
BranchingDQN = JointDQN


def optimize_model(
    online: JointDQN,
    target: JointDQN,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
    symmetry_rng: random.Random | None = None,
) -> float:
    if not transitions:
        raise ValueError("transitions must not be empty")
    if symmetry_rng is not None:
        transitions = tuple(
            mirror_transition(item) if symmetry_rng.random() < 0.5 else item
            for item in transitions
        )
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
        "joint_actions": [list(action) for action in JOINT_ACTIONS],
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
        "joint_actions": [list(action) for action in JOINT_ACTIONS],
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


class LiveTrainer(RewardCreditMixin):
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
        self.previous_started_branches: tuple[str, ...] = ()
        self.previous_charge_released = False
        self.previous_attack_opportunity: AttackOpportunity | None = None
        self.pending_credit_transitions: deque[PendingTransition] = deque()
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
        self.previous_macro_id = 0
        self.current_macro_key: tuple[str, int] | None = None
        self.next_macro_id = 1
        self.recent_macro_ids: deque[int] = deque(maxlen=2)
        self.current_macro_positions: deque[float] = deque(
            maxlen=STAGNATION_WINDOW_STEPS + 1
        )
        self.no_damage_steps = 0
        self.episode_finalized_reward = 0.0
        self.credit_step = 0
        self.evaluation_mode = False
        self.proximity_reward_available = True
        self.proximity_reward_balance = 0.0
        self.was_inside_boss_proximity = False
        self.was_inside_zero_space = False
        self.attack_end_grace_steps = max(
            1, math.ceil(ATTACK_END_GRACE_SECONDS * 1000.0 / self.executor.tick_ms)
        )
        self.zero_space_hurt_window_steps = max(
            1,
            math.ceil(
                ZERO_SPACE_HURT_WINDOW_SECONDS
                * 1000.0
                / self.executor.tick_ms
            ),
        )

    def current_epsilon(self) -> float:
        if self.evaluation_mode:
            return 0.0
        return epsilon_for_transition(self.global_step)

    def start_evaluation(self) -> None:
        if self.previous_state is not None:
            raise RuntimeError("cannot start evaluation during an active episode")
        self.evaluation_mode = True
        self.metrics = EpisodeMetrics(
            episode=self.completed_episodes,
            evaluation=True,
        )

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
        state = encode_snapshot(snapshot, self.executor.control_state(snapshot))
        masks, mask_reasons = branch_availability(
            snapshot,
            self.executor.continuing_action,
            harpoon_locked=self.executor.harpoon_locked,
            charge_protected=self.executor.charge_protected,
            charge_must_hold=self.executor.charge_must_hold,
        )
        opportunity = attack_opportunity(state)
        masks, range_reasons = apply_attack_opportunity_mask(
            masks,
            opportunity,
            self.executor.continuing_action,
        )
        mask_reasons = (*mask_reasons, *range_reasons)
        if range_reasons:
            self.metrics.out_of_range_attack_frames += 1
        if (
            (self.learning_enabled or self.evaluation_mode)
            and self.previous_state is not None
            and self.previous_action is not None
        ):
            transition_reward = (
                reward.total
                - reward.damage_reward
                - reward.dodge
                - reward.parry_reward
                - reward.player_hurt
                + self.previous_illegal_penalty
            )
            transition = Transition(
                state=self.previous_state.observation,
                action=joint_action_id(self.previous_action),
                reward=transition_reward,
                next_state=state.observation,
                done=reward.terminated,
                next_action_mask=joint_action_mask(masks),
            )
            pending = PendingTransition(
                transition,
                self.credit_step,
                macro_id=self.previous_macro_id,
            )
            self.pending_credit_transitions.append(pending)
            self._register_action_outcome(pending)
            self._register_combat_risk(pending)
            self.metrics.steps += 1
            self.metrics.reward += transition_reward
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
            self._apply_zero_space_shaping(state, pending)
            self._apply_action_outcomes(reward, state)
            self._apply_player_hurt_credit(reward)
            self._apply_zero_space_hurt_credit(reward)
            self._apply_dense_shaping(reward, state, pending)
            if reward.player_damage_taken > 0:
                self.current_macro_key = None
                self.current_macro_positions.clear()
            self._update_combat_risk_overlap(snapshot, state)
            self._apply_combat_hurt(reward.player_damage_taken)
            self._apply_attack_events(snapshot, state, reward, pending)
            self._resolve_mature_attack_windows()
            self._age_combat_risks()
            self.credit_step += 1
            self._finalize_pending()
            if not self.evaluation_mode:
                self.global_step += 1
                if len(self.replay) >= REPLAY_WARMUP:
                    loss = optimize_model(
                        self.online,
                        self.target,
                        self.optimizer,
                        self.replay.sample(BATCH_SIZE, self.rng),
                        self.device,
                        symmetry_rng=self.rng,
                    )
                    self.metrics.gradient_updates += 1
                    self.metrics.loss_total += loss
                if self.global_step % TARGET_UPDATE_INTERVAL == 0:
                    self.target.load_state_dict(self.online.state_dict())

        if reward.terminated:
            self.metrics.won = reward.boss_dead
            self.executor.release_all()
            return reward

        if (
            danger_requires_commitment_break(state)
            or reward.player_damage_taken > 0
            or bool(self.previous_illegal_branches)
        ):
            self.action_exploration_state.clear()

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
        self.previous_attack_opportunity = opportunity
        executed_action = action_result.get("action_vector", action)
        self.previous_action = validate_action(executed_action)
        temporal_owner = action_result.get("temporal_owner")
        player_x = _entity_value(state.player, "x")
        self._record_macro_action(
            self.previous_action,
            str(temporal_owner) if temporal_owner is not None else None,
            player_x,
        )
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
        self.action_exploration_state.reconcile(self.previous_action)
        raw_started = action_result.get("started_branches", ())
        self.previous_started_branches = tuple(str(value) for value in raw_started)
        self.previous_charge_released = bool(
            action_result.get("charge_released", False)
        )
        raw_illegal_branches = action_result.get("illegal_branches", ())
        self.previous_illegal_branches = tuple(
            str(value) for value in raw_illegal_branches
        )
        self.previous_illegal_penalty = (
            ILLEGAL_ACTION_PENALTY if self.previous_illegal_branches else 0.0
        )
        return reward

    def finish_episode(self) -> dict[str, object]:
        was_evaluation = self.evaluation_mode
        self._expire_action_outcomes()
        for attack_id in list(self.attack_windows):
            self._resolve_attack_window(attack_id)
        self._finalize_pending(force=True)
        self.metrics.reward = self.episode_finalized_reward
        result = self.metrics.as_dict(
            self.current_epsilon(),
            len(self.replay),
            self.global_step,
        )
        if not was_evaluation:
            self.completed_episodes += 1
        self.evaluation_mode = False
        self.executor.release_all()
        self.reward_tracker.reset()
        self.previous_state = None
        self.previous_action = None
        self.previous_illegal_penalty = 0.0
        self.previous_illegal_branches = ()
        self.previous_started_branches = ()
        self.previous_charge_released = False
        self.previous_attack_opportunity = None
        self.action_exploration_state.clear()
        self.combat_risk_trials.clear()
        self.attack_windows.clear()
        self.active_attack_id = 0
        self.previous_macro_id = 0
        self.current_macro_key = None
        self.recent_macro_ids.clear()
        self.current_macro_positions.clear()
        self.no_damage_steps = 0
        self.episode_finalized_reward = 0.0
        self.credit_step = 0
        self.proximity_reward_available = True
        self.proximity_reward_balance = 0.0
        self.was_inside_boss_proximity = False
        self.was_inside_zero_space = False
        self.metrics = EpisodeMetrics(episode=self.completed_episodes + 1)
        return result

    def reset_interrupted_episode(self) -> None:
        """Keep completed replay entries but break any cross-process transition."""

        for attack_id in list(self.attack_windows):
            self._resolve_attack_window(attack_id)
        self._finalize_pending(force=True)
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
        self.previous_started_branches = ()
        self.previous_charge_released = False
        self.previous_attack_opportunity = None
        self.action_exploration_state.clear()
        self.previous_macro_id = 0
        self.current_macro_key = None
        self.recent_macro_ids.clear()
        self.current_macro_positions.clear()
        self.no_damage_steps = 0
        self.episode_finalized_reward = 0.0
        self.credit_step = 0
        self.proximity_reward_available = True
        self.proximity_reward_balance = 0.0
        self.was_inside_boss_proximity = False
        self.was_inside_zero_space = False
        self.metrics = EpisodeMetrics(
            episode=(self.completed_episodes if self.evaluation_mode else self.completed_episodes + 1),
            evaluation=self.evaluation_mode,
        )


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
        if args.execute_actions and not args.full_window:
            place_game_window_top_left_quarter()
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

    def finish_current_episode() -> dict[str, object]:
        metric = trainer.finish_episode()
        append_metric(args.metrics, metric)
        if not bool(metric["evaluation"]):
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
        if (
            args.execute_actions
            and not bool(metric["evaluation"])
            and trainer.completed_episodes % EVALUATION_INTERVAL_EPISODES == 0
        ):
            trainer.start_evaluation()
            print(
                "starting independent greedy evaluation after "
                f"training episode {trainer.completed_episodes}",
                flush=True,
            )
        return metric

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
        while trainer.completed_episodes < args.episodes or trainer.evaluation_mode:
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
                        finish_current_episode()
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
                    finish_current_episode()
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
                        if args.execute_actions and not args.full_window:
                            place_game_window_top_left_quarter()
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
    parser.add_argument(
        "--full-window",
        action="store_true",
        help="leave the game window at its current size instead of using the top-left quarter",
    )
    parser.add_argument("--keep-game", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    train_live(args)


if __name__ == "__main__":
    main()
