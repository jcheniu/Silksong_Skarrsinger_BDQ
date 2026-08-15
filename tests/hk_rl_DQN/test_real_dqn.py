from pathlib import Path
from dataclasses import FrozenInstanceError
import random
import unittest
from unittest.mock import patch

import torch

from hk_rl_DQN.final_project.action_executor import (
    BRANCH_SIZES,
    KeyboardActionExecutor,
)
from hk_rl_DQN.final_project.action_recorder import ActionRecorder
from hk_rl_DQN.real_dqn import (
    ActionExplorationState,
    ArenaActionWatchdog,
    ArenaResetGate,
    BranchingDQN,
    COMBAT_HURT_EVENT_PENALTY,
    COMBAT_EXPLORATION_WEIGHTS,
    CREDIT_FINALIZATION_STEPS,
    EVADE_FAILURE_PENALTY,
    EVADE_SUCCESS_REWARD,
    JOINT_ACTION_COUNT,
    LiveTrainer,
    PendingTransition,
    PURE_EXPLORATION_STEPS,
    ReplayBuffer,
    Transition,
    apply_attack_opportunity_mask,
    attack_opportunity,
    checkpoint_metadata,
    decode_joint_action,
    danger_requires_commitment_break,
    joint_action_id,
    joint_action_mask,
    load_checkpoint,
    optimize_model,
    save_checkpoint,
    select_action,
    validate_checkpoint,
)
from hk_rl_DQN.real_state import STATE_DIMENSIONS, encode_snapshot
from hk_rl_DQN.real_reward import ILLEGAL_ACTION_PENALTY, STEP_PENALTY


ALL_ACTIONS_AVAILABLE = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
ONLY_NEUTRAL_AVAILABLE = tuple(
    tuple(index == 0 for index in range(size)) for size in BRANCH_SIZES
)


def transition(
    action: tuple[int, ...] = (0, 0, 0),
    reward: float = 0.0,
    done: bool = False,
) -> Transition:
    return Transition(
        state=(0.0,) * STATE_DIMENSIONS,
        action=joint_action_id(action),
        reward=reward,
        next_state=(0.5,) * STATE_DIMENSIONS,
        done=done,
        next_action_mask=joint_action_mask(ALL_ACTIONS_AVAILABLE),
    )


def snapshot(state: str, frame: int, health: int = 10) -> dict[str, object]:
    return {
        "type": "snapshot",
        "timestamp": frame / 10.0,
        "frame": frame,
        "scene": "Memory_Ant_Queen",
        "encounter_active": True,
        "player_grounded": True,
        "player": {"x": 150.0, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
        "player_health": {"health": health, "max_health": 10},
        "boss": {"x": 154.0, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
        "boss_vulnerable": True,
        "fsm": [
            {"path": "Boss Scene/Hunter Queen Boss", "name": "Control", "state": state},
            {"path": "Boss Scene/Hunter Queen Boss", "name": "Stun Control", "state": "Idle"},
        ],
    }


def with_attack_event(
    item: dict[str, object],
    *,
    attack_id: int,
    phase: str,
    started: int = 1,
    active: int = 1,
    finished: int = 0,
    hits: int = 0,
    last_finished_id: int = 0,
    last_hit_id: int = 0,
) -> dict[str, object]:
    item["boss_attack"] = {
        "id": attack_id,
        "type": "slash",
        "phase": phase,
        "started_events": started,
        "active_events": active,
        "finished_events": finished,
        "player_hit_events": hits,
        "last_finished_id": last_finished_id,
        "last_player_hit_id": last_hit_id,
    }
    return item


class BranchingDQNTests(unittest.TestCase):
    def test_terminal_snapshots_are_ignored_until_arena_exit(self) -> None:
        gate = ArenaResetGate()
        self.assertTrue(gate.allow_snapshot(True))
        gate.mark_episode_finished()
        self.assertFalse(gate.allow_snapshot(True))
        self.assertFalse(gate.allow_snapshot(True))
        self.assertFalse(gate.allow_snapshot(False))
        self.assertTrue(gate.allow_snapshot(True))

    def test_new_encounter_id_unlocks_without_inactive_snapshot(self) -> None:
        gate = ArenaResetGate()
        self.assertTrue(gate.allow_snapshot(True, 10))
        gate.mark_episode_finished()
        self.assertFalse(gate.allow_snapshot(True, 10))
        self.assertTrue(gate.allow_snapshot(True, 11))

    def test_action_watchdog_only_recovers_after_timeout(self) -> None:
        watchdog = ArenaActionWatchdog(timeout_seconds=1.0)
        self.assertFalse(watchdog.stalled(5.0))
        watchdog.record(5.0)
        self.assertFalse(watchdog.stalled(5.9))
        self.assertTrue(watchdog.stalled(6.0))
        watchdog.reset()
        self.assertFalse(watchdog.stalled(10.0))

    def test_network_scores_every_joint_action(self) -> None:
        network = BranchingDQN()
        values = network(torch.zeros(4, STATE_DIMENSIONS))
        self.assertEqual(tuple(values.shape), (4, JOINT_ACTION_COUNT))

    def test_random_action_is_valid(self) -> None:
        action = select_action(
            BranchingDQN(),
            (0.0,) * STATE_DIMENSIONS,
            epsilon=1.0,
            rng=random.Random(3),
            device=torch.device("cpu"),
        )
        self.assertTrue(all(0 <= value < size for value, size in zip(action, BRANCH_SIZES)))

    def test_greedy_action_respects_branch_masks(self) -> None:
        network = BranchingDQN()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.advantage.bias[-1] = 100.0
        action = select_action(
            network,
            (0.0,) * STATE_DIMENSIONS,
            epsilon=0.0,
            rng=random.Random(3),
            device=torch.device("cpu"),
            branch_masks=ONLY_NEUTRAL_AVAILABLE,
        )
        self.assertEqual(action, (0,) * len(BRANCH_SIZES))

    def test_random_exploration_only_uses_available_actions(self) -> None:
        for seed in range(20):
            action = select_action(
                BranchingDQN(),
                (0.0,) * STATE_DIMENSIONS,
                epsilon=1.0,
                rng=random.Random(seed),
                device=torch.device("cpu"),
                branch_masks=ONLY_NEUTRAL_AVAILABLE,
            )
            self.assertEqual(action, (0,) * len(BRANCH_SIZES))

    def test_exploration_keeps_v_rare_but_still_explores_combinations(self) -> None:
        actions = [
            select_action(
                BranchingDQN(),
                (0.0,) * STATE_DIMENSIONS,
                epsilon=1.0,
                rng=random.Random(seed),
                device=torch.device("cpu"),
                branch_masks=ALL_ACTIONS_AVAILABLE,
            )
            for seed in range(500)
        ]
        taunts = sum(action[2] == 4 for action in actions)
        self.assertGreater(taunts, 0)
        self.assertLess(taunts, 50)
        self.assertTrue(any(sum(value != 0 for value in action) >= 2 for action in actions))

    def test_exploration_uses_sparse_branch_activation(self) -> None:
        class CountingRandom:
            def __init__(self) -> None:
                self.random_calls = 0

            def random(self) -> float:
                self.random_calls += 1
                return 0.0

            def choice(self, values: list[int]) -> int:
                return values[0]

        rng = CountingRandom()
        select_action(
            BranchingDQN(),
            (0.0,) * STATE_DIMENSIONS,
            epsilon=1.0,
            rng=rng,
            device=torch.device("cpu"),
            branch_masks=ALL_ACTIONS_AVAILABLE,
        )
        self.assertEqual(rng.random_calls, 3 + len(BRANCH_SIZES))

    def test_combat_exploration_prefers_attacks_but_keeps_taunt_nonzero(self) -> None:
        network = BranchingDQN()
        rng = random.Random(119)
        combats = [
            select_action(
                network,
                (0.0,) * STATE_DIMENSIONS,
                epsilon=1.0,
                rng=rng,
                device=torch.device("cpu"),
                branch_masks=ALL_ACTIONS_AVAILABLE,
            )[2]
            for _ in range(3000)
        ]
        attack_count = sum(combats.count(value) for value in (1, 2, 5, 6))
        self.assertGreater(attack_count, combats.count(4) * 20)
        self.assertGreater(combats.count(4), 0)
        self.assertEqual(len(COMBAT_EXPLORATION_WEIGHTS), BRANCH_SIZES[2])

    def test_cold_start_uses_pure_exploration_before_normal_decay(self) -> None:
        path = Path("tests/.real_dqn_epsilon.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            self.assertEqual(trainer.current_epsilon(), 1.0)
            trainer.global_step = PURE_EXPLORATION_STEPS - 1
            self.assertEqual(trainer.current_epsilon(), 1.0)
            trainer.global_step = PURE_EXPLORATION_STEPS
            self.assertEqual(trainer.current_epsilon(), 0.60)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_selected_q_values_reuse_action_forward_pass(self) -> None:
        selected_q_values: list[float] = []
        action = select_action(
            BranchingDQN(),
            (0.0,) * STATE_DIMENSIONS,
            epsilon=0.0,
            rng=random.Random(2),
            device=torch.device("cpu"),
            selected_q_values=selected_q_values,
        )
        self.assertEqual(len(action), len(BRANCH_SIZES))
        self.assertEqual(len(selected_q_values), 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(selected_q_values))))

    def test_movement_exploration_prefers_direction_but_keeps_harpoon_nonzero(self) -> None:
        network = BranchingDQN()
        rng = random.Random(91)
        movements = [
            select_action(
                network,
                (0.0,) * STATE_DIMENSIONS,
                epsilon=1.0,
                rng=rng,
                device=torch.device("cpu"),
                branch_masks=ALL_ACTIONS_AVAILABLE,
            )[1]
            for _ in range(2000)
        ]
        counts = {value: movements.count(value) for value in range(BRANCH_SIZES[1])}
        self.assertGreater(counts[1] + counts[2], sum(counts[value] for value in (3, 4, 5, 6)))
        self.assertGreater(counts[1], counts[6])
        self.assertGreater(counts[2], counts[6])
        self.assertGreater(counts[6], 0)

    def test_exploratory_direction_persists_for_two_or_three_total_ticks(self) -> None:
        network = BranchingDQN()
        for seed in range(100):
            state = ActionExplorationState()
            first = select_action(
                network,
                (0.0,) * STATE_DIMENSIONS,
                epsilon=1.0,
                rng=random.Random(seed),
                device=torch.device("cpu"),
                branch_masks=ALL_ACTIONS_AVAILABLE,
                exploration_state=state,
            )
            if first[1] not in (1, 2):
                continue
            additional_ticks = state.remaining_ticks
            self.assertIn(additional_ticks, (1, 2))
            repeated = [
                select_action(
                    network,
                    (0.0,) * STATE_DIMENSIONS,
                    epsilon=0.0,
                    rng=random.Random(1000 + index),
                    device=torch.device("cpu"),
                    branch_masks=ALL_ACTIONS_AVAILABLE,
                    exploration_state=state,
                )[1]
                for index in range(additional_ticks)
            ]
            self.assertEqual(repeated, [first[1]] * additional_ticks)
            break
        else:
            self.fail("no exploratory left/right action was selected")

    def test_greedy_jump_and_direction_obey_short_commitment(self) -> None:
        network = BranchingDQN()
        state = ActionExplorationState()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.advantage.bias[joint_action_id((1, 1, 0))] = 10.0
        first = select_action(
            network,
            (0.0,) * STATE_DIMENSIONS,
            epsilon=0.0,
            rng=random.Random(4),
            device=torch.device("cpu"),
            branch_masks=ALL_ACTIONS_AVAILABLE,
            exploration_state=state,
        )
        self.assertEqual(first, (1, 1, 0))
        with torch.no_grad():
            network.advantage.bias.zero_()
            network.advantage.bias[joint_action_id((0, 2, 0))] = 100.0
        committed = select_action(
            network,
            (0.0,) * STATE_DIMENSIONS,
            epsilon=0.0,
            rng=random.Random(5),
            device=torch.device("cpu"),
            branch_masks=ALL_ACTIONS_AVAILABLE,
            exploration_state=state,
        )
        self.assertEqual(committed[:2], (2, 1))

    def test_attack_opportunity_has_confirmed_fringe_and_hard_mask_zones(self) -> None:
        close = encode_snapshot(snapshot("Movement 1", 1))
        self.assertTrue(attack_opportunity(close).horizontal_confirmed)

        fringe_snapshot = snapshot("Movement 1", 2)
        fringe_snapshot["boss"]["x"] = 157.2
        fringe = attack_opportunity(encode_snapshot(fringe_snapshot))
        self.assertTrue(fringe.horizontal_allowed)
        self.assertFalse(fringe.horizontal_confirmed)

        facing_away_snapshot = snapshot("Movement 1", 2)
        facing_away_snapshot["player"]["facing"] = -1.0
        facing_away = attack_opportunity(encode_snapshot(facing_away_snapshot))
        self.assertTrue(facing_away.horizontal_allowed)
        self.assertFalse(facing_away.horizontal_confirmed)

        far_snapshot = snapshot("Movement 1", 3)
        far_snapshot["boss"]["x"] = 165.0
        far = attack_opportunity(encode_snapshot(far_snapshot))
        masked, reasons = apply_attack_opportunity_mask(
            ALL_ACTIONS_AVAILABLE,
            far,
            (0, 0, 0),
        )
        self.assertFalse(masked[2][1])
        self.assertFalse(masked[2][2])
        self.assertTrue(any("outside predictive range" in reason for reason in reasons))

        invulnerable_snapshot = snapshot("Movement 1", 4)
        invulnerable_snapshot["boss_vulnerable"] = False
        invulnerable = attack_opportunity(encode_snapshot(invulnerable_snapshot))
        self.assertFalse(invulnerable.boss_vulnerable)
        self.assertFalse(invulnerable.horizontal_allowed)

    def test_active_boss_attack_breaks_action_commitment(self) -> None:
        self.assertTrue(
            danger_requires_commitment_break(
                encode_snapshot(snapshot("Slash 1", 1))
            )
        )

    def test_double_dqn_optimization_is_finite(self) -> None:
        online = BranchingDQN()
        target = BranchingDQN()
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
        transitions = [
            transition(
                tuple(index % size for size in BRANCH_SIZES),
                float(index),
                index == 3,
            )
            for index in range(4)
        ]
        loss = optimize_model(online, target, optimizer, transitions, torch.device("cpu"))
        self.assertGreaterEqual(loss, 0.0)

    def test_double_dqn_target_uses_joint_action_mask(self) -> None:
        online = BranchingDQN()
        target = BranchingDQN()
        with torch.no_grad():
            for network in (online, target):
                for parameter in network.parameters():
                    parameter.zero_()
            online.advantage.bias[1] = 10.0
            target.advantage.bias[1] = 20.0
        transition = Transition(
            state=(0.0,) * STATE_DIMENSIONS,
            action=joint_action_id((0, 0, 0)),
            reward=0.0,
            next_state=(0.0,) * STATE_DIMENSIONS,
            done=False,
            next_action_mask=joint_action_mask(ONLY_NEUTRAL_AVAILABLE),
        )
        with torch.no_grad():
            online_values = online(torch.zeros(1, STATE_DIMENSIONS))
            target_values = target(torch.zeros(1, STATE_DIMENSIONS))
            current = online_values[:, joint_action_id((0, 0, 0))]
            masked_next = target_values[:, joint_action_id((0, 0, 0))]
            expected_loss = torch.nn.functional.smooth_l1_loss(
                current,
                0.99 * masked_next,
            ).item()
        optimizer = torch.optim.AdamW(online.parameters(), lr=0.0)
        loss = optimize_model(online, target, optimizer, [transition], torch.device("cpu"))
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_checkpoint_metadata_validates(self) -> None:
        item = checkpoint_metadata(10, 2)
        item["online_state_dict"] = {}
        validate_checkpoint(item)

    def test_previous_state_protocol_checkpoint_is_rejected(self) -> None:
        item = checkpoint_metadata(10, 2)
        item["online_state_dict"] = {}
        item["checkpoint_version"] = 1
        item["state_dimensions"] = STATE_DIMENSIONS - 1
        with self.assertRaises(ValueError):
            validate_checkpoint(item)

    def test_missing_checkpoint_starts_new_without_reset_flag(self) -> None:
        online = BranchingDQN()
        target = BranchingDQN()
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
        path = Path("tests/.missing_checkpoint.pt")
        path.unlink(missing_ok=True)
        step, episodes = load_checkpoint(
            path,
            online,
            target,
            optimizer,
            torch.device("cpu"),
            reset=False,
        )
        self.assertEqual((step, episodes), (0, 0))
        for online_value, target_value in zip(
            online.state_dict().values(), target.state_dict().values()
        ):
            self.assertTrue(torch.equal(online_value, target_value))

    def test_incompatible_checkpoint_never_resets_implicitly(self) -> None:
        online = BranchingDQN()
        target = BranchingDQN()
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
        item = checkpoint_metadata(100, 5)
        item["checkpoint_version"] = 1
        item["online_state_dict"] = online.state_dict()
        path = Path("tests/.incompatible_checkpoint.pt")
        try:
            torch.save(item, path)
            with self.assertRaises(ValueError):
                load_checkpoint(
                    path,
                    online,
                    target,
                    optimizer,
                    torch.device("cpu"),
                    reset=False,
                )
        finally:
            path.unlink(missing_ok=True)

    def test_reset_must_be_explicit_to_ignore_existing_checkpoint(self) -> None:
        online = BranchingDQN()
        target = BranchingDQN()
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
        path = Path("tests/.explicit_reset_checkpoint.pt")
        try:
            torch.save({"checkpoint_version": 1}, path)
            result = load_checkpoint(
                path,
                online,
                target,
                optimizer,
                torch.device("cpu"),
                reset=True,
            )
            self.assertEqual(result, (0, 0))
        finally:
            path.unlink(missing_ok=True)

    def test_replay_buffer_tensor_state_round_trip(self) -> None:
        replay = ReplayBuffer()
        replay.append(
            Transition(
                state=(0.25,) * STATE_DIMENSIONS,
                action=joint_action_id((2, 5, 6)),
                reward=1.5,
                next_state=(-0.25,) * STATE_DIMENSIONS,
                done=True,
                next_action_mask=joint_action_mask(ALL_ACTIONS_AVAILABLE),
            )
        )
        restored = ReplayBuffer()
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(len(restored), 1)
        item = restored._items[0]
        self.assertEqual(item.action_vector, (2, 5, 6))
        self.assertEqual(item.done, True)
        self.assertAlmostEqual(item.reward, 1.5)
        self.assertEqual(item.next_action_mask, joint_action_mask(ALL_ACTIONS_AVAILABLE))

    def test_checkpoint_restores_replay_without_new_warmup(self) -> None:
        path = Path("tests/.replay_checkpoint.pt")
        online = BranchingDQN()
        target = BranchingDQN()
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
        replay = ReplayBuffer()
        for index in range(1000):
            replay.append(
                transition(
                    tuple(index % size for size in BRANCH_SIZES),
                    float(index) / 1000,
                )
            )
        try:
            save_checkpoint(path, online, target, optimizer, 1200, 3, replay=replay)
            restored_online = BranchingDQN()
            restored_target = BranchingDQN()
            restored_optimizer = torch.optim.AdamW(
                restored_online.parameters(), lr=1e-4
            )
            restored_replay = ReplayBuffer()
            result = load_checkpoint(
                path,
                restored_online,
                restored_target,
                restored_optimizer,
                torch.device("cpu"),
                reset=False,
                replay=restored_replay,
            )
            self.assertEqual(result, (1200, 3))
            self.assertEqual(len(restored_replay), 1000)
            self.assertGreaterEqual(len(restored_replay), 1000)
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)

    def test_explicit_reset_clears_restored_replay_target(self) -> None:
        replay = ReplayBuffer()
        replay.append(transition())
        online = BranchingDQN()
        result = load_checkpoint(
            Path("tests/.unused_reset_checkpoint.pt"),
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            torch.device("cpu"),
            reset=True,
            replay=replay,
        )
        self.assertEqual(result, (0, 0))
        self.assertEqual(len(replay), 0)

    def test_live_trainer_connects_state_reward_replay_and_action(self) -> None:
        path = Path("tests/.real_dqn_actions.jsonl")
        recorder = ActionRecorder(path)
        executor = KeyboardActionExecutor(recorder, send_input=False)
        online = BranchingDQN()
        target = BranchingDQN()
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
        trainer = LiveTrainer(
            online,
            target,
            optimizer,
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            trainer.observe(snapshot("Slash Antic", 1))
            trainer.observe(snapshot("Slash 1", 2))
            self.assertEqual(len(trainer.replay), 0)
            self.assertEqual(len(trainer.pending_credit_transitions), 1)
            for frame in range(3, CREDIT_FINALIZATION_STEPS + 3):
                trainer.observe(snapshot("Movement 1", frame))
            self.assertGreaterEqual(len(trainer.replay), 1)
            self.assertEqual(trainer.global_step, CREDIT_FINALIZATION_STEPS + 1)
            self.assertIsNotNone(trainer.previous_action)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_inactive_lifecycle_snapshot_writes_terminal_transition(self) -> None:
        path = Path("tests/.real_dqn_lifecycle_terminal.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            with patch("hk_rl_DQN.real_dqn.select_action", return_value=(0, 1, 0)):
                trainer.observe(snapshot("Movement 1", 1))
            inactive = snapshot("", 2)
            inactive["encounter_active"] = False
            inactive["boss"] = None
            trainer.observe(inactive, force_terminal=True)
            metric = trainer.finish_episode()
            self.assertEqual(len(trainer.replay), 1)
            self.assertTrue(trainer.replay._items[0].done)
            self.assertEqual(metric["replay_size"], 1)
            self.assertEqual(metric["global_step"], 1)
            self.assertEqual(metric["gradient_updates"], 0)
            self.assertEqual(metric["actual_action_counts"]["movement"]["hold_left"], 1)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_interrupted_episode_reset_breaks_cross_process_transition(self) -> None:
        path = Path("tests/.real_dqn_interrupted_episode.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            with patch("hk_rl_DQN.real_dqn.select_action", return_value=(0, 1, 0)):
                trainer.observe(snapshot("Movement 1", 1))
            trainer.reset_interrupted_episode()
            self.assertIsNone(trainer.previous_state)
            self.assertIsNone(trainer.previous_action)
            self.assertEqual(trainer.completed_episodes, 0)
            self.assertEqual(trainer.metrics.episode, 1)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_replay_uses_the_executor_actual_action(self) -> None:
        path = Path("tests/.real_dqn_executed_action.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        masks = ALL_ACTIONS_AVAILABLE
        try:
            with patch("hk_rl_DQN.real_dqn.branch_availability", return_value=(masks, ())):
                with patch("hk_rl_DQN.real_dqn.select_action", return_value=(2, 6, 1)):
                    trainer.observe(snapshot("Movement 1", 1))
                with patch("hk_rl_DQN.real_dqn.select_action", return_value=(0,) * len(BRANCH_SIZES)):
                    trainer.observe(snapshot("Movement 1", 2))
            trainer._finalize_pending(force=True)
            transition = trainer.replay.sample(1, random.Random(0))[0]
            self.assertEqual(transition.action_vector, (0, 6, 0))
            self.assertEqual(trainer.previous_action, (0, 0, 0))
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_successful_dodge_backfills_the_attack_window(self) -> None:
        path = Path("tests/.real_dqn_dodge_backfill.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        evade = (0, 1, 0)
        try:
            with patch("hk_rl_DQN.real_dqn.select_action", return_value=evade):
                trainer.observe(snapshot("Slash Antic", 1))
                trainer.observe(snapshot("Slash 1", 2))
                trainer.observe(snapshot("Slash End", 3))
                trainer.observe(snapshot("Movement 1", 4))
                trainer.observe(snapshot("Movement 1", 5))
                trainer.observe(snapshot("Movement 1", 6))
            trainer._finalize_pending(force=True)
            rewards = [item.reward for item in trainer.replay._items]
            self.assertGreater(trainer.metrics.dodge_backfill_reward, 0.0)
            self.assertGreater(sum(rewards), len(rewards) * STEP_PENALTY)
            self.assertAlmostEqual(trainer.metrics.dodge_backfill_reward, EVADE_SUCCESS_REWARD)
            self.assertAlmostEqual(
                sum(item.reward - STEP_PENALTY for item in trainer.replay._items),
                EVADE_SUCCESS_REWARD,
            )
            self.assertGreater(trainer.metrics.movement_dodge_reward, 0.0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_neutral_action_cannot_farm_successful_dodge_reward(self) -> None:
        path = Path("tests/.real_dqn_neutral_dodge.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            with patch("hk_rl_DQN.real_dqn.select_action", return_value=(0, 0, 0)):
                trainer.observe(snapshot("Slash Antic", 1))
                trainer.observe(snapshot("Slash 1", 2))
                trainer.observe(snapshot("Slash End", 3))
                trainer.observe(snapshot("Movement 1", 4))
                trainer.observe(snapshot("Movement 1", 5))
                trainer.observe(snapshot("Movement 1", 6))
            self.assertEqual(trainer.metrics.dodges, 1)
            self.assertEqual(trainer.metrics.dodge_backfill_reward, 0.0)
            self.assertEqual(trainer.metrics.unattributed_dodges, 1)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_neutral_action_still_receives_fixed_failed_dodge_budget(self) -> None:
        path = Path("tests/.real_dqn_neutral_failed_dodge.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            with patch("hk_rl_DQN.real_dqn.select_action", return_value=(0, 0, 0)):
                trainer.observe(snapshot("Slash Antic", 1, health=10))
                trainer.observe(snapshot("Slash 1", 2, health=9))
                trainer.observe(snapshot("Slash End", 3, health=9))
                trainer.observe(snapshot("Movement 1", 4, health=9))
                trainer.observe(snapshot("Movement 1", 5, health=9))
                trainer.observe(snapshot("Movement 1", 6, health=9))
            self.assertEqual(trainer.metrics.failed_dodges, 1)
            self.assertEqual(
                trainer.metrics.evade_failure_backfill_penalty,
                EVADE_FAILURE_PENALTY,
            )
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_failed_dodge_backfills_earlier_jump_and_movement_actions(self) -> None:
        path = Path("tests/.real_dqn_failed_dodge_backfill.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        evade = (1, 1, 0)
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch("hk_rl_DQN.real_dqn.select_action", return_value=evade),
            ):
                trainer.observe(snapshot("Slash Antic", 1, health=10))
                trainer.observe(snapshot("Slash 1", 2, health=9))
                trainer.observe(snapshot("Movement 1", 3, health=9))
                trainer.observe(snapshot("Movement 1", 4, health=9))
                trainer.observe(snapshot("Movement 1", 5, health=9))
            trainer._finalize_pending(force=True)
            self.assertEqual(trainer.metrics.failed_dodges, 1)
            self.assertLess(trainer.metrics.evade_failure_backfill_penalty, 0.0)
            self.assertEqual(trainer.metrics.failed_dodges_by_attack, {"slash": 1})
            shaped_total = sum(
                item.reward - STEP_PENALTY
                for item in trainer.replay._items
            )
            self.assertLessEqual(shaped_total, EVADE_FAILURE_PENALTY)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_taunt_has_light_cost_and_outcome_penalties(self) -> None:
        def make_trainer(path: Path) -> tuple[LiveTrainer, KeyboardActionExecutor]:
            executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
            online = BranchingDQN()
            return (
                LiveTrainer(
                    online,
                    BranchingDQN(),
                    torch.optim.AdamW(online.parameters(), lr=1e-4),
                    executor,
                    torch.device("cpu"),
                    random.Random(1),
                ),
                executor,
            )

        taunt = (0, 0, 4)
        neutral = (0,) * len(BRANCH_SIZES)
        cases = (
            ("miss", {}, -0.02),
            ("hurt", {"health": 9}, -1.02),
            ("hit", {"boss_damage_total": 10}, -0.02),
            ("hit_and_hurt", {"health": 9, "boss_damage_total": 10}, -1.02),
        )
        for name, outcome, expected in cases:
            path = Path(f"tests/.real_dqn_taunt_{name}.jsonl")
            trainer, executor = make_trainer(path)
            try:
                first = snapshot("Movement 1", 1)
                first["boss_damage_total"] = 0
                with patch("hk_rl_DQN.real_dqn.select_action", side_effect=[taunt] + [neutral] * 8):
                    trainer.observe(first)
                    if name == "miss":
                        for frame in range(2, 8):
                            trainer.observe(snapshot("Movement 1", frame))
                    else:
                        second = snapshot(
                            "Movement 1", 2, health=int(outcome.get("health", 10))
                        )
                        if "boss_damage_total" in outcome:
                            second["boss_damage_total"] = outcome["boss_damage_total"]
                        trainer.observe(second)
                self.assertEqual(trainer.metrics.taunt_penalty, expected)
                trainer._finalize_pending(force=True)
                affected = next(
                    item for item in trainer.replay._items if item.action_vector[2] == 4
                )
                expected_reward = STEP_PENALTY + expected
                if "health" in outcome:
                    expected_reward -= 3.0
                self.assertAlmostEqual(
                    affected.reward,
                    expected_reward,
                )
                self.assertEqual(trainer.metrics.unattributed_damage_reward, 0.5 if "boss_damage_total" in outcome else 0.0)
                if "health" in outcome:
                    self.assertEqual(trainer.metrics.combat_hurt_penalty, 0.0)
            finally:
                executor.close()
                path.unlink(missing_ok=True)

    def test_damage_reward_is_credited_to_recent_attack_start(self) -> None:
        path = Path("tests/.real_dqn_damage_credit.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        attack = (0, 0, 1)
        neutral = (0,) * len(BRANCH_SIZES)
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch(
                    "hk_rl_DQN.real_dqn.select_action",
                    side_effect=[attack] + [neutral] * 24,
                ),
            ):
                first = snapshot("Movement 1", 1)
                first["boss_damage_total"] = 0
                second = snapshot("Movement 1", 2)
                second["boss_damage_total"] = 0
                hit = snapshot("Movement 1", 3)
                hit["boss_damage_total"] = 10
                trainer.observe(first)
                trainer.observe(second)
                trainer.observe(hit)
                for frame in range(4, 23):
                    trainer.observe(snapshot("Movement 1", frame))
            credited = trainer.replay._items[0]
            self.assertAlmostEqual(credited.reward, STEP_PENALTY + 0.5)
            self.assertAlmostEqual(trainer.metrics.damage_credit_reward, 0.5)
            self.assertEqual(trainer.metrics.offensive_misses, 0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_harpoon_damage_credits_movement_without_miss_penalty(self) -> None:
        path = Path("tests/.real_dqn_harpoon_credit.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        harpoon = (0, 6, 0)
        neutral = (0,) * len(BRANCH_SIZES)
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch(
                    "hk_rl_DQN.real_dqn.select_action",
                    side_effect=[harpoon] + [neutral] * 24,
                ),
            ):
                first = snapshot("Movement 1", 1)
                first["boss_damage_total"] = 0
                second = snapshot("Movement 1", 2)
                second["boss_damage_total"] = 0
                hit = snapshot("Movement 1", 3)
                hit["boss_damage_total"] = 10
                trainer.observe(first)
                trainer.observe(second)
                trainer.observe(hit)
                for frame in range(4, 23):
                    trainer.observe(snapshot("Movement 1", frame))
            credited = next(
                item for item in trainer.replay._items if item.action_vector[1] == 6
            )
            self.assertAlmostEqual(credited.reward, STEP_PENALTY + 0.5)
            self.assertEqual(trainer.metrics.offensive_misses, 0)
            self.assertAlmostEqual(trainer.metrics.harpoon_damage_reward, 0.5)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_harpoon_without_damage_has_no_miss_penalty(self) -> None:
        path = Path("tests/.real_dqn_harpoon_no_hit.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        neutral = (0,) * len(BRANCH_SIZES)
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch(
                    "hk_rl_DQN.real_dqn.select_action",
                    side_effect=[(0, 6, 0)] + [neutral] * 24,
                ),
            ):
                for frame in range(1, 23):
                    trainer.observe(snapshot("Movement 1", frame))
            harpoon_transition = next(
                item for item in trainer.replay._items if item.action_vector[1] == 6
            )
            self.assertEqual(harpoon_transition.reward, STEP_PENALTY)
            self.assertEqual(trainer.metrics.offensive_misses, 0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_player_parry_rewards_recent_directional_attack_combat_head(self) -> None:
        path = Path("tests/.real_dqn_player_parry.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        up_attack = (0, 0, 5)
        neutral = (0,) * len(BRANCH_SIZES)
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch(
                    "hk_rl_DQN.real_dqn.select_action",
                    side_effect=[up_attack] + [neutral] * 24,
                ),
            ):
                first = snapshot("Slash Antic", 1)
                first["player_parry_events"] = 0
                parry = snapshot("Slash 1", 2)
                parry["player_parry_events"] = 1
                trainer.observe(first)
                trainer.observe(parry)
                for frame in range(3, 23):
                    later = snapshot("Movement 1", frame)
                    later["player_parry_events"] = 1
                    trainer.observe(later)
            credited = next(
                item for item in trainer.replay._items if item.action_vector[2] == 5
            )
            self.assertGreaterEqual(credited.reward, STEP_PENALTY + 2.0)
            self.assertEqual(trainer.metrics.player_parries, 1)
            self.assertEqual(trainer.metrics.parry_credit_reward, 2.0)
            self.assertEqual(trainer.metrics.offensive_misses, 0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_empty_attack_and_spell_each_penalize_combat_by_half(self) -> None:
        cases = (
            ("attack", "normal", 1, 154.0, 20.0, True),
            ("attack", "up", 5, 152.0, 24.0, True),
            ("attack", "down", 6, 152.0, 20.0, False),
            ("spell", "quick_cast", 3, 154.0, 20.0, True),
        )
        neutral = (0,) * len(BRANCH_SIZES)
        for action_kind, label, combat_value, boss_x, boss_y, grounded in cases:
            path = Path(f"tests/.real_dqn_{label}_miss.jsonl")
            executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
            online = BranchingDQN()
            trainer = LiveTrainer(
                online,
                BranchingDQN(),
                torch.optim.AdamW(online.parameters(), lr=1e-4),
                executor,
                torch.device("cpu"),
                random.Random(1),
            )
            action = (0, 0, combat_value)
            try:
                with (
                    patch(
                        "hk_rl_DQN.real_dqn.branch_availability",
                        return_value=(ALL_ACTIONS_AVAILABLE, ()),
                    ),
                    patch(
                        "hk_rl_DQN.real_dqn.select_action",
                        side_effect=[action] + [neutral] * 24,
                    ),
                ):
                    for frame in range(1, 23):
                        item = snapshot("Movement 1", frame)
                        item["player_grounded"] = grounded
                        if not grounded:
                            item["player"]["y"] = 24.0
                        item["boss"]["x"] = boss_x
                        item["boss"]["y"] = boss_y
                        trainer.observe(item)
                missed = next(
                    item
                    for item in trainer.replay._items
                    if item.action_vector[2] == combat_value
                )
                self.assertEqual(trainer.metrics.offensive_misses, 1)
                self.assertAlmostEqual(missed.reward, STEP_PENALTY - 0.5)
                self.assertEqual(
                    getattr(trainer.metrics, f"{action_kind}_misses"), 1
                )
            finally:
                executor.close()
                path.unlink(missing_ok=True)

    def test_fringe_or_interrupted_attack_is_not_counted_as_a_miss(self) -> None:
        cases = ("fringe", "interrupted")
        for case in cases:
            path = Path(f"tests/.real_dqn_no_miss_{case}.jsonl")
            executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
            online = BranchingDQN()
            trainer = LiveTrainer(
                online,
                BranchingDQN(),
                torch.optim.AdamW(online.parameters(), lr=1e-4),
                executor,
                torch.device("cpu"),
                random.Random(1),
            )
            try:
                with (
                    patch(
                        "hk_rl_DQN.real_dqn.branch_availability",
                        return_value=(ALL_ACTIONS_AVAILABLE, ()),
                    ),
                    patch(
                        "hk_rl_DQN.real_dqn.select_action",
                        side_effect=[(0, 0, 1)] + [(0, 0, 0)] * 24,
                    ),
                ):
                    for frame in range(1, 23):
                        health = 9 if case == "interrupted" and frame >= 2 else 10
                        item = snapshot("Movement 1", frame, health=health)
                        if case == "fringe":
                            item["boss"]["x"] = 157.2
                        trainer.observe(item)
                self.assertEqual(trainer.metrics.offensive_misses, 0)
                if case == "fringe":
                    self.assertEqual(trainer.metrics.fringe_range_attacks, 1)
                else:
                    self.assertEqual(trainer.metrics.confirmed_range_attacks, 1)
            finally:
                executor.close()
                path.unlink(missing_ok=True)

    def test_charge_release_hurt_during_recovery_is_not_a_miss(self) -> None:
        path = Path("tests/.real_dqn_charge_recovery_interrupt.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        actions = [(0, 0, 2)] * 14 + [(0, 0, 0)] * 26
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch("hk_rl_DQN.real_dqn.select_action", side_effect=actions),
            ):
                for frame in range(1, 40):
                    health = 9 if frame >= 17 else 10
                    trainer.observe(snapshot("Movement 1", frame, health=health))
            self.assertEqual(trainer.metrics.confirmed_range_attacks, 1)
            self.assertEqual(trainer.metrics.attack_misses, 0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_plugin_attack_events_assign_each_fixed_budget_once(self) -> None:
        for was_hit, expected_budget in (
            (False, EVADE_SUCCESS_REWARD),
            (True, EVADE_FAILURE_PENALTY),
        ):
            path = Path(f"tests/.real_dqn_attack_event_{was_hit}.jsonl")
            executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
            online = BranchingDQN()
            trainer = LiveTrainer(
                online,
                BranchingDQN(),
                torch.optim.AdamW(online.parameters(), lr=1e-4),
                executor,
                torch.device("cpu"),
                random.Random(1),
            )
            try:
                neutral = (0, 1, 0)
                with patch("hk_rl_DQN.real_dqn.select_action", return_value=neutral):
                    trainer.observe(with_attack_event(
                        snapshot("Movement 1", 1),
                        attack_id=41,
                        phase="startup",
                        active=0,
                    ))
                    trainer.observe(with_attack_event(
                        snapshot("Movement 1", 2),
                        attack_id=41,
                        phase="active",
                    ))
                    trainer.observe(with_attack_event(
                        snapshot("Movement 1", 3, health=9 if was_hit else 10),
                        attack_id=41,
                        phase="active",
                        hits=int(was_hit),
                        last_hit_id=41 if was_hit else 0,
                    ))
                    trainer.observe(with_attack_event(
                        snapshot("Movement 1", 4, health=9 if was_hit else 10),
                        attack_id=0,
                        phase="idle",
                        finished=1,
                        hits=int(was_hit),
                        last_finished_id=41,
                        last_hit_id=41 if was_hit else 0,
                    ))
                    dodge_count = trainer.metrics.dodges + trainer.metrics.failed_dodges
                    trainer.observe(with_attack_event(
                        snapshot("Movement 1", 5, health=9 if was_hit else 10),
                        attack_id=0,
                        phase="idle",
                        finished=1,
                        hits=int(was_hit),
                        last_finished_id=41,
                        last_hit_id=41 if was_hit else 0,
                    ))
                self.assertEqual(
                    trainer.metrics.dodges + trainer.metrics.failed_dodges,
                    dodge_count,
                )
                applied = (
                    trainer.metrics.evade_failure_backfill_penalty
                    if was_hit
                    else trainer.metrics.dodge_backfill_reward
                )
                self.assertAlmostEqual(applied, expected_budget)
            finally:
                executor.close()
                path.unlink(missing_ok=True)

    def test_player_hurt_is_credited_to_recent_combat_action(self) -> None:
        path = Path("tests/.real_dqn_temporal_hurt.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch(
                    "hk_rl_DQN.real_dqn.select_action",
                    side_effect=[(0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
                ),
            ):
                trainer.observe(snapshot("Slash Antic", 1, health=10))
                trainer.observe(snapshot("Slash 1", 2, health=10))
                trainer.observe(snapshot("Slash 2", 3, health=10))
                trainer.observe(snapshot("Slash End", 4, health=9))
            trainer._finalize_pending(force=True)
            attack = next(
                item for item in trainer.replay._items if item.action_vector == (0, 0, 1)
            )
            self.assertAlmostEqual(
                attack.reward,
                STEP_PENALTY + COMBAT_HURT_EVENT_PENALTY,
            )
            self.assertEqual(
                trainer.metrics.combat_hurt_penalty,
                COMBAT_HURT_EVENT_PENALTY,
            )
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_combat_hurt_without_active_overlap_has_no_extra_penalty(self) -> None:
        path = Path("tests/.real_dqn_hurt_without_threat_overlap.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            with (
                patch(
                    "hk_rl_DQN.real_dqn.branch_availability",
                    return_value=(ALL_ACTIONS_AVAILABLE, ()),
                ),
                patch(
                    "hk_rl_DQN.real_dqn.select_action",
                    side_effect=[(0, 0, 1), (0, 0, 0), (0, 0, 0)],
                ),
            ):
                trainer.observe(snapshot("Movement 1", 1, health=10))
                trainer.observe(snapshot("Movement 1", 2, health=10))
                trainer.observe(snapshot("Movement 1", 3, health=9))
            self.assertEqual(trainer.metrics.combat_hurt_penalty, 0.0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_replay_transition_is_immutable_after_pending_finalization(self) -> None:
        pending = PendingTransition(transition(reward=1.0), created_step=0)
        pending.add_reward(0.5)
        replay = ReplayBuffer()
        replay.append(pending.finalize())
        stored = replay._items[0]
        pending.add_reward(99.0)
        self.assertEqual(stored.reward, 1.5)
        with self.assertRaises(FrozenInstanceError):
            stored.reward = 7.0

    def test_dry_run_does_not_learn_from_unexecuted_actions(self) -> None:
        path = Path("tests/.real_dqn_dry_run_actions.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
            learning_enabled=False,
        )
        try:
            trainer.observe(snapshot("Slash Antic", 1))
            trainer.observe(snapshot("Slash 1", 2))
            self.assertEqual(len(trainer.replay), 0)
            self.assertEqual(trainer.global_step, 0)
            self.assertIsNotNone(trainer.previous_action)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_illegal_action_penalty_is_added_to_next_transition(self) -> None:
        path = Path("tests/.real_dqn_illegal_penalty.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        online = BranchingDQN()
        trainer = LiveTrainer(
            online,
            BranchingDQN(),
            torch.optim.AdamW(online.parameters(), lr=1e-4),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        try:
            neutral = (0,) * len(BRANCH_SIZES)
            with patch("hk_rl_DQN.real_dqn.select_action", return_value=neutral):
                trainer.observe(snapshot("Movement 1", 1))
                trainer.previous_illegal_penalty = ILLEGAL_ACTION_PENALTY
                trainer.previous_illegal_branches = ("jump_z",)
                trainer.observe(snapshot("Movement 1", 2))
            trainer._finalize_pending(force=True)
            self.assertEqual(
                trainer.replay.sample(1, random.Random(0))[0].reward,
                STEP_PENALTY + ILLEGAL_ACTION_PENALTY,
            )
            self.assertEqual(trainer.metrics.illegal_actions, 1)
            self.assertEqual(
                trainer.metrics.illegal_action_penalty,
                ILLEGAL_ACTION_PENALTY,
            )
        finally:
            executor.close()
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
