from pathlib import Path
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
    ArenaResetGate,
    BranchingDQN,
    LiveTrainer,
    Transition,
    checkpoint_metadata,
    load_checkpoint,
    optimize_model,
    select_action,
    validate_checkpoint,
)
from hk_rl_DQN.real_state import STATE_DIMENSIONS
from hk_rl_DQN.real_reward import ILLEGAL_ACTION_PENALTY, STEP_PENALTY


ALL_ACTIONS_AVAILABLE = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
ONLY_NEUTRAL_AVAILABLE = tuple(
    tuple(index == 0 for index in range(size)) for size in BRANCH_SIZES
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
        "boss": {"x": 165.0, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
        "fsm": [
            {"path": "Boss Scene/Hunter Queen Boss", "name": "Control", "state": state},
            {"path": "Boss Scene/Hunter Queen Boss", "name": "Stun Control", "state": "Idle"},
        ],
    }


class BranchingDQNTests(unittest.TestCase):
    def test_terminal_snapshots_are_ignored_until_arena_exit(self) -> None:
        gate = ArenaResetGate()
        self.assertTrue(gate.allow_snapshot(True))
        gate.mark_episode_finished()
        self.assertFalse(gate.allow_snapshot(True))
        self.assertFalse(gate.allow_snapshot(True))
        self.assertFalse(gate.allow_snapshot(False))
        self.assertTrue(gate.allow_snapshot(True))

    def test_network_has_one_head_per_branch(self) -> None:
        network = BranchingDQN()
        values = network(torch.zeros(4, STATE_DIMENSIONS))
        self.assertEqual(len(values), len(BRANCH_SIZES))
        self.assertEqual([tuple(value.shape) for value in values], [(4, size) for size in BRANCH_SIZES])

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
            for head in network.advantages:
                head.bias[-1] = 100.0
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

    def test_epsilon_is_sampled_once_for_the_whole_action_vector(self) -> None:
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
        self.assertEqual(rng.random_calls, 1)

    def test_double_dqn_optimization_is_finite(self) -> None:
        online = BranchingDQN()
        target = BranchingDQN()
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
        transitions = [
            Transition(
                state=(float(index % 2),) * STATE_DIMENSIONS,
                action=tuple(index % size for size in BRANCH_SIZES),
                reward=float(index),
                next_state=(0.5,) * STATE_DIMENSIONS,
                done=index == 3,
                next_action_masks=ALL_ACTIONS_AVAILABLE,
            )
            for index in range(4)
        ]
        loss = optimize_model(online, target, optimizer, transitions, torch.device("cpu"))
        self.assertGreaterEqual(loss, 0.0)

    def test_double_dqn_target_uses_next_branch_masks(self) -> None:
        online = BranchingDQN()
        target = BranchingDQN()
        with torch.no_grad():
            for network in (online, target):
                for parameter in network.parameters():
                    parameter.zero_()
            for head in online.advantages:
                head.bias[1] = 10.0
            for head in target.advantages:
                head.bias[1] = 20.0
        transition = Transition(
            state=(0.0,) * STATE_DIMENSIONS,
            action=(0,) * len(BRANCH_SIZES),
            reward=0.0,
            next_state=(0.0,) * STATE_DIMENSIONS,
            done=False,
            next_action_masks=ONLY_NEUTRAL_AVAILABLE,
        )
        with torch.no_grad():
            current = torch.stack([branch[:, 0] for branch in online(torch.zeros(1, STATE_DIMENSIONS))]).mean()
            masked_next = torch.stack([branch[:, 0] for branch in target(torch.zeros(1, STATE_DIMENSIONS))]).mean()
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
            self.assertEqual(len(trainer.pending_attack_transitions), 1)
            trainer.observe(snapshot("Movement 1", 3))
            self.assertGreaterEqual(len(trainer.replay), 2)
            self.assertEqual(trainer.global_step, 2)
            self.assertIsNotNone(trainer.previous_action)
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
                with patch("hk_rl_DQN.real_dqn.select_action", return_value=(2, 2, 0, 0, 0, 0, 0, 0)):
                    trainer.observe(snapshot("Movement 1", 1))
                with patch("hk_rl_DQN.real_dqn.select_action", return_value=(0,) * len(BRANCH_SIZES)):
                    trainer.observe(snapshot("Movement 1", 2))
            transition = trainer.replay.sample(1, random.Random(0))[0]
            self.assertEqual(transition.action[0], 2)
            self.assertEqual(transition.action[1], 2)
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
        neutral = (0,) * len(BRANCH_SIZES)
        try:
            with patch("hk_rl_DQN.real_dqn.select_action", return_value=neutral):
                trainer.observe(snapshot("Slash Antic", 1))
                trainer.observe(snapshot("Slash 1", 2))
                trainer.observe(snapshot("Slash End", 3))
                trainer.observe(snapshot("Movement 1", 4))
            rewards = [item.reward for item in trainer.replay._items]
            self.assertGreater(trainer.metrics.dodge_backfill_reward, 0.0)
            self.assertGreater(sum(rewards), len(rewards) * STEP_PENALTY)
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

        taunt = (0, 0, 0, 0, 0, 0, 0, 1)
        neutral = (0,) * len(BRANCH_SIZES)
        cases = (
            ("miss", {}, -0.52),
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
            finally:
                executor.close()
                path.unlink(missing_ok=True)

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
            trainer.observe(snapshot("Movement 1", 1))
            trainer.previous_illegal_penalty = ILLEGAL_ACTION_PENALTY
            trainer.previous_illegal_branches = ("jump_z",)
            trainer.observe(snapshot("Movement 1", 2))
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
