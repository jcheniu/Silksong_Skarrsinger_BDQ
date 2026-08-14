from pathlib import Path
import random
import unittest

import torch

from hk_rl_DQN.final_project.action_executor import (
    BRANCH_SIZES,
    KeyboardActionExecutor,
)
from hk_rl_DQN.final_project.action_recorder import ActionRecorder
from hk_rl_DQN.real_dqn import (
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

    def test_random_action_respects_branch_masks(self) -> None:
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
            self.assertEqual(len(trainer.replay), 1)
            self.assertEqual(trainer.global_step, 1)
            self.assertIsNotNone(trainer.previous_action)
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


if __name__ == "__main__":
    unittest.main()
