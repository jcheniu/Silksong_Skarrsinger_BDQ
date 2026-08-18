from __future__ import annotations

import io
import random
from types import SimpleNamespace
import unittest

import torch

from . import real_actions, real_replay, real_reward
from .final_project.action_executor import (
    BRANCH_SIZES,
    KeyboardActionExecutor,
    action_keys,
    decode_actions,
)
from .final_project.action_recorder import ActionRecorder
from .real_dqn import (
    JOINT_ACTIONS,
    JOINT_ACTION_COUNT,
    EVADE_SUCCESS_REWARD,
    HARPOON_SUCCESS_BONUS_FRACTION,
    BOSS_PROXIMITY_REWARD,
    ActionOutcomeTrial,
    JointDQN,
    LiveTrainer,
    PendingTransition,
    ZERO_SPACE_ENTRY_PENALTY,
    ZERO_SPACE_HOLD_PENALTY,
    ZERO_SPACE_HURT_PENALTY,
    Transition,
    decode_joint_action,
    joint_action_id,
    joint_action_mask,
    mirror_transition,
    select_action,
)
from .real_state import COLLISION_RISK_INDEX, STATE_DIMENSIONS, encode_snapshot


class CuratedActionCatalogTests(unittest.TestCase):
    def test_refactored_modules_own_public_components(self) -> None:
        self.assertIs(JOINT_ACTIONS, real_actions.JOINT_ACTIONS)
        self.assertIs(Transition, real_replay.Transition)
        self.assertIs(PendingTransition, real_reward.PendingTransition)
        self.assertIs(ActionOutcomeTrial, real_reward.ActionOutcomeTrial)

    def test_catalog_has_expected_shape_and_order(self) -> None:
        self.assertEqual(BRANCH_SIZES, (3, 6, 6))
        self.assertEqual(JOINT_ACTION_COUNT, 53)
        expected_ids = {
            0: (0, 0, 0),
            22: (0, 5, 0),
            23: (1, 0, 0),
            24: (1, 0, 2),
            30: (1, 4, 0),
            31: (2, 0, 0),
            52: (2, 4, 1),
        }
        for action_id, action in expected_ids.items():
            self.assertEqual(decode_joint_action(action_id), action)
            self.assertEqual(joint_action_id(action), action_id)

    def test_catalog_contains_only_curated_combinations(self) -> None:
        self.assertEqual(len(set(JOINT_ACTIONS)), JOINT_ACTION_COUNT)
        for jump, movement, combat in JOINT_ACTIONS:
            if movement == 5:
                self.assertEqual((jump, movement, combat), (0, 5, 0))
            if movement in (3, 4):
                self.assertIn(combat, (0, 1))
            if jump == 1:
                self.assertIn(combat, (0, 2))

    def test_removed_combinations_are_rejected(self) -> None:
        for action in (
            (0, 3, 2),
            (0, 4, 3),
            (1, 0, 1),
            (1, 1, 4),
            (2, 5, 0),
        ):
            with self.assertRaises(ValueError):
                joint_action_id(action)

    def test_new_semantic_indices_map_to_expected_keys(self) -> None:
        self.assertEqual(action_keys((0, 3, 1)), ("LeftArrow", "C", "X"))
        self.assertEqual(action_keys((0, 4, 1)), ("RightArrow", "C", "X"))
        self.assertEqual(action_keys((0, 5, 0)), ("S",))
        self.assertEqual(action_keys((0, 0, 4)), ("UpArrow", "X"))
        self.assertEqual(action_keys((0, 0, 5)), ("DownArrow", "X"))
        self.assertEqual(decode_actions((0, 3, 1)), ("left_dash", "attack"))
        self.assertEqual(decode_actions((0, 4, 1)), ("right_dash", "attack"))

    def test_directional_dash_frame_can_be_recorded(self) -> None:
        recorder = ActionRecorder.__new__(ActionRecorder)
        recorder.stream = io.StringIO()
        recorder.sequence = 0
        executor = KeyboardActionExecutor(recorder=recorder)
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        item = executor.apply(
            (0, 3, 1),
            branch_masks=masks,
        )
        self.assertEqual(item["actions"], ["left_dash", "attack"])
        self.assertEqual(item["keys"], ["LeftArrow", "C", "X"])

    def test_default_executor_uses_fifty_millisecond_ticks(self) -> None:
        recorder = ActionRecorder.__new__(ActionRecorder)
        recorder.stream = io.StringIO()
        recorder.sequence = 0
        executor = KeyboardActionExecutor(recorder=recorder)
        self.assertEqual(executor.tick_ms, 50)

    def test_spin_and_cyclone_have_distinct_semantic_codes(self) -> None:
        def snapshot(control_state: str) -> dict[str, object]:
            return {
                "player_grounded": True,
                "player": {"x": 150.0, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
                "boss": {"x": 156.0, "y": 20.0, "velocity_x": -2.0, "velocity_y": 0.0},
                "fsm": [{
                    "path": "Boss Scene/Hunter Queen Boss",
                    "name": "Control",
                    "state": control_state,
                }],
            }

        cyclone = encode_snapshot(snapshot("Cyclone 2"))
        spin = encode_snapshot(snapshot("Spin Attack"))
        self.assertEqual(cyclone.attack_type, "cyclone")
        self.assertEqual(spin.attack_type, "spin_attack")
        self.assertNotEqual(cyclone.observation[12], spin.observation[12])
        self.assertEqual(
            encode_snapshot(snapshot("Launch Antic")).attack_type,
            "spin_attack",
        )
        self.assertEqual(
            encode_snapshot(snapshot("Jump Launch")).attack_type,
            "jump_attack",
        )

    def test_collision_risk_increases_when_close_and_closing(self) -> None:
        def snapshot(boss_x: float, boss_velocity_x: float) -> dict[str, object]:
            return {
                "player_grounded": True,
                "player": {"x": 150.0, "y": 20.0, "velocity_x": 2.0, "velocity_y": 0.0},
                "boss": {"x": boss_x, "y": 20.0, "velocity_x": boss_velocity_x, "velocity_y": 0.0},
                "fsm": [],
            }

        far = encode_snapshot(snapshot(165.0, 0.0)).observation[COLLISION_RISK_INDEX]
        close = encode_snapshot(snapshot(153.0, -2.0)).observation[COLLISION_RISK_INDEX]
        self.assertEqual(far, 0.0)
        self.assertGreater(close, 0.5)

    def test_mirror_augmentation_swaps_directional_state_and_action(self) -> None:
        state = tuple((index + 1) / 100.0 for index in range(STATE_DIMENSIONS))
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        original = Transition(
            state=state,
            action=joint_action_id((2, 1, 4)),
            reward=1.25,
            next_state=state,
            done=False,
            next_action_mask=joint_action_mask(masks),
        )
        mirrored = mirror_transition(original)
        self.assertEqual(mirrored.action_vector, (2, 2, 4))
        for index in (0, 2, 4, 6, 9, 19):
            self.assertEqual(mirrored.state[index], -original.state[index])
        self.assertEqual(sum(mirrored.next_action_mask), sum(original.next_action_mask))

    def test_successful_harpoon_gets_hit_and_evade_bonuses(self) -> None:
        recorder = ActionRecorder.__new__(ActionRecorder)
        recorder.stream = io.StringIO()
        recorder.sequence = 0
        executor = KeyboardActionExecutor(recorder=recorder)
        online = JointDQN()
        trainer = LiveTrainer(
            online,
            JointDQN(),
            torch.optim.AdamW(online.parameters()),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)

        def harpoon_pending() -> PendingTransition:
            return PendingTransition(
                Transition(
                    state=(0.0,) * STATE_DIMENSIONS,
                    action=joint_action_id((0, 5, 0)),
                    reward=0.0,
                    next_state=(0.0,) * STATE_DIMENSIONS,
                    done=False,
                    next_action_mask=joint_action_mask(masks),
                ),
                created_step=0,
            )

        hit_pending = harpoon_pending()
        trainer.action_outcome_trials = [
            ActionOutcomeTrial(hit_pending, "harpoon", penalize_miss=False)
        ]
        trainer._apply_action_outcomes(
            SimpleNamespace(
                damage_reward=2.0,
                parry_reward=0.0,
                player_damage_taken=0,
                terminated=False,
            ),
            SimpleNamespace(phase_event="none", reaction="normal"),
        )
        self.assertAlmostEqual(
            hit_pending.delayed_reward,
            2.0 * (1.0 + HARPOON_SUCCESS_BONUS_FRACTION),
        )

        evade_pending = harpoon_pending()
        window = trainer._window(7, "spin_attack")
        window.active_seen = True
        window.transitions.append(evade_pending)
        trainer._resolve_attack_window(7)
        expected_evade = EVADE_SUCCESS_REWARD * (
            1.0 + HARPOON_SUCCESS_BONUS_FRACTION
        )
        self.assertAlmostEqual(evade_pending.delayed_reward, expected_evade)
        self.assertAlmostEqual(
            trainer.metrics.harpoon_evade_bonus_reward,
            EVADE_SUCCESS_REWARD * HARPOON_SUCCESS_BONUS_FRACTION,
        )

    def test_zero_space_is_a_boss_centered_lower_half_ellipse(self) -> None:
        def frame(player_x: float, player_y: float):
            return encode_snapshot({
                "player_grounded": False,
                "player": {
                    "x": player_x,
                    "y": player_y,
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                },
                "boss": {
                    "x": 150.0,
                    "y": 20.0,
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                },
                "fsm": [],
            })

        self.assertTrue(LiveTrainer._inside_zero_space(frame(150.0, 20.0)))
        self.assertTrue(LiveTrainer._inside_zero_space(frame(148.8, 20.0)))
        self.assertTrue(LiveTrainer._inside_zero_space(frame(150.0, 18.4)))
        self.assertFalse(LiveTrainer._inside_zero_space(frame(150.0, 20.01)))
        self.assertFalse(LiveTrainer._inside_zero_space(frame(148.8, 18.4)))

    def test_zero_space_entry_hold_and_hurt_credit_are_auditable(self) -> None:
        recorder = ActionRecorder.__new__(ActionRecorder)
        recorder.stream = io.StringIO()
        recorder.sequence = 0
        executor = KeyboardActionExecutor(recorder=recorder)
        online = JointDQN()
        trainer = LiveTrainer(
            online,
            JointDQN(),
            torch.optim.AdamW(online.parameters()),
            executor,
            torch.device("cpu"),
            random.Random(1),
        )
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)

        def pending() -> PendingTransition:
            return PendingTransition(
                Transition(
                    state=(0.0,) * STATE_DIMENSIONS,
                    action=joint_action_id((0, 1, 1)),
                    reward=0.0,
                    next_state=(0.0,) * STATE_DIMENSIONS,
                    done=False,
                    next_action_mask=joint_action_mask(masks),
                ),
                created_step=0,
            )

        state = encode_snapshot({
            "player_grounded": True,
            "player": {"x": 150.0, "y": 19.5, "velocity_x": 0.0, "velocity_y": 0.0},
            "boss": {"x": 150.0, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
            "fsm": [],
        })
        trainer.proximity_reward_balance = BOSS_PROXIMITY_REWARD
        first = pending()
        trainer._apply_zero_space_shaping(state, first)
        self.assertAlmostEqual(
            first.delayed_reward,
            ZERO_SPACE_ENTRY_PENALTY - BOSS_PROXIMITY_REWARD,
        )
        second = pending()
        trainer._apply_zero_space_shaping(state, second)
        self.assertAlmostEqual(second.delayed_reward, ZERO_SPACE_HOLD_PENALTY)

        trainer._credit_offensive_reward(second, 2.0)
        trainer.pending_credit_transitions.append(second)
        trainer._apply_zero_space_hurt_credit(
            SimpleNamespace(player_damage_taken=1)
        )
        self.assertAlmostEqual(
            second.delayed_reward,
            ZERO_SPACE_HOLD_PENALTY + ZERO_SPACE_HURT_PENALTY,
        )
        trainer._credit_offensive_reward(second, 1.0)
        self.assertAlmostEqual(
            second.delayed_reward,
            ZERO_SPACE_HOLD_PENALTY + ZERO_SPACE_HURT_PENALTY,
        )
        self.assertAlmostEqual(trainer.metrics.zero_space_offensive_clawback, -3.0)

    def test_network_outputs_one_value_per_curated_action(self) -> None:
        output = JointDQN()(torch.zeros((2, STATE_DIMENSIONS)))
        self.assertEqual(tuple(output.shape), (2, 53))

    def test_exploration_never_returns_a_removed_combination(self) -> None:
        network = JointDQN()
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        rng = random.Random(7)
        observation = (0.0,) * STATE_DIMENSIONS
        for _ in range(100):
            action = select_action(
                network,
                observation,
                epsilon=1.0,
                rng=rng,
                device=torch.device("cpu"),
                branch_masks=masks,
            )
            self.assertIn(action, JOINT_ACTIONS)


if __name__ == "__main__":
    unittest.main()
