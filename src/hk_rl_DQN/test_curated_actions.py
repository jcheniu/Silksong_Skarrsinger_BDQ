from __future__ import annotations

import io
import random
import unittest

import torch

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
    JointDQN,
    decode_joint_action,
    joint_action_id,
    select_action,
)
from .real_state import STATE_DIMENSIONS


class CuratedActionCatalogTests(unittest.TestCase):
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
