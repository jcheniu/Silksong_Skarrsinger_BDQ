import unittest
from pathlib import Path

from hk_rl_DQN.final_project.action_catalog import (
    ACTION_NAMES,
    ACTION_VECTORS,
    get_action,
    get_action_vector,
)
from hk_rl_DQN.final_project.action_executor import (
    BRANCH_SIZES,
    KeyboardActionExecutor,
    validate_action,
)
from hk_rl_DQN.final_project.action_functions import ACTION_FUNCTIONS
from hk_rl_DQN.final_project.action_recorder import ActionRecorder, ChargeState
from hk_rl_DQN.tools.cold_start_action_test import execute_action_vector
from hk_rl_DQN.tools.run_all_actions import BDQ_ACTION_CASES


class FinalActionCatalogTests(unittest.TestCase):
    def test_menu_actions_are_absent(self) -> None:
        self.assertFalse(set(ACTION_NAMES) & {"tool", "inventory", "quick_map", "journal", "quests"})

    def test_healing_is_absent(self) -> None:
        self.assertNotIn("bind_heal", ACTION_NAMES)
        self.assertFalse(any(get_action(name).key == "A" for name in ACTION_NAMES))

    def test_hold_rules(self) -> None:
        self.assertEqual(get_action("jump_hold").min_hold_ms, 100)
        self.assertEqual(get_action("attack_charge").min_hold_ms, 1350)
        self.assertTrue(get_action("quick_cast").consumes_silk)
        self.assertFalse(get_action("harpoon_dash").consumes_silk)

    def test_wall_jump_is_learned_as_a_combination(self) -> None:
        self.assertNotIn("wall_jump", ACTION_NAMES)

    def test_every_catalog_action_has_a_callable(self) -> None:
        self.assertEqual(set(ACTION_FUNCTIONS), set(ACTION_NAMES))

    def test_every_atomic_action_maps_to_a_valid_three_head_vector(self) -> None:
        self.assertEqual(set(ACTION_VECTORS), set(ACTION_NAMES))
        for name in ACTION_NAMES:
            self.assertEqual(
                validate_action(get_action_vector(name)), ACTION_VECTORS[name]
            )

    def test_batch_cases_cover_every_non_neutral_branch_value(self) -> None:
        vectors = [validate_action(case[0]) for case in BDQ_ACTION_CASES.values()]
        for branch_index, size in enumerate(BRANCH_SIZES):
            self.assertEqual(
                {
                    vector[branch_index]
                    for vector in vectors
                    if vector[branch_index] != 0
                },
                set(range(1, size)),
            )
        self.assertTrue(
            any(sum(value != 0 for value in vector) == 3 for vector in vectors)
        )

    def test_vector_tool_uses_executor_ticks_and_records_release(self) -> None:
        path = Path("tests/.cold_start_vector.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            frames = execute_action_vector(
                executor,
                (0, 0, 2),
                14,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(len(frames), 15)
            self.assertTrue(frames[13]["charge_completed"])
            self.assertEqual(frames[-1]["action_vector"], [0, 0, 0])
            self.assertIn("attack_x", frames[-1]["started_branches"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_composed_frames_are_independent(self) -> None:
        path = Path("runs/test_composed_actions.jsonl")
        path.parent.mkdir(exist_ok=True)
        try:
            recorder = ActionRecorder(path)
            item = recorder.record_frame(["right", "attack"], duration_ms=50)
            recorder.close()
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(item["actions"], ["right", "attack"])
        self.assertEqual(item["keys"], ["RightArrow", "X"])

    def test_directional_attacks_record_both_physical_keys(self) -> None:
        path = Path("tests/.directional_attack_record.jsonl")
        try:
            recorder = ActionRecorder(path)
            up = recorder.record("up_attack", duration_ms=50)
            down = recorder.record("down_attack", duration_ms=50)
            recorder.close()
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(up["keys"], ["UpArrow", "X"])
        self.assertEqual(down["keys"], ["DownArrow", "X"])

    def test_charge_accumulates_and_resets(self) -> None:
        state = ChargeState()
        for _ in range(27):
            result = state.step(True, 50)
        self.assertTrue(result["charge_completed"])
        self.assertFalse(result["charge_at_max"])
        for _ in range(33):
            result = state.step(True, 50)
        self.assertEqual(result["charge_elapsed_ms"], 3000)
        self.assertTrue(result["charge_at_max"])
        result = state.step(False, 50)
        self.assertEqual(result["charge_elapsed_ms"], 0)
        self.assertFalse(result["charge_completed"])
