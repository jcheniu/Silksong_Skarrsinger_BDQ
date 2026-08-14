import unittest

from hk_rl_DQN.final_project.action_catalog import ACTION_NAMES, get_action
from hk_rl_DQN.final_project.action_recorder import ActionRecorder, ChargeState
from pathlib import Path


class FinalActionCatalogTests(unittest.TestCase):
    def test_menu_actions_are_absent(self) -> None:
        self.assertFalse(set(ACTION_NAMES) & {"tool", "inventory", "quick_map", "journal", "quests"})

    def test_healing_is_absent(self) -> None:
        self.assertNotIn("bind_heal", ACTION_NAMES)
        self.assertFalse(any(get_action(name).key == "A" for name in ACTION_NAMES))

    def test_hold_rules(self) -> None:
        self.assertEqual(get_action("quick_run").min_hold_ms, 300)
        self.assertEqual(get_action("attack_charge").min_hold_ms, 1350)
        self.assertTrue(get_action("quick_cast").consumes_silk)
        self.assertFalse(get_action("harpoon_dash").consumes_silk)

    def test_wall_jump_is_learned_as_a_combination(self) -> None:
        self.assertNotIn("wall_jump", ACTION_NAMES)

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

    def test_charge_accumulates_and_resets(self) -> None:
        state = ChargeState()
        for _ in range(27):
            result = state.step(True, 50)
        self.assertTrue(result["charge_completed"])
        result = state.step(False, 50)
        self.assertEqual(result["charge_elapsed_ms"], 0)
        self.assertFalse(result["charge_completed"])
