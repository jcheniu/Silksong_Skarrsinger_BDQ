from pathlib import Path
import unittest
from unittest.mock import call, patch

from hk_rl_DQN.final_project import action_executor
from hk_rl_DQN.final_project.action_executor import (
    BRANCH_SIZES,
    KeyboardActionExecutor,
    action_keys,
    branch_availability,
    decode_actions,
    validate_action,
)
from hk_rl_DQN.final_project.action_recorder import ActionRecorder
from hk_rl_DQN.real_state import decode_player_resources
from hk_rl_DQN.tools.cold_start_action_test import KEYS as COLD_START_KEYS


class ActionExecutorTests(unittest.TestCase):
    def test_focus_waits_for_cold_started_game_window(self) -> None:
        with (
            patch.object(
                action_executor,
                "find_game_window",
                side_effect=[RuntimeError("not ready"), 123],
            ),
            patch.object(action_executor.time, "sleep") as sleep,
            patch.object(action_executor.ctypes, "windll") as windll,
        ):
            self.assertEqual(action_executor.focus_game_window(timeout_s=1.0), 123)
        sleep.assert_called_once_with(0.25)
        windll.user32.SetForegroundWindow.assert_called_once_with(123)

    def test_heal_key_is_not_supported(self) -> None:
        self.assertNotIn("A", COLD_START_KEYS)

    def test_dreamnail_is_disabled(self) -> None:
        masks, reasons = branch_availability({})
        self.assertEqual(masks[6], (True, False))
        self.assertIn("dream_d disabled by policy", reasons)
        self.assertNotIn("D", action_keys((0, 0, 0, 0, 0, 0, 1, 0)))

    def test_combined_action_decodes_to_simultaneous_keys(self) -> None:
        action = (2, 1, 0, 1, 0, 0, 0, 0)
        self.assertEqual(action_keys(action), ("RightArrow", "Z", "X"))
        self.assertEqual(decode_actions(action), ("right", "jump", "attack"))

        mobility = (1, 3, 2, 2, 0, 0, 0, 0)
        self.assertEqual(action_keys(mobility), ("LeftArrow", "Z", "C", "X"))
        self.assertEqual(
            decode_actions(mobility),
            ("left", "double_jump", "quick_run", "attack_charge"),
        )

    def test_double_jump_retriggers_z_keydown(self) -> None:
        path = Path("tests/.double_jump_retrigger.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        executor.send_input = True
        try:
            with patch.object(action_executor, "_send_key") as send_key:
                executor.apply((0, 1, 0, 0, 0, 0, 0, 0), branch_masks=masks)
                send_key.reset_mock()
                executor.apply((0, 3, 0, 0, 0, 0, 0, 0), branch_masks=masks)
                self.assertIn(call("Z", True), send_key.call_args_list)
                self.assertIn(call("Z", False), send_key.call_args_list)
        finally:
            executor.send_input = False
            executor.close()
            path.unlink(missing_ok=True)

    def test_short_jump_releases_on_next_tick(self) -> None:
        path = Path("tests/.short_jump.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100, send_input=False)
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        try:
            first = executor.apply((0, 1, 0, 0, 0, 0, 0, 0), branch_masks=masks)
            second = executor.apply((0, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks)
            self.assertIn("jump", first["actions"])
            self.assertEqual(second["actions"], ["wait"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_hold_jump_continues_after_intent_tick(self) -> None:
        path = Path("tests/.hold_jump.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100, send_input=False)
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        try:
            frames = [
                executor.apply((0, 2, 0, 0, 0, 0, 0, 0), branch_masks=masks),
                executor.apply((0, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks),
                executor.apply((0, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks),
                executor.apply((0, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks),
                executor.apply((0, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks),
            ]
            self.assertTrue(all("Z" in item["keys"] for item in frames[:4]))
            self.assertTrue(all(item["action_vector"][1] == 2 for item in frames[:4]))
            self.assertNotIn("Z", frames[4]["keys"])
            self.assertEqual(frames[4]["action_vector"][1], 0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_tap_attack_is_exposed_and_charge_requires_value_two(self) -> None:
        path = Path("tests/.attack_fragments.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100, send_input=False)
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        try:
            tap = executor.apply((0, 0, 0, 1, 0, 0, 0, 0), branch_masks=masks)
            charge = executor.apply((0, 0, 0, 2, 0, 0, 0, 0), branch_masks=masks)
            release = executor.apply((0, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks)
            self.assertEqual(tap["actions"], ["attack"])
            self.assertEqual(tap["charge_elapsed_ms"], 0)
            self.assertEqual(charge["actions"], ["attack_charge"])
            self.assertEqual(charge["charge_elapsed_ms"], 100)
            self.assertEqual(release["charge_elapsed_ms"], 0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_all_branches_remain_composable(self) -> None:
        action = (2, 2, 2, 2, 1, 1, 0, 1)
        self.assertEqual(
            action_keys(action),
            ("RightArrow", "Z", "C", "X", "S", "LeftShift", "V"),
        )

    def test_taunt_press_edge_is_recorded(self) -> None:
        path = Path("tests/.taunt_press_edge.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        try:
            first = executor.apply((0, 0, 0, 0, 0, 0, 0, 1))
            held = executor.apply((0, 0, 0, 0, 0, 0, 0, 1))
            self.assertIn("V", first["newly_pressed_keys"])
            self.assertNotIn("V", held["newly_pressed_keys"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_invalid_branch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_action((3,) + (0,) * (len(BRANCH_SIZES) - 1))

    def test_dry_run_records_vector_without_sending_input(self) -> None:
        path = Path("tests/.action_executor.jsonl")
        recorder = ActionRecorder(path)
        try:
            executor = KeyboardActionExecutor(recorder, send_input=False)
            item = executor.apply((1, 0, 1, 0, 0, 0, 0, 0))
            executor.close()
            self.assertEqual(item["actions"], ["left", "dash"])
            self.assertEqual(item["action_vector"], [1, 0, 1, 0, 0, 0, 0, 0])
        finally:
            path.unlink(missing_ok=True)

    def test_attack_branch_accumulates_charge_metadata(self) -> None:
        path = Path("tests/.action_executor_charge.jsonl")
        recorder = ActionRecorder(path)
        try:
            executor = KeyboardActionExecutor(recorder, tick_ms=100, send_input=False)
            first = executor.apply((0, 0, 0, 2, 0, 0, 0, 0))
            second = executor.apply((0, 0, 0, 2, 0, 0, 0, 0))
            released = executor.apply((0, 0, 0, 0, 0, 0, 0, 0))
            executor.close()
            self.assertEqual(first["charge_elapsed_ms"], 100)
            self.assertEqual(second["charge_elapsed_ms"], 200)
            self.assertEqual(released["charge_elapsed_ms"], 0)
        finally:
            path.unlink(missing_ok=True)

    def test_control_state_tracks_x_c_s_holds_and_interruption(self) -> None:
        path = Path("tests/.action_executor_state.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100, send_input=False)
        try:
            executor.apply((0, 2, 1, 1, 1, 0, 0, 0))
            state = executor.control_state(
                {
                    "player_grounded": True,
                    "player_control": {
                        "jump_available": True,
                        "double_jump_available": False,
                        "dash_available": True,
                        "sprint_available": True,
                        "attack_available": True,
                    },
                }
            )
            self.assertTrue(state.jump_held)
            self.assertAlmostEqual(state.jump_hold_progress, 100 / 350)
            self.assertTrue(state.jump_available)
            self.assertFalse(state.double_jump_available)
            self.assertTrue(state.attack_held)
            self.assertTrue(state.dash_held)
            self.assertTrue(state.skill_held)
            self.assertAlmostEqual(state.attack_hold_progress, 100 / 1350)
            self.assertAlmostEqual(state.dash_hold_progress, 100 / 300)
            self.assertAlmostEqual(state.skill_hold_progress, 100 / 900)

            executor.release_all()
            interrupted = executor.control_state({})
            self.assertFalse(interrupted.jump_held)
            self.assertEqual(interrupted.jump_hold_progress, 0.0)
            self.assertFalse(interrupted.attack_held)
            self.assertEqual(interrupted.attack_hold_progress, 0.0)
            self.assertEqual(interrupted.dash_hold_progress, 0.0)
            self.assertEqual(interrupted.skill_hold_progress, 0.0)
            self.assertTrue(interrupted.interrupted)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_action_log_contains_masks_and_reasons(self) -> None:
        path = Path("tests/.action_executor_masks.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        try:
            masks, reasons = branch_availability({})
            item = executor.apply(
                (0, 0, 0, 0, 1, 1, 0, 0),
                branch_masks=masks,
                masked_reasons=reasons,
                player_resources=decode_player_resources({}),
            )
            self.assertEqual(item["action_vector"], [0] * len(BRANCH_SIZES))
            self.assertEqual(item["branch_masks"], [list(branch) for branch in masks])
            self.assertEqual(item["masked_reasons"], list(reasons))
            self.assertIn("player_resources", item)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_explicit_availability_is_required(self) -> None:
        masks, _ = branch_availability(
            {
                "player_resources": {
                    "silk": 5,
                    "silk_max": 9,
                    "silk_parts": 0,
                    "skill_cost": 4,
                    "silk_abilities_disabled": False,
                    "skill_available": False,
                    "spell_available": False,
                }
            }
        )
        self.assertFalse(masks[4][1])
        self.assertFalse(masks[5][1])

    def test_harpoon_does_not_require_silk(self) -> None:
        masks, _ = branch_availability(
            {
                "player_resources": {
                    "silk": 0,
                    "silk_max": 9,
                    "silk_parts": 0,
                    "skill_cost": 4,
                    "silk_abilities_disabled": False,
                    "skill_available": True,
                    "spell_available": False,
                }
            }
        )
        self.assertTrue(masks[4][1])
        self.assertFalse(masks[5][1])

    def test_quick_cast_requires_enough_silk(self) -> None:
        snapshot = {
            "player_resources": {
                "silk": 3,
                "silk_max": 9,
                "silk_parts": 0,
                "skill_cost": 4,
                "silk_abilities_disabled": False,
                "skill_available": True,
                "spell_available": True,
            }
        }
        masks, reasons = branch_availability(snapshot)
        self.assertFalse(masks[5][1])
        self.assertTrue(any("silk 3 < cost 4" in reason for reason in reasons))

        snapshot["player_resources"]["silk"] = 4
        masks, _ = branch_availability(snapshot)
        self.assertTrue(masks[5][1])

    def test_quick_cast_rejects_disabled_or_incomplete_resources(self) -> None:
        disabled = {
            "player_resources": {
                "silk": 9,
                "silk_max": 9,
                "silk_parts": 0,
                "skill_cost": 4,
                "silk_abilities_disabled": True,
                "skill_available": True,
                "spell_available": True,
            }
        }
        disabled_masks, _ = branch_availability(disabled)
        incomplete_masks, _ = branch_availability(
            {"player_resources": {"silk": 9, "skill_cost": 4}}
        )
        self.assertFalse(disabled_masks[5][1])
        self.assertFalse(incomplete_masks[5][1])

    def test_player_control_masks_core_actions(self) -> None:
        masks, reasons = branch_availability(
            {
                "player_control": {
                    "jump_available": False,
                    "dash_available": False,
                    "attack_available": False,
                }
            }
        )
        self.assertFalse(masks[1][1])
        self.assertFalse(masks[2][1])
        self.assertFalse(masks[3][1])
        self.assertTrue(any("jump_available" in reason for reason in reasons))

    def test_double_jump_and_sprint_have_separate_legality(self) -> None:
        airborne_masks, _ = branch_availability(
            {
                "player_grounded": False,
                "player_control": {
                    "jump_available": False,
                    "double_jump_available": True,
                    "dash_available": False,
                    "sprint_available": True,
                    "attack_available": True,
                },
            }
        )
        self.assertEqual(airborne_masks[1], (True, False, False, True))
        self.assertEqual(airborne_masks[2], (True, False, False))

        grounded_masks, _ = branch_availability(
            {
                "player_grounded": True,
                "player_control": {
                    "jump_available": True,
                    "double_jump_available": False,
                    "dash_available": True,
                    "sprint_available": True,
                    "attack_available": True,
                },
            }
        )
        self.assertEqual(grounded_masks[1], (True, True, True, False))
        self.assertEqual(grounded_masks[2], (True, True, True))

        active_sprint_masks, _ = branch_availability(
            {
                "player_grounded": True,
                "player_control": {
                    "sprint_available": False,
                    "sprinting": True,
                },
            }
        )
        self.assertTrue(active_sprint_masks[2][2])

    def test_direction_and_run_have_minimum_hold(self) -> None:
        path = Path("tests/.smooth_movement.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        masks = tuple(tuple(True for _ in range(size)) for size in BRANCH_SIZES)
        try:
            first = executor.apply((2, 0, 2, 0, 0, 0, 0, 0), branch_masks=masks)
            second = executor.apply((1, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks)
            third = executor.apply((1, 0, 0, 0, 0, 0, 0, 0), branch_masks=masks)
            self.assertEqual(first["action_vector"][0:3], [2, 0, 2])
            self.assertEqual(second["action_vector"][0:3], [2, 0, 2])
            self.assertEqual(third["action_vector"][0:3], [2, 0, 2])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_illegal_action_is_neutralized_and_recorded(self) -> None:
        path = Path("tests/.illegal_action.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path))
        try:
            masks, reasons = branch_availability({})
            item = executor.apply(
                (0, 2, 0, 0, 0, 0, 0, 0),
                branch_masks=masks,
                masked_reasons=reasons,
            )
            self.assertEqual(item["attempted_action_vector"][1], 2)
            self.assertEqual(item["action_vector"][1], 0)
            self.assertIn("jump_z", item["illegal_branches"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)
