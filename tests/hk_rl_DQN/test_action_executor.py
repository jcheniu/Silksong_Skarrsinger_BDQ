import unittest
from pathlib import Path
from unittest.mock import patch

from hk_rl_DQN.final_project import action_executor
from hk_rl_DQN.final_project.action_executor import (
    BRANCH_NAMES,
    BRANCH_SIZES,
    KeyboardActionExecutor,
    action_keys,
    branch_availability,
    decode_actions,
    validate_action,
)
from hk_rl_DQN.final_project.action_recorder import ActionRecorder
from hk_rl_DQN.real_state import decode_player_resources


ALL_ACTIONS_AVAILABLE = tuple(
    tuple(True for _ in range(size)) for size in BRANCH_SIZES
)


class ActionExecutorTests(unittest.TestCase):
    def test_standard_three_head_schema(self) -> None:
        self.assertEqual(BRANCH_NAMES, ("jump_z", "movement", "combat"))
        self.assertEqual(BRANCH_SIZES, (3, 7, 7))

    def test_heads_compose_in_one_control_frame(self) -> None:
        action = (2, 5, 1)
        self.assertEqual(action_keys(action), ("Z", "RightArrow", "C", "X"))
        self.assertEqual(
            decode_actions(action),
            ("jump_hold", "right", "dash", "attack"),
        )
        self.assertEqual(action_keys((0, 6, 1)), ("S", "X"))
        self.assertEqual(
            decode_actions((0, 6, 1)), ("harpoon_dash", "attack")
        )

    def test_movement_head_contains_direction_and_directed_dash(self) -> None:
        self.assertEqual(action_keys((0, 1, 0)), ("LeftArrow",))
        self.assertEqual(action_keys((0, 2, 0)), ("RightArrow",))
        self.assertEqual(action_keys((0, 3, 0)), ("C",))
        self.assertEqual(action_keys((0, 4, 0)), ("LeftArrow", "C"))
        self.assertEqual(action_keys((0, 5, 0)), ("RightArrow", "C"))
        self.assertEqual(action_keys((0, 6, 0)), ("S",))

    def test_combat_head_is_mutually_exclusive(self) -> None:
        expected = {
            0: (),
            1: ("X",),
            2: ("X",),
            3: ("LeftShift",),
            4: ("V",),
            5: ("UpArrow", "X"),
            6: ("DownArrow", "X"),
        }
        for value, keys in expected.items():
            self.assertEqual(action_keys((0, 0, value)), keys)

    def test_directional_attacks_decode_and_start_as_x_attacks(self) -> None:
        path = Path("tests/.directional_attacks.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path))
        try:
            up = executor.apply((0, 0, 5), branch_masks=ALL_ACTIONS_AVAILABLE)
            down = executor.apply((0, 0, 6), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertEqual(decode_actions((0, 0, 5)), ("up_attack",))
            self.assertEqual(decode_actions((0, 0, 6)), ("down_attack",))
            self.assertIn("attack_x", up["started_branches"])
            self.assertIn("attack_x", down["started_branches"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_generic_jump_press_retriggers_z(self) -> None:
        path = Path("tests/.jump_retrigger.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        executor.send_input = True
        try:
            with patch.object(action_executor, "_send_key") as send_key:
                executor.apply((1, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
                executor.apply((1, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertIn(("Z", True), [call.args for call in send_key.call_args_list])
            self.assertGreaterEqual(
                sum(call.args == ("Z", False) for call in send_key.call_args_list), 2
            )
        finally:
            executor.send_input = False
            executor.close()
            path.unlink(missing_ok=True)

    def test_jump_hold_is_controlled_each_tick_without_phase_logic(self) -> None:
        path = Path("tests/.generic_jump_hold.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            held = executor.apply((2, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            released = executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertIn("Z", held["keys"])
            self.assertNotIn("Z", released["keys"])
            self.assertEqual(held["action_vector"], [2, 0, 0])
            self.assertEqual(released["action_vector"], [0, 0, 0])
        finally:
            executor.send_input = False
            executor.close()
            path.unlink(missing_ok=True)

    def test_jump_hold_can_continue_beyond_known_game_effect_window(self) -> None:
        path = Path("tests/.unbounded_jump_hold.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            frames = [
                executor.apply((2, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
                for _ in range(12)
            ]
            self.assertTrue(all("Z" in item["keys"] for item in frames))
            state = executor.control_state({})
            self.assertEqual(state.jump_state, 1.0)
        finally:
            executor.send_input = False
            executor.close()
            path.unlink(missing_ok=True)

    def test_jump_branch_is_never_semantically_masked(self) -> None:
        masks, reasons = branch_availability(
            {
                "player_grounded": False,
                "player_control": {
                    "jump_available": False,
                    "double_jump_available": False,
                    "dash_available": False,
                    "attack_available": False,
                },
            }
        )
        self.assertEqual(masks[0], (True, True, True))
        self.assertFalse(any("jump" in reason for reason in reasons))

    def test_dash_values_share_one_movement_mask(self) -> None:
        unavailable, _ = branch_availability(
            {"player_control": {"dash_available": False}}
        )
        available, _ = branch_availability(
            {"player_control": {"dash_available": True}}
        )
        self.assertEqual(
            unavailable[1], (True, True, True, False, False, False, False)
        )
        self.assertEqual(
            available[1], (True, True, True, True, True, True, False)
        )

    def test_charge_can_continue_but_harpoon_requires_current_availability(self) -> None:
        snapshot = {
            "player_control": {"dash_available": False, "attack_available": False},
            "player_resources": {
                "silk": 0,
                "silk_max": 9,
                "silk_parts": 0,
                "skill_cost": 4,
                "silk_abilities_disabled": False,
                "skill_available": False,
                "spell_available": False,
            },
        }
        charge_masks, _ = branch_availability(snapshot, (0, 0, 2))
        skill_masks, _ = branch_availability(snapshot, (0, 6, 0))
        self.assertFalse(charge_masks[2][1])
        self.assertFalse(charge_masks[2][5])
        self.assertFalse(charge_masks[2][6])
        self.assertTrue(charge_masks[2][2])
        self.assertFalse(skill_masks[1][6])

    def test_quick_cast_mask_uses_resource_telemetry(self) -> None:
        allowed, _ = branch_availability(
            {
                "player_resources": {
                    "silk": 5,
                    "silk_max": 9,
                    "silk_parts": 0,
                    "skill_cost": 4,
                    "silk_abilities_disabled": False,
                    "skill_available": True,
                    "spell_available": True,
                }
            }
        )
        blocked, _ = branch_availability({})
        self.assertTrue(allowed[2][3])
        self.assertFalse(blocked[2][3])

    def test_dash_and_shift_are_real_retriggerable_pulses(self) -> None:
        path = Path("tests/.three_head_pulses.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), send_input=False)
        executor.send_input = True
        try:
            with patch.object(action_executor, "_send_key") as send_key:
                executor.apply((0, 3, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
                executor.apply((0, 3, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
                executor.apply((0, 0, 3), branch_masks=ALL_ACTIONS_AVAILABLE)
                executor.apply((0, 0, 3), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertGreaterEqual(
                sum(call.args == ("C", False) for call in send_key.call_args_list), 2
            )
            self.assertGreaterEqual(
                sum(call.args == ("LeftShift", False) for call in send_key.call_args_list), 2
            )
        finally:
            executor.send_input = False
            executor.close()
            path.unlink(missing_ok=True)

    def test_charge_credit_event_occurs_only_on_completed_release(self) -> None:
        path = Path("tests/.three_head_charge.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            held = [
                executor.apply((0, 0, 2), branch_masks=ALL_ACTIONS_AVAILABLE)
                for _ in range(14)
            ]
            released = executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertFalse(any("attack_x" in x["started_branches"] for x in held))
            self.assertTrue(held[-1]["charge_completed"])
            self.assertIn("attack_x", released["started_branches"])
            self.assertTrue(released["charge_released"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_harpoon_is_one_tick_pulse_followed_by_action_lock(self) -> None:
        path = Path("tests/.three_head_skill.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            launch = executor.apply((2, 6, 1), branch_masks=ALL_ACTIONS_AVAILABLE)
            locked = [
                executor.apply((2, 2, 1), branch_masks=ALL_ACTIONS_AVAILABLE)
                for _ in range(8)
            ]
            self.assertEqual(launch["attempted_action_vector"], [2, 6, 1])
            self.assertEqual(launch["action_vector"], [0, 6, 0])
            self.assertEqual(launch["keys"], ["S"])
            self.assertIn("skill_s", launch["started_branches"])
            self.assertTrue(launch["adjusted_reasons"])
            self.assertTrue(all(item["action_vector"] == [0, 0, 0] for item in locked))
            self.assertTrue(all("S" not in item["keys"] for item in locked))
            self.assertTrue(all(item["illegal_branches"] == [] for item in locked))
            self.assertFalse(executor.harpoon_locked)
            relaunched = executor.apply((0, 6, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertEqual(relaunched["action_vector"], [0, 6, 0])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_harpoon_cannot_interrupt_charge_or_its_release(self) -> None:
        path = Path("tests/.three_head_charge_harpoon_guard.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            started = executor.apply(
                (0, 6, 2), branch_masks=ALL_ACTIONS_AVAILABLE
            )
            self.assertEqual(started["action_vector"], [0, 0, 2])
            self.assertFalse(executor.harpoon_locked)
            self.assertTrue(executor.charge_protected)
            held = [
                executor.apply((0, 6, 2), branch_masks=ALL_ACTIONS_AVAILABLE)
                for _ in range(13)
            ]
            self.assertTrue(held[-1]["charge_completed"])
            released = executor.apply(
                (0, 6, 0), branch_masks=ALL_ACTIONS_AVAILABLE
            )
            self.assertEqual(released["action_vector"], [0, 0, 0])
            self.assertIn("attack_x", released["started_branches"])
            self.assertNotIn("skill_s", released["started_branches"])
            self.assertTrue(executor.charge_protected)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_charge_can_release_between_1400_and_3000_ms(self) -> None:
        path = Path("tests/.three_head_charge_window.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            held = [
                executor.apply((0, 0, 2), branch_masks=ALL_ACTIONS_AVAILABLE)
                for _ in range(30)
            ]
            self.assertTrue(held[13]["charge_completed"])
            self.assertFalse(held[13]["charge_at_max"])
            self.assertEqual(held[-1]["charge_elapsed_ms"], 3000)
            self.assertTrue(held[-1]["charge_at_max"])
            forced_release = executor.apply(
                (0, 0, 2), branch_masks=ALL_ACTIONS_AVAILABLE
            )
            self.assertEqual(forced_release["action_vector"], [0, 0, 0])
            self.assertIn("attack_x", forced_release["started_branches"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_incomplete_charge_is_a_minimum_commitment(self) -> None:
        path = Path("tests/.three_head_charge_commitment.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            executor.apply((0, 0, 2), branch_masks=ALL_ACTIONS_AVAILABLE)
            frames = [
                executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
                for _ in range(13)
            ]
            self.assertTrue(all(item["action_vector"][2] == 2 for item in frames))
            self.assertTrue(frames[-1]["charge_completed"])
            released = executor.apply(
                (0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE
            )
            self.assertEqual(released["action_vector"], [0, 0, 0])
            self.assertIn("attack_x", released["started_branches"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_harpoon_control_state_separates_active_and_recovery(self) -> None:
        path = Path("tests/.three_head_harpoon_state.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            executor.apply((0, 6, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            active = executor.control_state({})
            self.assertEqual(active.harpoon_phase, 0.25)
            executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            recovery = executor.control_state({})
            self.assertEqual(recovery.harpoon_phase, 0.25)
            executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertGreater(executor.control_state({}).harpoon_phase, 0.25)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_harpoon_key_is_released_on_the_next_tick(self) -> None:
        path = Path("tests/.three_head_harpoon_pulse.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        executor.send_input = True
        try:
            with patch.object(action_executor, "_send_key") as send_key:
                executor.apply((0, 6, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
                executor.apply((0, 0, 0), branch_masks=ALL_ACTIONS_AVAILABLE)
            calls = [call.args for call in send_key.call_args_list]
            self.assertIn(("S", False), calls)
            self.assertIn(("S", True), calls)
            self.assertLess(calls.index(("S", False)), calls.index(("S", True)))
        finally:
            executor.send_input = False
            executor.close()
            path.unlink(missing_ok=True)

    def test_interruption_clears_harpoon_lock(self) -> None:
        path = Path("tests/.three_head_harpoon_interrupt.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            item = executor.apply(
                (2, 6, 1),
                interrupted=True,
                branch_masks=ALL_ACTIONS_AVAILABLE,
            )
            self.assertEqual(item["action_vector"], [0, 0, 0])
            self.assertNotIn("skill_s", item["started_branches"])
            self.assertFalse(executor.harpoon_locked)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_harpoon_lock_masks_all_policy_branches(self) -> None:
        masks, reasons = branch_availability({}, harpoon_locked=True)
        self.assertEqual(
            masks,
            tuple(tuple(index == 0 for index in range(size)) for size in BRANCH_SIZES),
        )
        self.assertTrue(any("harpoon" in reason for reason in reasons))

    def test_charge_protection_masks_harpoon_only(self) -> None:
        masks, reasons = branch_availability({}, charge_protected=True)
        self.assertFalse(masks[1][6])
        self.assertTrue(masks[1][1])
        self.assertTrue(masks[2][0])
        self.assertTrue(any("charge" in reason for reason in reasons))

    def test_incomplete_charge_masks_combat_to_hold_x(self) -> None:
        masks, reasons = branch_availability({}, charge_must_hold=True)
        self.assertEqual(
            masks[2],
            tuple(index == 2 for index in range(BRANCH_SIZES[2])),
        )
        self.assertTrue(any("keep holding" in reason for reason in reasons))

    def test_taunt_has_one_press_edge_while_held(self) -> None:
        path = Path("tests/.three_head_taunt.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path))
        try:
            first = executor.apply((0, 0, 4), branch_masks=ALL_ACTIONS_AVAILABLE)
            held = executor.apply((0, 0, 4), branch_masks=ALL_ACTIONS_AVAILABLE)
            self.assertIn("taunt_v", first["started_branches"])
            self.assertNotIn("taunt_v", held["started_branches"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_control_state_tracks_compact_executed_action(self) -> None:
        path = Path("tests/.three_head_state.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path), tick_ms=100)
        try:
            executor.apply((2, 4, 2), branch_masks=ALL_ACTIONS_AVAILABLE)
            state = executor.control_state(
                {
                    "player_control": {
                        "jump_available": False,
                        "double_jump_available": False,
                        "dash_available": True,
                        "attack_available": True,
                    }
                }
            )
            self.assertEqual(state.jump_state, 1.0)
            self.assertEqual(state.movement_direction, -1.0)
            self.assertEqual(state.movement_mode, 0.5)
            self.assertAlmostEqual(state.combat_action, 2 / 6)
            self.assertAlmostEqual(state.attack_charge_progress, 100 / 3000)
            self.assertEqual(state.harpoon_phase, 0.0)
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_illegal_combat_action_is_neutralized_and_recorded(self) -> None:
        path = Path("tests/.three_head_illegal.jsonl")
        executor = KeyboardActionExecutor(ActionRecorder(path))
        try:
            masks, reasons = branch_availability({})
            item = executor.apply(
                (0, 0, 3),
                branch_masks=masks,
                masked_reasons=reasons,
                player_resources=decode_player_resources({}),
            )
            self.assertEqual(item["action_vector"], [0, 0, 0])
            self.assertEqual(item["illegal_branches"], ["combat"])
        finally:
            executor.close()
            path.unlink(missing_ok=True)

    def test_validate_action_rejects_wrong_shape_and_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_action((0, 0))
        with self.assertRaises(ValueError):
            validate_action((3, 0, 0))
        with self.assertRaises(ValueError):
            validate_action((0, 7, 0))
        with self.assertRaises(ValueError):
            validate_action((0, 0, 7))


if __name__ == "__main__":
    unittest.main()
