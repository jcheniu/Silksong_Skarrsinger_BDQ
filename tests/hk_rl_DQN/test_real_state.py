import unittest

from hk_rl_DQN.real_state import (
    BASE_STATE_DIMENSIONS,
    BOSS_SEMANTIC_FEATURES,
    KINEMATIC_STATE_DIMENSIONS,
    KeyHoldState,
    STATE_DIMENSIONS,
    decode_jsonl,
    encode_snapshot,
)


def snapshot(state: str = "Throw Antic") -> dict[str, object]:
    return {
        "type": "snapshot",
        "timestamp": 2.5,
        "frame": 150,
        "scene": "Memory_Ant_Queen",
        "player": {"x": 150.0, "y": 19.5, "velocity_x": 2.0, "velocity_y": 0.0},
        "boss": {"x": 170.0, "y": 21.5, "velocity_x": -3.0, "velocity_y": 1.0},
        "player_resources": {
            "silk": 3,
            "silk_max": 9,
            "silk_parts": 2,
            "skill_cost": 4,
            "silk_abilities_disabled": False,
            "skill_available": True,
            "spell_available": False,
        },
        "fsm": [
            {"path": "Boss Scene/Hunter Queen Boss", "name": "Control", "state": state},
            {"path": "Boss Scene/Hunter Queen Boss", "name": "Stun Control", "state": "Idle"},
        ],
    }


class RealStateTests(unittest.TestCase):
    def test_observation_is_fixed_size(self) -> None:
        frame = encode_snapshot(snapshot())
        self.assertEqual(len(frame.observation), STATE_DIMENSIONS)
        self.assertEqual(STATE_DIMENSIONS, 24)
        self.assertEqual(frame.attack_type, "ground_throw")
        self.assertEqual(frame.attack_phase, "anticipation")

    def test_attack_families_are_distinct(self) -> None:
        self.assertNotEqual(
            encode_snapshot(snapshot("Slash 3")).observation,
            encode_snapshot(snapshot("Cyclone 2")).observation,
        )
        self.assertEqual(encode_snapshot(snapshot("Slash 3")).attack_type, "slash")
        self.assertEqual(encode_snapshot(snapshot("Air Throw")).attack_type, "air_throw")
        self.assertEqual(encode_snapshot(snapshot("Launch Spin")).attack_type, "spin_attack")

    def test_numbered_boss_states_become_continuous_progress(self) -> None:
        offset = KINEMATIC_STATE_DIMENSIONS + 1
        progress_index = offset + BOSS_SEMANTIC_FEATURES.index("behavior_progress")
        first = encode_snapshot(snapshot("Slash 1")).observation[progress_index]
        middle = encode_snapshot(snapshot("Slash 5")).observation[progress_index]
        last = encode_snapshot(snapshot("Slash 9")).observation[progress_index]
        self.assertAlmostEqual(first, 0.2)
        self.assertAlmostEqual(middle, 0.5)
        self.assertAlmostEqual(last, 0.8)

    def test_semantic_features_explain_attack_geometry(self) -> None:
        offset = KINEMATIC_STATE_DIMENSIONS + 1
        features = dict(
            zip(
                BOSS_SEMANTIC_FEATURES,
                encode_snapshot(snapshot("Air Throw Slash")).observation[
                    offset : offset + len(BOSS_SEMANTIC_FEATURES)
                ],
            )
        )
        self.assertEqual(features["attack_category"], 1.0)
        self.assertEqual(features["aerial"], 1.0)

    def test_reaction_and_phase_event_are_separate(self) -> None:
        value = snapshot("P2 Roar")
        value["fsm"].append(
            {"path": "Boss Scene/Hunter Queen Boss", "name": "Stun Control", "state": "Stunned"}
        )
        frame = encode_snapshot(value)
        self.assertEqual(frame.phase_event, "phase_transition")
        self.assertEqual(frame.reaction, "stunned")

    def test_hornet_dead_is_not_boss_dead(self) -> None:
        self.assertEqual(encode_snapshot(snapshot("Hornet Dead")).reaction, "normal")

    def test_jsonl_decoder_skips_lifecycle_lines(self) -> None:
        lines = ['{"type":"telemetry_start"}', json_line(snapshot()), '{"type":"telemetry_stop"}']
        self.assertEqual(len(list(decode_jsonl(lines))), 1)

    def test_control_state_is_appended_to_observation(self) -> None:
        controls = KeyHoldState(
            jump_state=1.0,
            movement_direction=-1.0,
            movement_mode=0.5,
            combat_action=2 / 6,
            attack_charge_progress=0.5,
            harpoon_phase=0.4,
        )
        observation = encode_snapshot(snapshot(), controls).observation
        self.assertEqual(observation[BASE_STATE_DIMENSIONS:], controls.as_tuple())

    def test_compact_action_state_is_part_of_the_observation(self) -> None:
        controls = KeyHoldState(
            jump_state=0.5,
            movement_direction=1.0,
            movement_mode=1.0,
            combat_action=0.5,
            attack_charge_progress=0.75,
            harpoon_phase=0.25,
        )
        values = encode_snapshot(snapshot(), controls).observation[BASE_STATE_DIMENSIONS:]
        self.assertEqual(values, controls.as_tuple())

    def test_silk_is_normalized_and_exposed(self) -> None:
        frame = encode_snapshot(snapshot())
        self.assertEqual(frame.resources.silk, 3)
        self.assertEqual(frame.resources.silk_max, 9)
        self.assertTrue(frame.resources.is_complete)
        self.assertAlmostEqual(frame.observation[KINEMATIC_STATE_DIMENSIONS], 1 / 3)

    def test_missing_silk_is_encoded_conservatively(self) -> None:
        value = snapshot()
        del value["player_resources"]
        frame = encode_snapshot(value)
        self.assertIsNone(frame.resources.silk)
        self.assertFalse(frame.resources.is_complete)
        self.assertEqual(frame.observation[KINEMATIC_STATE_DIMENSIONS], 0.0)

    def test_relative_velocity_and_player_facing_remain_informative(self) -> None:
        value = snapshot()
        value["player"]["facing"] = 1.0
        value["boss"]["velocity_x"] = 70.0
        observation = encode_snapshot(value).observation
        self.assertEqual(observation[9], 1.0)
        self.assertGreater(observation[6], 0.8)
        self.assertLess(observation[6], 1.0)

    def test_player_health_is_not_exposed_to_the_observation(self) -> None:
        healthy = snapshot()
        healthy["player_health"] = {"health": 10, "max_health": 10}
        one_hit = snapshot()
        one_hit["player_health"] = {"health": 1, "max_health": 10}
        self.assertEqual(
            encode_snapshot(healthy).observation,
            encode_snapshot(one_hit).observation,
        )

    def test_relative_velocity_encodes_closing_motion(self) -> None:
        approaching = snapshot()
        approaching["player"]["velocity_x"] = 2.0
        approaching["boss"]["velocity_x"] = -3.0
        retreating = snapshot()
        retreating["player"]["velocity_x"] = -3.0
        retreating["boss"]["velocity_x"] = 2.0
        self.assertLess(encode_snapshot(approaching).observation[6], 0.0)
        self.assertGreater(encode_snapshot(retreating).observation[6], 0.0)


def json_line(value: dict[str, object]) -> str:
    import json

    return json.dumps(value)
