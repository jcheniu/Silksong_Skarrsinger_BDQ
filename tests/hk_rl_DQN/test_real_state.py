import unittest

from hk_rl_DQN.real_state import (
    BASE_STATE_DIMENSIONS,
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
            attack_held=True,
            attack_hold_progress=0.5,
            dash_held=True,
            dash_hold_progress=0.25,
            skill_held=True,
            skill_hold_progress=0.75,
            interrupted=True,
            skill_available=True,
            spell_available=False,
        )
        observation = encode_snapshot(snapshot(), controls).observation
        self.assertEqual(observation[BASE_STATE_DIMENSIONS:], controls.as_tuple())

    def test_silk_is_normalized_and_exposed(self) -> None:
        frame = encode_snapshot(snapshot())
        self.assertEqual(frame.resources.silk, 3)
        self.assertEqual(frame.resources.silk_max, 9)
        self.assertTrue(frame.resources.is_complete)
        self.assertAlmostEqual(frame.observation[9], 1 / 3)

    def test_missing_silk_is_encoded_conservatively(self) -> None:
        value = snapshot()
        del value["player_resources"]
        frame = encode_snapshot(value)
        self.assertIsNone(frame.resources.silk)
        self.assertFalse(frame.resources.is_complete)
        self.assertEqual(frame.observation[9], 0.0)


def json_line(value: dict[str, object]) -> str:
    import json

    return json.dumps(value)
