import unittest

from hk_rl_DQN.real_reward import (
    ATTACK_RANGE_REWARD,
    DAMAGE_REWARD_PER_HP,
    DODGE_REWARD,
    PLAYER_HURT_PENALTY,
    PLAYER_DAMAGE_PENALTY_PER_HP,
    PLAYER_PARRY_REWARD,
    STEP_PENALTY,
    SILK_SPEND_PENALTY_PER_UNIT,
    VICTORY_REWARD,
    RewardTracker,
)


def snapshot(
    state: str = "Movement 1",
    *,
    health: int = 10,
    player_x: float = 150.0,
    boss_x: float = 170.0,
    stun_state: str = "Idle",
    death_state: str = "",
    boss_damage_events: int | None = None,
    boss_damage_total: int | None = None,
    silk: int | None = None,
    timestamp: float = 1.0,
    player_parry_events: int | None = None,
) -> dict[str, object]:
    fsms = [
        {"path": "Boss Scene/Hunter Queen Boss", "name": "Control", "state": state},
        {"path": "Boss Scene/Hunter Queen Boss", "name": "Stun Control", "state": stun_state},
    ]
    if death_state:
        fsms.append(
            {
                "path": "Boss Scene/Hunter Queen Boss/Corpse Hunter Queen(Clone)",
                "name": "Death",
                "state": death_state,
            }
        )
    result = {
        "type": "snapshot",
        "timestamp": timestamp,
        "frame": 60,
        "scene": "Memory_Ant_Queen",
        "player_grounded": True,
        "player": {"x": player_x, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
        "player_health": {"health": health, "max_health": 10},
        "boss": {"x": boss_x, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
        "fsm": fsms,
    }
    if boss_damage_events is not None:
        result["boss_damage_events"] = boss_damage_events
    if boss_damage_total is not None:
        result["boss_damage_total"] = boss_damage_total
    if silk is not None:
        result["player_resources"] = {
            "silk": silk,
            "silk_max": 9,
            "silk_parts": 0,
            "skill_cost": 4,
            "silk_abilities_disabled": False,
            "skill_available": True,
            "spell_available": True,
        }
    if player_parry_events is not None:
        result["player_parry_events"] = player_parry_events
    return result


class RewardTrackerTests(unittest.TestCase):
    def test_step_penalty_is_always_present(self) -> None:
        reward = RewardTracker().step(snapshot())
        self.assertEqual(reward.total, STEP_PENALTY)

    def test_attack_range_reward_is_once_per_episode(self) -> None:
        tracker = RewardTracker()
        first = tracker.step(snapshot(player_x=165.0))
        second = tracker.step(snapshot(player_x=165.0))
        self.assertEqual(first.entered_attack_range, ATTACK_RANGE_REWARD)
        self.assertEqual(second.entered_attack_range, 0.0)

    def test_health_drop_is_one_hurt_event(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot(health=10))
        reward = tracker.step(snapshot(health=8))
        self.assertEqual(reward.player_health_lost, 2)
        self.assertEqual(reward.player_damage_taken, 2)
        self.assertEqual(PLAYER_DAMAGE_PENALTY_PER_HP, -3.6)
        self.assertEqual(PLAYER_HURT_PENALTY, PLAYER_DAMAGE_PENALTY_PER_HP)
        self.assertEqual(reward.player_hurt, -7.2)

    def test_completed_attack_without_health_loss_rewards_dodge(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Slash Antic", timestamp=1.0))
        held_antic = tracker.step(snapshot("Slash Antic", timestamp=1.1))
        self.assertEqual(held_antic.dodge, 0.0)
        tracker.step(snapshot("Slash 3", timestamp=1.2))
        reward = tracker.step(snapshot("Slash End", timestamp=1.3))
        self.assertEqual(reward.dodge, 0.0)
        tracker.step(snapshot("Movement 1", timestamp=1.4))
        reward = tracker.step(snapshot("Movement 1", timestamp=1.6))
        self.assertEqual(reward.attack_finished, "slash")
        self.assertEqual(reward.dodge, DODGE_REWARD)
        self.assertEqual(DODGE_REWARD, 0.6)
        self.assertFalse(reward.attack_hurt_player)

    def test_hurt_during_attack_cancels_dodge(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Throw Antic", health=10, timestamp=1.0))
        tracker.step(snapshot("Throw 1", health=9, timestamp=1.1))
        tracker.step(snapshot("Movement 1", health=9, timestamp=1.2))
        reward = tracker.step(snapshot("Movement 1", health=9, timestamp=1.4))
        self.assertEqual(reward.dodge, 0.0)
        self.assertTrue(reward.attack_hurt_player)

    def test_player_parry_during_boss_attack_uses_configured_reward(self) -> None:
        tracker = RewardTracker()
        tracker.step(
            snapshot("Slash Antic", timestamp=1.0, player_parry_events=0)
        )
        reward = tracker.step(
            snapshot("Slash 1", timestamp=1.1, player_parry_events=1)
        )
        self.assertEqual(reward.player_parries, 1)
        self.assertEqual(reward.parry_reward, PLAYER_PARRY_REWARD)
        self.assertEqual(PLAYER_PARRY_REWARD, 0.8)

    def test_player_parry_counter_outside_boss_attack_is_not_rewarded(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot(timestamp=1.0, player_parry_events=0))
        reward = tracker.step(snapshot(timestamp=1.1, player_parry_events=1))
        self.assertEqual(reward.player_parries, 0)
        self.assertEqual(reward.parry_reward, 0.0)

    def test_attack_without_active_phase_does_not_reward_dodge(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Slash Antic", timestamp=1.0))
        tracker.step(snapshot("Movement 1", timestamp=1.1))
        reward = tracker.step(snapshot("Movement 1", timestamp=1.4))
        self.assertEqual(reward.dodge, 0.0)

    def test_nearby_attack_types_form_one_hurt_sensitive_combo(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Slash Antic", health=10, timestamp=1.0))
        tracker.step(snapshot("Slash 3", health=10, timestamp=1.1))
        tracker.step(snapshot("Spin Antic", health=10, timestamp=1.2))
        tracker.step(snapshot("Spin Attack", health=10, timestamp=1.3))
        tracker.step(snapshot("Movement 1", health=9, timestamp=1.4))
        reward = tracker.step(snapshot("Movement 1", health=9, timestamp=1.6))
        self.assertEqual(reward.attack_finished, "slash+spin_attack")
        self.assertEqual(reward.dodge, 0.0)
        self.assertTrue(reward.attack_hurt_player)

    def test_attack_after_combo_gap_starts_a_new_window(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Slash Antic", timestamp=1.0))
        tracker.step(snapshot("Slash 3", timestamp=1.1))
        reward = tracker.step(snapshot("Spin Antic", timestamp=1.4))
        self.assertEqual(reward.attack_finished, "slash")
        self.assertEqual(reward.dodge, DODGE_REWARD)

    def test_damage_total_scales_reward_proportionally(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Movement 1", boss_damage_total=0))
        five_damage = tracker.step(snapshot("Cyclone Antic", boss_damage_total=5))
        self.assertEqual(five_damage.damage_deal, 5)
        self.assertEqual(DAMAGE_REWARD_PER_HP, 0.1)
        self.assertEqual(five_damage.damage_reward, 0.5)
        twenty_damage = tracker.step(snapshot("Cyclone 1", boss_damage_total=25))
        self.assertEqual(twenty_damage.damage_deal, 20)
        self.assertEqual(twenty_damage.damage_reward, 2.0)

    def test_damage_counter_reset_is_only_a_baseline(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot(boss_damage_total=4))
        reset = tracker.step(snapshot(boss_damage_total=0))
        self.assertEqual(reset.damage_deal, 0)

    def test_silk_spending_has_a_light_proportional_penalty(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot(silk=9))
        reward = tracker.step(snapshot(silk=5))
        self.assertEqual(reward.silk_spent, 4)
        self.assertEqual(SILK_SPEND_PENALTY_PER_UNIT, -0.04)
        self.assertEqual(reward.silk_penalty, 4 * SILK_SPEND_PENALTY_PER_UNIT)
        self.assertEqual(reward.total, STEP_PENALTY + reward.silk_penalty)

    def test_hit_and_victory_rewards_are_edge_triggered(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot())
        hit = tracker.step(snapshot(stun_state="Stunned"))
        held = tracker.step(snapshot(stun_state="Stunned"))
        self.assertEqual(hit.damage_deal, 1)
        self.assertEqual(hit.damage_reward, DAMAGE_REWARD_PER_HP)
        self.assertEqual(held.damage_reward, 0.0)

        victory = tracker.step(snapshot(death_state="Heart Death"))
        repeated = tracker.step(snapshot(death_state="Heart Death"))
        self.assertEqual(victory.victory, VICTORY_REWARD)
        self.assertTrue(victory.terminated)
        self.assertEqual(repeated.victory, 0.0)


if __name__ == "__main__":
    unittest.main()
