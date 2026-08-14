import unittest

from hk_rl_DQN.real_reward import (
    ATTACK_RANGE_REWARD,
    BOSS_HIT_REWARD,
    DODGE_REWARD,
    PLAYER_HURT_PENALTY,
    STEP_PENALTY,
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
    return {
        "type": "snapshot",
        "timestamp": 1.0,
        "frame": 60,
        "scene": "Memory_Ant_Queen",
        "player_grounded": True,
        "player": {"x": player_x, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
        "player_health": {"health": health, "max_health": 10},
        "boss": {"x": boss_x, "y": 20.0, "velocity_x": 0.0, "velocity_y": 0.0},
        "fsm": fsms,
    }


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
        self.assertEqual(reward.player_hurt, PLAYER_HURT_PENALTY)

    def test_completed_attack_without_health_loss_rewards_dodge(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Slash Antic"))
        held_antic = tracker.step(snapshot("Slash Antic"))
        self.assertEqual(held_antic.dodge, 0.0)
        tracker.step(snapshot("Slash 3"))
        reward = tracker.step(snapshot("Slash End"))
        self.assertEqual(reward.dodge, 0.0)
        reward = tracker.step(snapshot("Movement 1"))
        self.assertEqual(reward.attack_finished, "slash")
        self.assertEqual(reward.dodge, DODGE_REWARD)

    def test_hurt_during_attack_cancels_dodge(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot("Throw Antic", health=10))
        tracker.step(snapshot("Throw 1", health=9))
        reward = tracker.step(snapshot("Movement 1", health=9))
        self.assertEqual(reward.dodge, 0.0)

    def test_hit_and_victory_rewards_are_edge_triggered(self) -> None:
        tracker = RewardTracker()
        tracker.step(snapshot())
        hit = tracker.step(snapshot(stun_state="Stunned"))
        held = tracker.step(snapshot(stun_state="Stunned"))
        self.assertEqual(hit.boss_hit, BOSS_HIT_REWARD)
        self.assertEqual(held.boss_hit, 0.0)

        victory = tracker.step(snapshot(death_state="Heart Death"))
        repeated = tracker.step(snapshot(death_state="Heart Death"))
        self.assertEqual(victory.victory, VICTORY_REWARD)
        self.assertTrue(victory.terminated)
        self.assertEqual(repeated.victory, 0.0)


if __name__ == "__main__":
    unittest.main()
