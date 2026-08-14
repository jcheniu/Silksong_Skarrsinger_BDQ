from hk_rl.boss_env import BossDodgeEnv
from hk_rl.train_q import evaluate, train


def test_reset_is_reproducible():
    """The same seed should produce the same initial state.

    相同的种子应产生相同的初始状态。
    """
    first = BossDodgeEnv(seed=1).reset()[0]
    second = BossDodgeEnv(seed=1).reset()[0]
    assert first == second


def test_invalid_action_is_rejected():
    """Fail early instead of silently training on an undefined action.

    应尽早失败，而不是让智能体在未定义的动作上静默训练。
    """
    env = BossDodgeEnv(seed=1)
    env.reset()
    try:
        env.step(99)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action should raise ValueError")


def test_training_returns_evaluable_policy():
    """Even a short run should produce metrics with valid ranges.

    即使短时间运行，也应产生范围有效的指标。
    """
    _, q_data = train(episodes=20, seed=3)
    metrics = evaluate(q_data, episodes=5, seed=4)
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["average_damage_taken"] >= 0.0
