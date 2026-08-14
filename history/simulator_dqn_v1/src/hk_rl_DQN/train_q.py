"""Backward-compatible entry point for the Double DQN trainer.

The complete implementation lives in :mod:`hk_rl_DQN.train_dqn`.
"""

try:
    from .train_dqn import *  # noqa: F403 - preserve the former public API
    from .train_dqn import main
except ImportError:  # Support `python -m train_q` from this directory.
    from train_dqn import *  # noqa: F403
    from train_dqn import main


if __name__ == "__main__":
    main()
