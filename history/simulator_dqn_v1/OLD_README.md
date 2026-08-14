# Boss Dodge Double DQN

This package replaces the tabular Expected SARSA agent with a PyTorch Double
DQN. It uses a normalized 18-value observation, an MLP Q network, replay
memory, a target network, epsilon-greedy exploration, invalid-action masking,
Huber loss, and gradient clipping.

Run commands from the repository root so Python can find the package under
`src`.

```powershell
python -m pip install torch

# Start a new run. CUDA is selected automatically when available.
python -m hk_rl_DQN.train_dqn --episodes 3000 --reset

# Resume from checkpoints/dqn.pt.
python -m hk_rl_DQN.train_dqn --episodes 1000

# Force CPU or GPU explicitly.
python -m hk_rl_DQN.train_dqn --episodes 3000 --reset --device cpu
python -m hk_rl_DQN.train_dqn --episodes 3000 --reset --device cuda

# Replay the learned policy.
python -m hk_rl_DQN.visualize --checkpoint checkpoints/dqn.pt

# Play manually.
python -m hk_rl_DQN.visualize --manual
```

The resumable network and optimizer checkpoint is written to
`checkpoints/dqn.pt`. Episode rewards, losses, exploration rate, and evaluation
metrics are written to `runs/dqn.json`.

The complete implementation is in `train_dqn.py`. `train_q.py` remains as a
small compatibility entry point for older commands and imports.

Runtime modules live directly in `src/hk_rl_DQN`. Automated tests are under
`tests/hk_rl_DQN`; explicit real-game acceptance commands are under
`src/hk_rl_DQN/tools`. The live state cold-start check is:

```powershell
python -m hk_rl_DQN.tools.cold_start_state_test
```
