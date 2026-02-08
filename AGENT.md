# Project Overview
Build a minimal version of MuZero for the simple 3x3 Tic-Tac-Toe game.

# Plan
1. **Environment** – 3x3 Tic-Tac-Toe: state, actions, encoding, win/draw/valid moves. *(Done)*
2. **Model** – MuZero net for 3x3 only: representation, dynamics (hidden + one-hot action plane), prediction (policy + value). *(Done)*
3. **MCTS** – Monte Carlo tree search using the model: selection, expansion, backup; support for batched inference where useful.
4. **Self-play** – Generate games by running MCTS at each step; store trajectories (observations, actions, rewards, policies).
5. **Training** – Replay buffer, sample batches, MuZero loss (policy, value, reward); train representation, dynamics, prediction.
6. **Evaluation** – Play vs random or greedy baseline; log win/draw rates.

# Architecture
- **Observation / state**: 3x3 board, encoded as 3 channels (one-hot: -1, 0, 1). Shape `(B, 3, 3, 3)`.
- **Representation**: `(B, 3, 3, 3)` → hidden `(B, 3, 3, 3)` (small conv stack).
- **Dynamics**: Hidden `(B, 3, 3, 3)` + one-hot action plane `(B, 1, 3, 3)` → next hidden `(B, 3, 3, 3)` and reward `(B, 1)`. Input shape `(B, 4, 3, 3)`.
- **Prediction**: Hidden `(B, 3, 3, 3)` → policy logits `(B, 9)`, value `(B, 1)`.
- **Actions**: 9 discrete moves (cells); action encoding for dynamics is a single 3x3 one-hot plane (1 at played cell).
