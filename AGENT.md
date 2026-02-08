# Project Overview
Minimal MuZero for 3×3 Tic-Tac-Toe: torch-based environment and small conv model (representation, dynamics, prediction).

# Questions
How to handle termination in dynamics MCTS?
How to handle change of perspective in dynamics MCTS? (for value and state)
Irwie store reward in nodes?
Bei train irwie gehe K steps in die Zukunft
Beachte value loss weight

---

# Plan
| Step | Description | Status |
|------|-------------|--------|
| 1 | **Environment** – 3×3 Tic-Tac-Toe (state, actions, encoding, win/draw, valid moves) | Done |
| 2 | **Model** – Representation, dynamics (hidden + one-hot action plane), prediction (policy + value) | Done |
| 3 | **MCTS** – Tree search using the model (selection, expansion, backup); batched inference where useful | Todo |
| 4 | **Self-play** – Generate games with MCTS; store trajectories (observations, actions, rewards, policies) | Todo |
| 5 | **Training** – Replay buffer, MuZero loss (policy, value, reward); train all three networks | Todo |
| 6 | **Evaluation** – Play vs random/greedy baseline; log win/draw rates | Todo |

---

# Conventions
- **Environment**: `state` is always a tensor of shape **(3, 3)**. `action` is always an **int** in `[0, 8]` (cell index, row-major). Env runs on CPU by default.
- **Model**: All inputs/outputs are **batched** (batch dimension `B`). Single-state callers must add a batch dimension (e.g. `observation.unsqueeze(0)`).

---

# Architecture

## Environment (`env.py`)
- **State**: Tensor shape **(3, 3)**. Values: `1` (current player), `-1` (opponent), `0` (empty).
- **Encoded state**: `get_encoded_state(state)` → **(3, 3, 3)** (channels: one-hot for -1, 0, 1). For the model, use `.unsqueeze(0)` to get **(1, 3, 3, 3)**.
- **Valid actions**: `get_valid_actions(state)` → **(9,)**, float 0/1.
- **Action**: Integer in `[0, 8]`; cell index = `row * 3 + col`. Mutates `state` in place in `get_next_state`.

## Model (`model.py`)
| Component | Input | Output |
|-----------|--------|--------|
| **Representation** | Observation **(B, 3, 3, 3)** | Hidden state **(B, 3, 3, 3)** |
| **Dynamics** | Hidden **(B, 3, 3, 3)** + actions **(B, 9)** one-hot (torch) | Next hidden **(B, 3, 3, 3)**, reward **(B, 1)** (We assume it already changes the perspective of the game for TicTacToe) |
| **Prediction** | Hidden **(B, 3, 3, 3)** | Policy logits **(B, 9)**, value **(B, 1)** |

- **Dynamics action encoding**: `actions` (B, 9) is reshaped to **(B, 1, 3, 3)** and concatenated with hidden state → dynamics input **(B, 4, 3, 3)** (one extra channel, not nine).
- **Actions**: 9 discrete moves; one-hot over 9 cells, row-major.
