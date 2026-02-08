import torch
import torch.nn as nn

# 3x3 Tic-Tac-Toe only. Observation: (B, 3, 3, 3). Hidden: (B, 3, 3, 3). Actions: 9.

class MuZero(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        self.env = env
        self.device = device
        self.representationFunction = RepresentationFunction()
        self.dynamicsFunction = DynamicsFunction()
        self.predictionFunction = PredictionFunction()

    def represent(self, observation):
        return self.representationFunction(observation)

    def dynamics(self, hidden_state, actions):
        # hidden_state (B, 3, 3, 3), actions (B, 9) one-hot torch tensor -> (B, 1, 3, 3), merge -> (B, 4, 3, 3)
        plane = actions.to(device=hidden_state.device, dtype=hidden_state.dtype).reshape(-1, 1, 3, 3)
        x = torch.cat((hidden_state, plane), dim=1)
        return self.dynamicsFunction(x)

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)


class RepresentationFunction(nn.Module):
    """Observation (B, 3, 3, 3) -> hidden (B, 3, 3, 3)."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DynamicsFunction(nn.Module):
    """(hidden + action planes) (B, 4, 3, 3) -> next hidden (B, 3, 3, 3), reward (B, 1)."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.reward = nn.Sequential(nn.Flatten(), nn.Linear(3 * 3 * 3, 1))

    def forward(self, x):
        h = self.net(x)
        r = self.reward(h)
        return h, r


class PredictionFunction(nn.Module):
    """Hidden (B, 3, 3, 3) -> policy (B, 9), value (B, 1)."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(nn.Flatten(), nn.Linear(16 * 3 * 3, 9))
        self.value = nn.Sequential(nn.Flatten(), nn.Linear(16 * 3 * 3, 1))

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)
