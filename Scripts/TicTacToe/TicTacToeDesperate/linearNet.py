import torch
import torch.nn as nn

class MuZeroLinear(nn.Module):
    def __init__(self, game, device):
        super().__init__()
        self.game = game
        self.device = device

        self.predictionFunction = PredictionFunctionLinear(game)
        self.dynamicsFunction = DynamicsFunctionLinear(game)
        self.representationFunction = RepresentationFunctionLinear(game)

        self.to(device)

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        hidden_state = self.representationFunction(observation)
        return self.normalize_hidden_state(hidden_state)

    def dynamics(self, hidden_state, actions):
        actionPlane = torch.zeros((hidden_state.shape[0], self.game.action_size), device=self.device, dtype=torch.float32)
        for i, a in enumerate(actions):
            actionPlane[i, a] = 1
        x = torch.cat((hidden_state, actionPlane), dim=1)
        hidden_state = self.dynamicsFunction(x)
        return self.normalize_hidden_state(hidden_state)
    
    def normalize_hidden_state(self, hidden_state):
        _min = hidden_state.min(dim=1, keepdim=True)[0]
        _max = hidden_state.max(dim=1, keepdim=True)[0]
        return (hidden_state - _min) / (_max - _min + 1e-6)

class DynamicsFunctionLinear(nn.Module):
    def __init__(self, game):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32 + game.action_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
class PredictionFunctionLinear(nn.Module):
    def __init__(self, game):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, game.action_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v
    
class RepresentationFunctionLinear(nn.Module):
    def __init__(self, game):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(game.row_count * game.column_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x
