import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroLinearNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.value_support = DiscreteSupport(0, 20)
        self.reward_support = DiscreteSupport(0, 5)
        
        self.predictionFunction = PredictionFunction(self.game, self.value_support)
        self.dynamicsFunction = DynamicsFunction(self.reward_support)
        self.representationFunction = RepresentationFunction()

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        return self.representationFunction(observation)

    def dynamics(self, hidden_state, action):
        actionArr = torch.zeros((hidden_state.shape[0], 2), device=self.device, dtype=torch.float32)
        for i, a in enumerate(action):
            actionArr[i, a] = 1
        x = torch.hstack((hidden_state, actionArr))
        return self.dynamicsFunction(x)

    def inverse_value_transform(self, value):
        return self.inverse_scalar_transform(value, self.value_support)

    def inverse_reward_transform(self, reward):
        return self.inverse_scalar_transform(reward, self.reward_support)

    def inverse_scalar_transform(self, output, support):
        output_propbs = torch.softmax(output, dim=1)
        output_support = torch.ones(output_propbs.shape, dtype=torch.float32, device=self.device)
        output_support[:, :] = torch.tensor([x for x in support.range], device=self.device)
        scalar_output = (output_propbs * output_support).sum(dim=1, keepdim=True)

        epsilon = 0.001
        sign = torch.ones(scalar_output.shape, device=self.device)
        sign[scalar_output < 0] = -1.0
        inverse_scalar_output = sign * (((torch.sqrt(1 + 4 * epsilon * (torch.abs(scalar_output) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        return inverse_scalar_output

    def scalar_transform(self, x):
        epsilon = 0.001
        sign = torch.ones(x.shape, dtype=torch.float32, device=self.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    @staticmethod
    def _phi(x, min, max, set_size: int):
        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = (x - x_low)
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min, x_low - min
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

# Creates hidden state + reward based on old hidden state and action 
class DynamicsFunctionLinear(nn.Module):
    def __init__(self, reward_support):
        super().__init__()
        
        self.startBlock = nn.Sequential(
            nn.Linear(34, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )

        self.rewardBlock = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, reward_support.size),
        )

    def forward(self, x):
        x = self.startBlock(x)
        reward = self.rewardBlock(x)
        return x, reward
    
# Creates policy and value based on hidden state
class PredictionFunctionLinear(nn.Module):
    def __init__(self, game, value_support):
        super().__init__()
        self.game = game
        
        self.startBlock = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(64, self.game.action_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, value_support.size), 
        )

    def forward(self, x):
        x = self.startBlock(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

# Creates initial hidden state based on observation | several observations
class RepresentationFunctionLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.startBlock(x)
        return x
