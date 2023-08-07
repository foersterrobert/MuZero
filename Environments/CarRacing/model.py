import torch.nn as nn
import torch
import torch.nn.functional as F

class MuZero(nn.Module):
    def __init__(self, game, device):
        super().__init__()
        self.device = device

        self.value_support = DiscreteSupport(-20, 20)
        self.reward_support = DiscreteSupport(-5, 5)
        
        self.predictionFunction = PredictionFunction(game, self.value_support)
        self.dynamicsFunction = DynamicsFunction(self.reward_support)
        self.representationFunction = RepresentationFunction(game)

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        return self.representationFunction(observation)

    def dynamics(self, hidden_state, action):
        actionArr = torch.zeros((hidden_state.shape[0], 1, 6, 6), device=hidden_state.device, dtype=torch.float32)
        for i, a in enumerate(action):
            actionArr[i, 0, a] = 1
        x = torch.cat((hidden_state, actionArr), dim=1)
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
        sign = torch.sign(scalar_output)
        inverse_scalar_output = sign * (((torch.sqrt(1 + 4 * epsilon * (torch.abs(scalar_output) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        return inverse_scalar_output

    def scalar_transform(self, x):
        epsilon = 0.001
        sign = torch.sign(x)
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    def _phi(self, x, min, max, set_size):
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
class DynamicsFunction(nn.Module):
    def __init__(self, reward_support, num_resBlocks=4, hidden_planes=32):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(4, hidden_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU()
        )
        self.resBlocks = nn.ModuleList([ResBlock(hidden_planes, hidden_planes) for _ in range(num_resBlocks)])
        self.endBlock = nn.Sequential(
            nn.Conv2d(hidden_planes, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
        )

        self.rewardBlock = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(6 * 6, reward_support.size)
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.resBlocks:
            x = block(x)
        x = self.endBlock(x)
        reward = self.rewardBlock(x)
        return x, reward
    
# Creates policy and value based on hidden state
class PredictionFunction(nn.Module):
    def __init__(self, game, value_support, num_resBlocks=4, hidden_planes=32):
        super().__init__()
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, hidden_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU()
        )
        self.resBlocks = nn.ModuleList([ResBlock(hidden_planes, hidden_planes) for _ in range(num_resBlocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, game.action_size)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 6 * 6, 32),
            nn.ReLU(),
            nn.Linear(32, value_support.size),
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.resBlocks:
            x = block(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

# Creates initial hidden state based on observation | several observations
class RepresentationFunction(nn.Module):
    def __init__(self, game, hidden_planes=32):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(game.sequence_lenth, hidden_planes // 2, kernel_size=3, stride=2, padding=1), # 48x48
            nn.BatchNorm2d(hidden_planes // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_planes // 2, hidden_planes, kernel_size=3, stride=2, padding=1), # 24x24
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU(),
            ResBlock(hidden_planes, hidden_planes),
            nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=2, padding=1), # 12x12
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU(),
            ResBlock(hidden_planes, hidden_planes),
            nn.Conv2d(hidden_planes, hidden_planes // 2, kernel_size=3, stride=2, padding=1), # 6x6
            nn.BatchNorm2d(hidden_planes // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_planes // 2, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class DiscreteSupport:
    def __init__(self, min, max):
        assert min < max
        self.min = min
        self.max = max
        self.range = range(min, max + 1)
        self.size = len(self.range)
