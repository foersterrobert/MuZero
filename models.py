import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZero(nn.Module):
    def __init__(self, game):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamicsFunction = DynamicsFunction()
        self.predictionFunction = PredictionFunction(game)
        self.representationFunction = RepresentationFunction()

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        return self.representationFunction(observation)

    def dynamics(self, hidden_state, action):
        row = action // 3
        col = action % 3
        action = torch.zeros((1, 1, 3, 3)).to(self.device)
        action[0, 0, row, col] = 1
        x = torch.cat((hidden_state, action), dim=1)
        return self.dynamicsFunction(x)

# Creates hidden state + reward based on old hidden state and action 
class DynamicsFunction(nn.Module):
    def __init__(self, num_resBlocks=16, hidden_planes=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            nn.Linear(9, 1)
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
    def __init__(self, game, num_resBlocks=20, hidden_planes=256):
        super().__init__()
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, hidden_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU()
        )
        self.resBlocks = nn.ModuleList([ResBlock(hidden_planes, hidden_planes) for _ in range(num_resBlocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.game.row_count * self.game.column_count, self.game.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.game.row_count * self.game.column_count, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
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
    def __init__(self, num_resBlocks=16, hidden_planes=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, hidden_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU()
        )
        self.resBlocks = nn.ModuleList([ResBlock(hidden_planes, hidden_planes) for _ in range(num_resBlocks)])
        self.endBlock = nn.Sequential(
            nn.Conv2d(hidden_planes, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.resBlocks:
            x = block(x)
        x = self.endBlock(x)
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
