import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroResNet(nn.Module):
    def __init__(self, game, device):
        super().__init__()
        self.game = game
        self.device = device

        self.predictionFunction = PredictionFunctionResNet(game)
        self.dynamicsFunction = DynamicsFunctionResNet()
        self.representationFunction = RepresentationFunctionResNet()

        self.to(device)

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        return self.representationFunction(observation)

    def dynamics(self, hidden_state, actions):
        actionPlane = torch.zeros((hidden_state.shape[0], 1, self.game.row_count, self.game.column_count), device=self.device, dtype=torch.float32)
        for i, a in enumerate(actions):
            row = a // self.game.column_count
            col = a % self.game.column_count
            actionPlane[i, 0, row, col] = 1
        x = torch.cat((hidden_state, actionPlane), dim=1)
        hidden_state = self.dynamicsFunction(x)
        return self.normalize_hidden_state(hidden_state)
    
    def normalize_hidden_state(self, hidden_state):
        _min = hidden_state.min(dim=1, keepdim=True)[0]
        _max = hidden_state.max(dim=1, keepdim=True)[0]
        return (hidden_state - _min) / (_max - _min + 1e-6)
    
class DynamicsFunctionResNet(nn.Module):
    def __init__(self, num_resBlocks=2, num_hidden=16):
        super().__init__()
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(4, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])
        self.endBlock = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
        )

    def forward(self, x):
        x = self.startBlock(x)
        for resblock in self.backBone:
            x = resblock(x)
        x = self.endBlock(x)
        return x

class PredictionFunctionResNet(nn.Module):
    def __init__(self, game, num_resBlocks=2, num_hidden=16):
        super().__init__()

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_hidden // 2 * game.row_count * game.column_count, game.action_size)
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
 
class RepresentationFunctionResNet(nn.Module):
    def __init__(self, num_hidden=16):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, num_hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden // 2),
            nn.ReLU(),
            nn.Conv2d(num_hidden // 2, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            # ResBlock(num_hidden),
            nn.Conv2d(num_hidden, num_hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden // 2),
            nn.ReLU(),
            nn.Conv2d(num_hidden // 2, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x