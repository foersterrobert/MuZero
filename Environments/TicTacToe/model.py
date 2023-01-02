import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MuZeroResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.predictionFunction = PredictionFunctionResNet(**self.args['predictionFunction'])
        self.dynamicsFunction = DynamicsFunctionResNet(**self.args['dynamicsFunction'])
        self.representationFunction = RepresentationFunctionResNet(**self.args['representationFunction'])

    def __repr__(self):
        return "model"

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        return self.representationFunction(observation)

    def dynamics(self, hidden_state, action):
        actionT = torch.zeros((hidden_state.shape[0], 1, 3, 3)).to(hidden_state.device)
        for i in range(hidden_state.shape[0]):
            row = action[i] // 3
            col = action[i] % 3
            actionT[i, 0, row, col] = 1
        x = torch.cat((hidden_state, actionT), dim=1)
        hidden_state, _ = self.dynamicsFunction(x)
        return hidden_state, 0

# Creates hidden state + reward based on old hidden state and action 
class DynamicsFunctionResNet(nn.Module):
    def __init__(self, num_resBlocks=16, hidden_planes=256, predict_reward=True, reward_support_size=1):
        super().__init__()
        self.predict_reward = predict_reward
        
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

        if self.predict_reward:
            self.rewardBlock = nn.Sequential(
                nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
                nn.Flatten(),
                nn.Linear(9, reward_support_size)
            )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.resBlocks:
            x = block(x)
        x = self.endBlock(x)
        if self.predict_reward:
            reward = self.rewardBlock(x)
            return x, reward
        return x, 0
    
# Creates policy and value based on hidden state
class PredictionFunctionResNet(nn.Module):
    def __init__(self, num_resBlocks=20, hidden_planes=256, screen_size=9, action_size=9, value_support_size=1, value_activation='tanh'):
        super().__init__()
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, hidden_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU()
        )
        self.resBlocks = nn.ModuleList([ResBlock(hidden_planes, hidden_planes) for _ in range(num_resBlocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_planes, hidden_planes // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_planes // 2 * screen_size, action_size)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * screen_size, value_support_size),
        )
        if value_activation == 'tanh':
            self.value_head.add_module('tanh', nn.Tanh())

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.resBlocks:
            x = block(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

# Creates initial hidden state based on observation | several observations
class RepresentationFunctionResNet(nn.Module):
    def __init__(self, num_resBlocks=16, hidden_planes=256):
        super().__init__()
        
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


### CHEAT MODEL ###
class MuZeroResNetCheat(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.predictionFunction = PredictionFunctionResNet(**self.args['predictionFunction'])

    def __repr__(self):
        return "modelCheat"

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        return observation

    def dynamics(self, hidden_state, action):
        if len(hidden_state.shape) == 4:
            device = hidden_state.device
            hidden_state = hidden_state.cpu().detach().numpy()
            for i in range(hidden_state.shape[0]):
                hidden_state[i], _ = self.dynamics(hidden_state[i], action[i])
            hidden_state = torch.from_numpy(hidden_state).to(device)
        else:
            row = action // 3
            col = action % 3
            if (hidden_state[1, row, col] == 1
                and np.max(np.sum(hidden_state[0], axis=0)) < 3 
                and np.max(np.sum(hidden_state[0], axis=1)) < 3
                and np.sum(np.diag(hidden_state[0])) < 3
                and np.sum(np.diag(np.fliplr(hidden_state[0]))) < 3
                and np.max(np.sum(hidden_state[2], axis=0)) < 3
                and np.max(np.sum(hidden_state[2], axis=1)) < 3
                and np.sum(np.diag(hidden_state[2])) < 3
                and np.sum(np.diag(np.fliplr(hidden_state[2]))) < 3
            ):
                hidden_state[1, row, col] = 0
                hidden_state[2, row, col] = 1
        return hidden_state, 0