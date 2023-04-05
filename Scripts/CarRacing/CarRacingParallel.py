# %%
from PIL import Image
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import math
from tqdm import trange

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
class CarRacing:
    def __init__(self, sequence_lenth=3, skip_frames=3, clip_reward=False, time_out_tolerance=100, time_out=25, render=False):
        self.env = gym.make('CarRacing-v2', continuous=False, domain_randomize=False, render_mode='human' if render else 'rgb_array')
        self.action_size = self.env.action_space.n
        self.obs_stack = []
        self.sequence_lenth = sequence_lenth
        self.skip_frames = skip_frames
        self.clip_reward = clip_reward
        self.time_out_tolerance = time_out_tolerance
        self.time_out = time_out
        self.counter = 0
        self.negative_reward_counter = 0

    def __repr__(self):
        return "CarRacing"

    def get_initial_state(self):
        self.counter = 0
        self.negative_reward_counter = 0
        observation, _ = self.env.reset()
        action, reward, is_terminal = 0, 0, False 
        observation = self.get_encoded_observation(observation, action)
        self.obs_stack = [observation for _ in range(self.sequence_lenth)]
        return np.stack(self.obs_stack), reward, is_terminal

    def step(self, action):
        reward = 0
        for _ in range(self.skip_frames + 1):
            observation, r, is_terminal, _, _ = self.env.step(action)
            reward += r
            if is_terminal:
                break
        observation = self.get_encoded_observation(observation, action)
        self.obs_stack = self.obs_stack[1:] + [observation]
        if self.clip_reward:
            reward = np.clip(reward, -float('inf'), 1)
        if reward < 0 and self.counter > self.time_out_tolerance:
            self.negative_reward_counter += 1
            if self.negative_reward_counter >= self.time_out:
                is_terminal = True
                self.env.close()
        else:
            self.counter += 1
        return np.stack(self.obs_stack), reward, is_terminal

    def get_encoded_observation(self, observation, action):
        obs = Image.fromarray(observation.copy())
        obs = obs.crop((0, 0, 96, 84))
        obs = obs.convert('L')
        obs = np.array(obs)
        actionPlane = np.full((12, 96), action * (255 / self.action_size), dtype=np.float32)
        obs = np.concatenate((obs, actionPlane), axis=0)
        obs /= 255
        return obs

# %%
class ReplayBuffer:
    def __init__(self, args, game):
        self.memory = []
        self.trajectories = []
        self.args = args
        self.game = game

    def __len__(self):
        return len(self.memory)

    def empty(self):
        self.memory = []
        self.trajectories = []

    def build_trajectories(self):
        for i in range(len(self.memory)):
            observation, action, policy, reward, _, game_idx, is_terminal = self.memory[i]
            if is_terminal:
                action = np.random.choice(self.game.action_size)

            policy_list, action_list, value_list, reward_list = [policy], [action], [], [reward]

            # value bootstrap for N-step return
            # value starts at root value n steps ahead
            if i + self.args['N'] + 1 < len(self.memory) and self.memory[i + self.args['N'] + 1][5] == game_idx:
                value = self.memory[i + self.args['N'] + 1][4] * self.args['gamma'] ** self.args['N']
            else:
                value = 0
            # add discounted rewards until end of game or N steps
            for n in range(2, self.args['N'] + 2):
                if i + n < len(self.memory) and self.memory[i + n][5] == game_idx:
                    _, _, _, reward, _, _, _ = self.memory[i + n]
                    value += reward * self.args['gamma'] ** (n - 2)
                else:
                    break
            value_list.append(value)

            for k in range(1, self.args['K'] + 1):
                if i + k < len(self.memory) and self.memory[i + k][5] == game_idx:
                    _, action, policy, reward, _, _, is_terminal = self.memory[i + k]
                    if is_terminal:
                        action = np.random.choice(self.game.action_size)
                    action_list.append(action)
                    policy_list.append(policy)
                    reward_list.append(reward)

                    if i + k + self.args['N'] + 1 < len(self.memory) and self.memory[i + k + self.args['N'] + 1][5] == game_idx:
                        value = self.memory[i + k + self.args['N'] + 1][4] * self.args['gamma'] ** self.args['N']
                    else:
                        value = 0
                    for n in range(2, self.args['N'] + 2):
                        if i + k + n < len(self.memory) and self.memory[i + k + n][5] == game_idx:
                            _, _, _, reward, _, _, _ = self.memory[i + k + n]
                            value += reward * self.args['gamma'] ** (n - 2)
                        else:
                            break
                    value_list.append(value)

                else:
                    action_list.append(np.random.choice(self.game.action_size))
                    policy_list.append(policy_list[-1])
                    value_list.append(0)
                    reward_list.append(0)

            policy_list = np.stack(policy_list)
            self.trajectories.append((observation, action_list, policy_list, value_list, reward_list))


# %%
class MinMaxStats:
    def __init__(self, known_bounds):
        self.maximum = known_bounds['max'] if known_bounds else -float('inf')
        self.minimum = known_bounds['min'] if known_bounds else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, state, reward, prior, muZero, args, game, parent=None, action_taken=None, visit_count=0):
        self.state = state
        self.reward = reward
        self.children = []
        self.parent = parent
        self.total_value = 0
        self.visit_count = visit_count # Should start at 1 for root node
        self.prior = prior
        self.muZero = muZero
        self.action_taken = action_taken
        self.args = args
        self.game = game

    @torch.no_grad()
    def expand(self, action_probs):
        actions = [a for a in range(self.game.action_size) if action_probs[a] > 0]
        expand_state = self.state.copy()
        expand_state = np.expand_dims(expand_state, axis=0).repeat(len(actions), axis=0)

        expand_state, reward = self.muZero.dynamics(
            torch.tensor(expand_state, dtype=torch.float32, device=self.muZero.device), actions)
        expand_state = expand_state.cpu().numpy()
        reward = self.muZero.inverse_reward_transform(reward).cpu().numpy().flatten()
        
        for i, a in enumerate(actions):
            child = Node(
                expand_state[i],
                reward[i],
                action_probs[a],
                self.muZero,
                self.args,
                self.game,
                parent=self,
                action_taken=a,
            )
            self.children.append(child)

    def backpropagate(self, value, minMaxStats):
        self.total_value += value
        self.visit_count += 1
        minMaxStats.update(self.value())
        if self.parent is not None:
            value = self.reward + self.args['gamma'] * value
            self.parent.backpropagate(value, minMaxStats)

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def select_child(self, minMaxStats):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            ucb_score = self.get_ucb_score(child, minMaxStats)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def get_ucb_score(self, child, minMaxStats):
        pb_c = math.log((self.visit_count + self.args["pb_c_base"] + 1) /
                  self.args["pb_c_base"]) + self.args["pb_c_init"]
        pb_c *= math.sqrt(self.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = minMaxStats.normalize(child.reward + self.args['gamma'] * child.value())
        else:
            # value_score = 0
            value_score = minMaxStats.normalize(child.reward)
        return prior_score + value_score

class MCTS:
    def __init__(self, muZero, game, args):
        self.muZero = muZero
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self, observations, spGames):
        minMaxStats = MinMaxStats(self.args['known_bounds'])

        hidden_states = torch.tensor(observations, dtype=torch.float32, device=self.muZero.device)
        hidden_states = self.muZero.represent(hidden_states)

        policy, _ = self.muZero.predict(hidden_states)
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        hidden_states = hidden_states.cpu().numpy()

        for i, spg in enumerate(spGames):
            spg.root = Node(
                hidden_states[i], # cannonical if 2 players
                spg.reward,
                0, self.muZero, self.args, self.game,
                visit_count=1
            )
            spg.root.expand(policy[i])

        for search in range(self.args['num_mcts_searches']):
            for spg in spGames:
                node = spg.root

                while node.is_expanded():
                    node = node.select_child(minMaxStats)

                spg.node = node

            hidden_states = np.stack([spg.node.state for spg in spGames])
            policy, value = self.muZero.predict(
                torch.tensor(hidden_states, dtype=torch.float32, device=self.muZero.device)
            )
            policy = torch.softmax(policy, dim=1).cpu().numpy()
            value = self.muZero.inverse_value_transform(value).cpu().numpy().flatten()

            for i, spg in enumerate(spGames):
                spg.node.expand(policy[i])
                spg.node.backpropagate(value[i], minMaxStats)

# %%
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

# %%
class Trainer:
    def __init__(self, muZero, optimizer, game, args):
        self.muZero = muZero
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(self.muZero, self.game, self.args)
        self.replayBuffer = ReplayBuffer(self.args, self.game)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def self_play(self, game_idx_group, self_play_bar):
        spGames = [
            SelfPlayGame(
                CarRacing(), 
                game_idx_group * self.args['num_parallel_games'] + i
            ) for i in range(self.args['num_parallel_games'])
        ]

        while len(spGames) > 0:
            observations = np.stack([self_play_game.observation for self_play_game in spGames])
            self.mcts.search(observations, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.memory.append((spg.observation, action, action_probs, spg.root.total_value / spg.root.visit_count, spg.reward, spg.is_terminal))
                
                spg.observation, spg.reward, spg.is_terminal = spg.game.step(action)

                if spg.is_terminal:
                    for hist_state, hist_action, hist_action_probs, hist_root_value, hist_reward, hist_terminal in spg.memory:
                        self.replayBuffer.memory.append((
                            hist_state,
                            hist_action, 
                            hist_action_probs,
                            hist_reward,
                            hist_root_value,
                            spg.game_idx,
                            hist_terminal
                        ))
                    if not self.args['K'] > 0:
                        self.replayBuffer.memory.append((
                            spg.observation,
                            None,
                            np.array([1 / self.game.action_size] * self.game.action_size),
                            0,
                            0,
                            spg.game_idx,
                            spg.is_terminal
                        ))
                    del spGames[i]
                    self_play_bar.set_description(
                        f"Games finished: {self.args['num_parallel_games'] - len(spGames) + self.args['num_parallel_games'] * game_idx_group} | Avg. steps: \
                        {len(self.replayBuffer) / (self.args['num_parallel_games'] - len(spGames) + self.args['num_parallel_games'] * (game_idx_group % (self.args['num_train_games'] // self.args['num_parallel_games'])))}"
                    )

    def train(self):
        random.shuffle(self.replayBuffer.trajectories)
        for batchIdx in range(0, len(self.replayBuffer) - 1, self.args['batch_size']): 
            state, action, policy, value, reward = list(zip(*self.replayBuffer.trajectories[batchIdx:min(len(self.replayBuffer) -1, batchIdx + self.args['batch_size'])]))

            state = torch.tensor(np.stack(state), dtype=torch.float32, device=self.device)
            action = np.array(action)
            policy = torch.tensor(np.stack(policy), dtype=torch.float32, device=self.device)
            value = torch.tensor(np.array(value), dtype=torch.float32, device=self.device)
            reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device)

            transformed_reward = self.muZero.scalar_transform(reward)
            phi_reward = self.muZero.reward_phi(transformed_reward)
            transformed_value = self.muZero.scalar_transform(value)
            phi_value = self.muZero.value_phi(transformed_value)

            state = self.muZero.represent(state)
            out_policy, out_value = self.muZero.predict(state)

            policy_loss = F.cross_entropy(out_policy, policy[:, 0]) 
            value_loss = self.scalar_value_loss(out_value, phi_value[:, 0])
            reward_loss = torch.zeros(value_loss.shape, device=self.device)

            if self.args['K'] > 0:
                for k in range(1, self.args['K'] + 1):
                    state, out_reward = self.muZero.dynamics(state, action[:, k - 1])
                    reward_loss += self.scalar_reward_loss(out_reward, phi_reward[:, k])
                    state.register_hook(lambda grad: grad * 0.5)

                    out_policy, out_value = self.muZero.predict(state)

                    policy_loss += F.cross_entropy(out_policy, policy[:, k])
                    value_loss += self.scalar_value_loss(out_value, phi_value[:, k])

            loss = (value_loss * self.args['value_loss_weight'] + policy_loss + reward_loss).mean()
            loss.register_hook(lambda grad: grad * 1 / self.args['K'])

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.muZero.parameters(), self.args['max_grad_norm'])
            self.optimizer.step()

    def run(self):
        for iteration in range(self.args['num_iterations']):
            print(f"iteration: {iteration}")
            self.replayBuffer.empty()

            self.muZero.eval()
            for train_game_idx in (self_play_bar := trange(self.args['num_train_games'] // self.args['num_parallel_games'], desc="train_game")):
                self.self_play(train_game_idx + iteration * (self.args['num_train_games'] // self.args['num_parallel_games']), self_play_bar)
            self.replayBuffer.build_trajectories()

            self.muZero.train()
            for epoch in trange(self.args['num_epochs'], desc="epochs"):
                self.train()

            torch.save(self.muZero.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

class SelfPlayGame:
    def __init__(self, game, game_idx):
        self.game = game
        self.game_idx = game_idx
        self.memory = []
        self.observation, self.reward, self.is_terminal = self.game.get_initial_state()
        self.root = None
        self.node = None

# %%
args = {
    'num_iterations': 20,
    'num_train_games': 400,
    'num_parallel_games': 100,
    'num_mcts_searches': 50,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1,
    'K': 5,
    'pb_c_base': 19625,
    'pb_c_init': 1.75,
    'N': 10,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25,
    'gamma': 0.997,
    'value_loss_weight': 0.25,
    'max_grad_norm': 5,
    'known_bounds': {} #{'min': 0, 'max': 1},
}

LOAD = True

game = CarRacing(clip_reward=True)
muZero = MuZero(game, device).to(device)
optimizer = torch.optim.Adam(muZero.parameters(), lr=0.001)

if LOAD:
    muZero.load_state_dict(torch.load(f"model.pt"))
    optimizer.load_state_dict(torch.load(f"optimizer.pt"))

trainer = Trainer(muZero, optimizer, game, args)
trainer.run()
