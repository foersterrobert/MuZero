import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import trange
from mcts import Node
from replaybuffer import ReplayBuffer

class Trainer:
    def __init__(self, muZero, optimizer, game, args):
        self.muZero = muZero
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.replayBuffer = ReplayBuffer(self.args, self.game)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def self_play(self, game_idx_group, group_size=100):
        self_play_games = [SelfPlayGame(self.game, game_idx_group * group_size + i) for i in range(group_size)]
        self_play_memory = []
        player = 1

        while len(self_play_games) > 0:
            del_list = []

            observations = np.stack([self_play_game.observation for self_play_game in self_play_games])
            encoded_observations = self.game.get_encoded_observation(observations, parallel=True)
            canonical_observations = self.game.get_canonical_state(encoded_observations, player).copy()
            hidden_state = torch.tensor(canonical_observations, dtype=torch.float32, device=self.device) # self.muZero.represent(canonical_observations) 

            action_probs, value = self.muZero.predict(hidden_state)
            action_probs = torch.softmax(action_probs, dim=1).cpu().numpy()
            hidden_state = hidden_state.cpu().numpy()

            for i, self_play_game in enumerate(self_play_games):
                self_play_game.root = Node(
                    hidden_state[i],
                    self_play_game.reward,
                    1, 0, self.muZero, self.args, self.game
                )
                self_play_game_action_probs = action_probs[i]
                self_play_game_action_probs *= self_play_game.valid_locations
                self_play_game_action_probs /= np.sum(self_play_game_action_probs)
                self_play_game.root.expand(self_play_game_action_probs)

            for simulation in range(self.args['num_mcts_runs']):
                for self_play_game in self_play_games:
                    node = self_play_game.root

                    while node.is_expandable():
                        node = node.select_child()

                    self_play_game.node = node

                hidden_states = np.stack([self_play_game.node.state for self_play_game in self_play_games])
                canonical_hidden_states = self.game.get_canonical_state(hidden_states, self_play_games[0].node.player).copy()
                canonical_hidden_states = torch.tensor(canonical_hidden_states, dtype=torch.float32, device=self.device)
                action_probs, value = self.muZero.predict(canonical_hidden_states)
                action_probs = torch.softmax(action_probs, dim=1).cpu().numpy()
                value = value.cpu().numpy().squeeze(1)

                for i, self_play_game in enumerate(self_play_games):
                    self_play_game.node.expand(action_probs[i])
                    self_play_game.node.backpropagate(value[i])

            for self_play_game in self_play_games:
                action_probs = np.zeros(self.game.action_size, dtype=np.float32)
                for child in self_play_game.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                # sample action from the mcts policy | based on temperature
                if self.args['temperature'] == 0:
                    action = np.argmax(action_probs)
                elif self.args['temperature'] == float('inf'):
                    action = np.random.choice([r for r in range(self.game.action_size) if action_probs[r] > 0])
                else:
                    temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                    temperature_action_probs /= np.sum(temperature_action_probs)
                    action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)

                self_play_game.game_memory.append((self_play_game.root.state, action, player, action_probs, self_play_game.reward, self_play_game.is_terminal))

                self_play_game.observation, self_play_game.valid_locations, self_play_game.reward, self_play_game.is_terminal = self.game.step(self_play_game.observation, action, player)

                if self_play_game.is_terminal:
                    return_memory = []
                    for hist_state, hist_action, hist_player, hist_action_probs, hist_reward, hist_terminal in self_play_game.game_memory:
                        return_memory.append((
                            hist_state, hist_action, hist_action_probs, self_play_game.reward * ((-1) ** (hist_player != player)), hist_reward, self_play_game.game_idx, hist_terminal
                        ))
                    return_memory.append((
                        self.game.get_canonical_state(self.game.get_encoded_observation(self_play_game.observation), self.game.get_opponent_player(player)),
                        0,
                        np.zeros(self.game.action_size),
                        -1 * self_play_game.reward,
                        0,
                        self_play_game.game_idx,
                        self_play_game.is_terminal
                    ))
                    self_play_memory.extend(return_memory)
                    del_list.append(self_play_game)

            for self_play_game in del_list:
                self_play_games.remove(self_play_game)

            player = self.game.get_opponent_player(player)
        
        return self_play_memory

    def train(self):
        random.shuffle(self.replayBuffer.trajectories)
        for batchIdx in range(0, len(self.replayBuffer), self.args['batch_size']): 
            policy_loss = 0
            value_loss = 0
            # reward_loss = 0

            state, action, policy, value, reward = list(zip(*self.replayBuffer.trajectories[batchIdx:min(len(self.replayBuffer) -1, batchIdx + self.args['batch_size'])]))
            state = torch.tensor(np.stack(state), dtype=torch.float32, device=self.device)
            policy = torch.tensor(np.stack(policy).swapaxes(0, 1), device=self.device)
            value = torch.tensor(np.array(value).swapaxes(0, 1).reshape(self.args['K'] + 1, -1, 1), dtype=torch.float32, device=self.device)
            action = np.array(action).swapaxes(0, 1)

            hidden_state = self.muZero.represent(state)
            out_policy, out_value = self.muZero.predict(hidden_state)

            policy_loss += F.cross_entropy(out_policy, policy[0]) 
            value_loss += F.mse_loss(out_value, value[0])

            for k in range(1, self.args['K'] + 1):
                hidden_state, out_reward = self.muZero.dynamics(hidden_state.clone().cpu().numpy(), action[k - 1], parallel=True)
                hidden_state = self.game.get_canonical_state(hidden_state, -1).copy()
                hidden_state = torch.tensor(hidden_state, dtype=torch.float32, device=self.device)
                out_policy, out_value = self.muZero.predict(hidden_state)

                policy_loss += F.cross_entropy(out_policy, policy[k])
                value_loss += F.mse_loss(out_value, value[k])
                # reward_loss += F.mse_loss(out_reward, reward[k])

        loss = value_loss * self.args['value_loss_weight'] + policy_loss #+ reward_loss
        loss = loss.mean()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        for iteration in range(self.args['num_iterations']):
            print(f"iteration: {iteration}")
            self.replayBuffer.empty()

            self.muZero.eval()
            for train_game_idx in trange(self.args['num_train_games'] // self.args['group_size'], desc="train_game"):
                game_memory = self.self_play(train_game_idx + iteration * self.args['num_train_games'] // self.args['group_size'], group_size=self.args['group_size'])
                self.replayBuffer.add(game_memory)
            self.replayBuffer.build_trajectories()

            self.muZero.train()
            for epoch in trange(self.args['num_epochs'], desc="epochs"):
                self.train()

            torch.save(self.muZero.state_dict(), f"Models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.game}/optimizer_{iteration}.pt")

class SelfPlayGame:
    def __init__(self, game, game_idx):
        self.game = game
        self.game_idx = game_idx
        self.game_memory = []
        self.observation, self.valid_locations, self.reward, self.is_terminal = self.game.get_initial_state()
        self.root = None
        self.node = None
