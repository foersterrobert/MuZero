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
    def self_play(self, game_idx_group):
        self_play_games = [SelfPlayGame(self.game, game_idx_group * self.args['group_size'] + i) for i in range(self.args['group_size'])]
        player = 1

        while len(self_play_games) > 0:
            observations = np.stack([self_play_game.observation for self_play_game in self_play_games])
            encoded_observations = self.game.get_encoded_observation(observations)
            canonical_observations = self.game.get_canonical_state(encoded_observations, player).copy()
            
            hidden_state = torch.tensor(canonical_observations, dtype=torch.float32, device=self.device) 
            # hidden_state = self.muZero.represent(hidden_state) 

            action_probs, value = self.muZero.predict(hidden_state)
            action_probs = torch.softmax(action_probs, dim=1).cpu().numpy()

            for i, self_play_game in enumerate(self_play_games):
                self_play_game.root = Node(
                    canonical_observations[i],
                    self_play_game.reward,
                    0, self.muZero, self.args, self.game
                )

                my_action_probs = action_probs[i]
                my_action_probs = (1 - self.args['dirichlet_epsilon']) * my_action_probs + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
                my_action_probs *= self_play_game.valid_locations
                my_action_probs /= np.sum(my_action_probs)

                self_play_game.root.expand(my_action_probs)

            for simulation in range(self.args['num_mcts_runs']):
                for self_play_game in self_play_games:
                    node = self_play_game.root

                    while node.is_expandable():
                        node = node.select_child()

                    self_play_game.node = node

                hidden_states = np.stack([self_play_game.node.state for self_play_game in self_play_games])
                action_probs, value = self.muZero.predict(
                    torch.tensor(hidden_states, dtype=torch.float32, device=self.device)
                )
                action_probs = torch.softmax(action_probs, dim=1).cpu().numpy()
                value = value.cpu().numpy().squeeze(1)

                for i, self_play_game in enumerate(self_play_games):
                    my_value, my_action_probs = None, None
                    if self.args['cheatAvailableActions'] or self.args['cheatTerminalState']:
                        unencoded_state = self_play_game.node.state.copy()
                        unencoded_state = (
                            unencoded_state * np.array([-1, 0, 1]).repeat(9).reshape(3, 3, 3)
                        ).sum(axis=0)

                        if self.args['cheatTerminalState']:
                            is_terminal, my_value = self.game.check_terminal_and_value(unencoded_state, self_play_game.node.action_taken)
                            my_value = self.game.get_opponent_value(my_value)

                    if not self.args['cheatTerminalState'] or not is_terminal:
                        my_action_probs, my_value = action_probs[i], value[i]

                        if self.args['cheatAvailableActions']:
                            available_actions = self.game.get_valid_locations(unencoded_state)
                            my_action_probs *= available_actions
                            my_action_probs /= np.sum(my_action_probs)

                        self_play_game.node.expand(my_action_probs)
                    self_play_game.node.backpropagate(my_value)

            for i in range(len(self_play_games) - 1, -1, -1):
                self_play_game = self_play_games[i]
                action_probs = [0] * self.game.action_size
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
                    for hist_state, hist_action, hist_player, hist_action_probs, hist_reward, hist_terminal in self_play_game.game_memory:
                        self.replayBuffer.memory.append((
                            hist_state, 
                            hist_action, 
                            hist_action_probs, 
                            self_play_game.reward * ((-1) ** (hist_player != player)),  # value
                            hist_reward,
                            self_play_game.game_idx, 
                            hist_terminal, 
                        ))
                    if not self.args['cheatTerminalState']:
                        self.replayBuffer.memory.append((
                            self.game.get_canonical_state(self.game.get_encoded_observation(self_play_game.observation), self.game.get_opponent_player(player)).copy(),
                            None,
                            np.zeros(self.game.action_size, dtype=np.float32),
                            self.game.get_opponent_value(1) * self_play_game.reward, # also works for single player games
                            0,
                            self_play_game.game_idx,
                            self_play_game.is_terminal,
                        ))
                    del self_play_games[i]

            player = self.game.get_opponent_player(player)

    def train(self):
        random.shuffle(self.replayBuffer.trajectories)
        for batchIdx in range(0, len(self.replayBuffer) - 1, self.args['batch_size']): 
            policy_loss = 0
            value_loss = 0
            # reward_loss = 0

            observation, action, policy, value, reward = list(zip(*self.replayBuffer.trajectories[batchIdx:min(len(self.replayBuffer) -1, batchIdx + self.args['batch_size'])]))
            observation = np.stack(observation)

            state = torch.tensor(observation, dtype=torch.float32, device=self.device)
            policy = torch.tensor(np.stack(policy).swapaxes(0, 1), dtype=torch.float32, device=self.device)
            value = torch.tensor(np.array(value).swapaxes(0, 1).reshape(self.args['K'] + 1, -1, 1), dtype=torch.float32, device=self.device)

            if not self.args['cheatRepresentationFunction']:
                state = self.muZero.represent(state)
            out_policy, out_value = self.muZero.predict(state)

            policy_loss += F.cross_entropy(out_policy, policy[0]) 
            value_loss += F.mse_loss(out_value, value[0])

            if self.args['K'] > 0:
                action = np.array(action).swapaxes(0, 1)

                if self.args['cheatDynamicsFunction']:
                    observation, out_reward = self.muZero.dynamics(observation, action[0])

                else:
                    state, out_reward = self.muZero.dynamics(state, action[0])
                    observation = state.detach().cpu().numpy()
                
                # reward_loss += F.mse_loss(out_reward, reward[0])

                for k in range(1, self.args['K'] + 1):
                    observation = self.game.get_canonical_state(observation, -1).copy()
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device)

                    out_policy, out_value = self.muZero.predict(state)

                    policy_loss += F.cross_entropy(out_policy, policy[k])
                    value_loss += F.mse_loss(out_value, value[k])

                    if self.args['cheatDynamicsFunction']:
                        observation, out_reward = self.muZero.dynamics(observation, action[k])

                    else:
                        state, out_reward = self.muZero.dynamics(state, action[k])
                        observation = state.detach().cpu().numpy()

                    # reward_loss += F.mse_loss(out_reward, reward[k])

            loss = value_loss * self.args['value_loss_weight'] + policy_loss #+ reward_loss
            loss /= self.args['K'] + 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run(self):
        for iteration in range(self.args['num_iterations']):
            print(f"iteration: {iteration}")
            self.replayBuffer.empty()

            self.muZero.eval()
            for train_game_idx in trange(self.args['num_train_games'] // self.args['group_size'], desc="train_game"):
                self.self_play(train_game_idx + iteration * (self.args['num_train_games'] // self.args['group_size']))
            self.replayBuffer.build_trajectories()

            self.muZero.train()
            for epoch in trange(self.args['num_epochs'], desc="epochs"):
                self.train()

            torch.save(self.muZero.state_dict(), f"Models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.game}/optimizer_{iteration}.pt")

class SelfPlayGame:
    def __init__(self, game, game_idx):
        self.game_idx = game_idx
        self.game_memory = []
        self.observation, self.valid_locations, self.reward, self.is_terminal = game.get_initial_state()
        self.root = None
        self.node = None
