import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import trange
from mcts import MCTS
from replaybuffer import ReplayBuffer

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = config.model
        self.optimizer = config.optimizer
        self.game = config.game
        self.mcts = MCTS(self.model, self.game, self.config)
        self.replayBuffer = ReplayBuffer(self.config, self.game)

    def self_play(self, game_idx):
        game_memory = []
        player = 1
        observation, valid_locations, reward, is_terminal = self.game.get_initial_state()

        while True:
            encoded_observation = self.game.get_encoded_observation(observation)
            canonical_observation = self.game.get_canonical_state(encoded_observation, player).copy()
            root = self.mcts.search(canonical_observation, reward, valid_locations)

            action_probs = [0] * self.game.action_size
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            # sample action from the mcts policy | based on temperature
            if self.config.temperature == 0:
                action = np.argmax(action_probs)
            elif self.config.temperature == float('inf'):
                action = np.random.choice([r for r in range(self.game.action_size) if action_probs[r] > 0])
            else:
                temperature_action_probs = action_probs ** (1 / self.config.temperature)
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)

            game_memory.append((canonical_observation, action, player, action_probs, reward, is_terminal))

            observation, valid_locations, reward, is_terminal = self.game.step(observation, action, player)

            if is_terminal:
                return_memory = []
                for hist_state, hist_action, hist_player, hist_action_probs, hist_reward, hist_terminal in game_memory:
                    return_memory.append((
                        hist_state,
                        hist_action, 
                        hist_action_probs,
                        reward * ((-1) ** (hist_player != player)),
                        hist_reward,
                        game_idx,
                        hist_terminal
                    ))
                if not self.config.K > 0:
                    return_memory.append((
                        self.game.get_canonical_state(self.game.get_encoded_observation(observation), self.game.get_opponent_player(player)).copy(),
                        None,
                        np.zeros(self.game.action_size, dtype=np.float32),
                        self.game.get_opponent_value(1) * reward, # also works for single player games
                        0,
                        game_idx,
                        is_terminal
                    ))
                return return_memory

            player = self.game.get_opponent_player(player)

    def train(self):
        random.shuffle(self.replayBuffer.trajectories)
        for batchIdx in range(0, len(self.replayBuffer) - 1, self.config.batch_size): 
            policy_loss = 0
            value_loss = 0
            # reward_loss = 0

            observation, action, policy, value, reward = list(zip(*self.replayBuffer.trajectories[batchIdx:min(len(self.replayBuffer) -1, batchIdx + self.config.batch_size)]))
            observation = np.stack(observation)

            state = torch.tensor(observation, dtype=torch.float32, device=self.config.device)
            action = np.array(action)
            policy = torch.tensor(np.stack(policy), dtype=torch.float32, device=self.config.device)
            value = torch.tensor(np.expand_dims(np.array(value), axis=-1), dtype=torch.float32, device=self.config.device)

            if not self.config.cheatRepresentationFunction:
                state = self.muZero.represent(state)
            out_policy, out_value = self.muZero.predict(state)

            policy_loss += F.cross_entropy(out_policy, policy[:, 0]) 
            value_loss += F.mse_loss(out_value, value[:, 0])

            if self.config.K > 0:
                for k in range(1, self.config.K + 1):
                    if self.config.cheatDynamicsFunction:
                        observation, out_reward = self.muZero.dynamics(observation, action[:, k - 1])
                    else:
                        state, out_reward = self.muZero.dynamics(state, action[:, k - 1])
                        observation = state.detach().cpu().numpy()
                
                    # reward_loss += F.mse_loss(out_reward, reward[k])

                    observation = self.game.get_canonical_state(observation, -1).copy()
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device)

                    out_policy, out_value = self.muZero.predict(state)

                    policy_loss += F.cross_entropy(out_policy, policy[:, k])
                    value_loss += F.mse_loss(out_value, value[:, k])

            loss = value_loss * self.config.value_loss_weight + policy_loss #+ reward_loss
            loss /= self.config.K + 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run(self):
        for iteration in range(self.config.num_iterations):
            print(f"iteration: {iteration}")
            self.replayBuffer.empty()

            self.model.eval()
            for train_game_idx in trange(self.config.num_train_games, desc="train_game"):
                self.replayBuffer.memory += self.self_play(train_game_idx + iteration * self.config.num_train_games)
            self.replayBuffer.build_trajectories()

            self.model.train()
            for epoch in trange(self.config.num_epochs, desc="epochs"):
                self.train()

            torch.save(self.model.state_dict(), f"Environment/{self.config}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Weights/{self.config}/optimizer_{iteration}.pt")
