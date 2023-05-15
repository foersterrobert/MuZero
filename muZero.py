import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import trange
from mcts import MCTS
from replaybuffer import ReplayBuffer

class MuZero:
    def __init__(self, config):
        self.config = config
        self.model = config.model
        self.optimizer = config.optimizer
        self.game = config.game
        self.mcts = MCTS(self.model, self.game, self.config)
        self.replayBuffer = ReplayBuffer(self.config, self.game)

    def self_play(self, game_idx):
        memory = []
        player = 1
        observation, valid_moves, reward, is_terminal = self.game.get_initial_state()

        while True:
            neutral_observation = self.game.change_perspective(observation, player).copy()
            encoded_observation = self.game.get_encoded_observation(neutral_observation)
            action_probs, root_value = self.mcts.search(encoded_observation, valid_moves)

            temperature_action_probs = action_probs ** (1 / self.config.temperature)
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            memory.append((encoded_observation, action, action_probs, root_value, reward, player))

            observation, valid_moves, reward, is_terminal = self.game.step(observation, action, player)

            if is_terminal:
                return_memory = []
                for hist_observation, hist_action, hist_action_probs, hist_root_value, hist_reward, hist_player in memory:
                    hist_outcome = reward if hist_player == player else self.game.get_opponent_value(reward)
                    return_memory.append((
                        hist_observation,
                        hist_action, 
                        hist_action_probs,
                        hist_outcome,
                        hist_reward,
                        hist_root_value,
                        game_idx,
                        False
                    ))
                hist_outcome = reward if self.game.get_opponent(player) == player else self.game.get_opponent_value(reward)
                return_memory.append((
                    self.game.get_encoded_observation(self.game.change_perspective(observation, self.game.get_opponent(player)).copy()),
                    None,
                    np.zeros(self.game.action_size, dtype=np.float32),
                    hist_outcome,
                    0,
                    0,
                    game_idx,
                    True
                ))
                return return_memory

            player = self.game.get_opponent_player(player)

    def train(self):
        random.shuffle(self.replayBuffer.trajectories)
        for batchIdx in range(0, len(self.replayBuffer), self.config.batch_size): 
            sample = self.replayBuffer.trajectories[batchIdx:batchIdx+self.config.batch_size]
            observation, policy_targets, action, value_targets, reward = list(zip(*sample))
            
            observation = np.tensor(np.array(observation), dtype=torch.float32, device=self.config.device)
            action = np.array(action)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.config.device)
            value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32, device=self.config.device).unsqueeze(-1)

            hidden_state = self.model.represent(observation)
            out_policy, out_value = self.model.predict(hidden_state)

            predictions = [(out_policy, out_value)]
            for k in range(1, self.config.K + 1):
                hidden_state, out_reward = self.model.dynamics(hidden_state, action[:, k - 1])
                out_policy, out_value = self.model.predict(hidden_state)
                predictions.append((out_policy, out_value, out_reward))

                hidden_state.register_hook(lambda grad: grad * 0.5)

            policy_loss = F.cross_entropy(predictions[0][0], policy_targets[:, 0])
            value_loss = F.mse_loss(predictions[0][1], value_targets[:, 0])
            reward_loss = 0
            for k in range(1, self.config.K + 1):
                current_policy_loss = F.cross_entropy(predictions[k][0], policy_targets[:, k], reduction="sum") \
                    / (policy_targets[:, k].sum(axis=1)!=0).sum()
                current_value_loss = F.mse_loss(predictions[k][1], value_targets[:, k])
                current_reward_loss = F.mse_loss(predictions[k][2], reward[:, k - 1])
                
                current_policy_loss.register_hook(lambda grad: grad / self.config.K)
                current_value_loss.register_hook(lambda grad: grad / self.config.K)
                current_reward_loss.register_hook(lambda grad: grad / self.config.K)

                policy_loss += current_policy_loss
                value_loss += current_value_loss
                reward_loss += current_reward_loss
            loss = value_loss * self.config.value_loss_weight + policy_loss + reward_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
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

            torch.save(self.model.state_dict(), f"Environments/{self.config}/Models/{self.model}_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Environments/{self.config}/Models/{self.model}_optimizer_{iteration}.pt")
