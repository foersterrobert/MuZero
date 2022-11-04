import torch
import torch.nn.functional as F
from tqdm import trange
from mcts import MCTS
from replaybuffer import ReplayBuffer
import numpy as np

class Trainer:
    def __init__(self, muZero, optimizer, game, args):
        self.muZero = muZero
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(self.muZero, self.game, self.args)
        self.replayBuffer = ReplayBuffer(self.args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def self_play(self, game_idx):
        game_memory = []
        player = 1
        observation, valid_locations, reward = self.game.get_initial_state()

        while True:
            encoded_observation = self.game.get_encoded_observation(observation)
            canonical_observation = self.game.get_canonical_state(encoded_observation, player)
            root = self.mcts.search(canonical_observation, reward, valid_locations, player=1)

            action_probs = torch.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= torch.sum(action_probs)

            # sample action from the mcts policy | based on temperature
            if self.args['temperature'] == 0:
                action = torch.argmax(action_probs).item()
            elif self.args['temperature'] == float('inf'):
                action = np.random.choice([r for r in range(self.game.action_size) if action_probs[r] > 0])
            else:
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= torch.sum(temperature_action_probs)
                action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs.detach().numpy())

            game_memory.append((root.state, action, player, action_probs, reward))

            observation, valid_locations, reward, is_terminal = self.game.step(observation, action, player)

            if is_terminal:
                return_memory = []
                for hist_state, hist_action, hist_player, hist_action_probs, hist_reward in game_memory:
                    return_memory.append((
                        hist_state, hist_action, hist_action_probs, reward * ((-1) ** (hist_player != player)), hist_reward, game_idx
                    ))
                return return_memory

            player = self.game.get_opponent_player(player)

    def train(self):
        policy_loss = 0
        value_loss = 0
        reward_loss = 0

        batch = self.replayBuffer.sample(self.args['batch_size'])
        for observation, actions, action_probs, values, rewards in batch:
            values = torch.tensor(values, dtype=torch.float).to(self.device).reshape(-1, 1)
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.device).reshape(-1, 1)

            hidden_state = self.muZero.represent(observation)
            predicted_action_probs, predicted_value = self.muZero.predict(hidden_state)

            policy_loss += -torch.sum(torch.log(predicted_action_probs) * action_probs[0])
            value_loss += F.mse_loss(predicted_value[0], values[0])

            for k in range(1, self.args['K'] + 1):
                hidden_state = self.game.get_canonical_state(hidden_state, -1)
                hidden_state, predicted_reward = self.muZero.dynamics(hidden_state, actions[k - 1])
                predicted_action_probs, predicted_value = self.muZero.predict(hidden_state)

                policy_loss += -torch.sum(torch.log(predicted_action_probs) * action_probs[k])
                value_loss += F.mse_loss(predicted_value[0], values[k])
                reward_loss += F.mse_loss(predicted_reward[0], rewards[k])

        loss = value_loss * self.args['value_loss_weight'] + policy_loss + reward_loss
        loss = loss.mean()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def run(self):
        for iteration in range(self.args['num_iterations']):
            print(f"iteration: {iteration}")
            self.replayBuffer.empty()

            self.muZero.eval()
            for train_game_idx in trange(self.args['num_train_games'], desc="train_game"):
                self.replayBuffer.add(
                    self.self_play(train_game_idx + iteration * self.args['num_train_games'])
                )

            self.muZero.train()
            for training_step in trange(self.args['num_training_steps'], desc="training_step"):
                self.train()

            torch.save(self.muZero.state_dict(), f"Models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.game}/optimizer_{iteration}.pt")
