import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from mcts import MCTS
from replaybuffer import ReplayBuffer

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
        observation, valid_locations, immediate_reward = self.game.get_initial_state()

        while True:
            encoded_observation = self.game.get_encoded_observation(observation)
            encoded_observation = encoded_observation if player == 1 else encoded_observation[::-1].copy()
            root = self.mcts.search(encoded_observation, immediate_reward, valid_locations, player=1)

            action_probs = [0] * self.game.action_size
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            visit_counts = [child.visit_count for child in root.children]
            actions = [child.action_taken for child in root.children]
            if self.args['temperature'] == 0:
                action = actions[np.argmax(visit_counts)]
            elif self.args['temperature'] == float('inf'):
                action = np.random.choice(actions)
            else:
                visit_count_distribution = np.array(visit_counts) ** (1 / self.args['temperature'])
                visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
                action = np.random.choice(actions, p=visit_count_distribution)

            game_memory.append((root.state, action, player, action_probs))

            observation, valid_locations, immediate_reward = self.game.step(observation, action, player)

            is_terminal, reward = self.game.check_terminal_and_value(observation, action)
            if is_terminal:
                return_memory = []
                for hist_state, hist_action, hist_player, hist_action_probs in game_memory[:-1]:
                    return_memory.append((
                        hist_state, hist_action, hist_action_probs, reward * ((-1) ** (hist_player != player)), 0, game_idx
                    ))
                return_memory.append((
                    game_memory[-1][0], game_memory[-1][1], game_memory[-1][3], reward * ((-1) ** (game_memory[-1][2] != player)), reward * ((-1) ** (game_memory[-1][2] != player)), game_idx
                ))
                return return_memory

            player = player * -1

    def train(self):
        for training_step in range(self.args['num_training_steps']):
            policy_loss = 0
            value_loss = 0
            reward_loss = 0

            batch = self.replayBuffer.sample(self.args['batch_size'])
            for observation, actions, action_probs, values, rewards in batch:
                observation = torch.tensor(observation).to(self.device)
                hidden_state = self.muZero.represent(observation)
                predicted_action_probs, predicted_value = self.muZero.predict(hidden_state)

                policy_loss += -torch.sum(torch.log(predicted_action_probs) * action_probs[0])
                value_loss += F.mse_loss(predicted_value, values[0])

                for k in range(1, self.args['K']):
                    hidden_state, predicted_reward = self.muZero.dynamics(hidden_state, actions[k - 1])
                    predicted_action_probs, predicted_value = self.muZero.predict(hidden_state)

                    policy_loss += -torch.sum(torch.log(predicted_action_probs) * action_probs[k])
                    value_loss += F.mse_loss(predicted_value, values[k])
                    reward_loss += F.mse_loss(predicted_reward, rewards[k])

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
                self.replayBuffer.add(self.self_play(train_game_idx + iteration * self.args['num_train_games']))

            self.muZero.train()
            for epoch in trange(self.args['num_epochs'], desc="epoch"):
                self.train()

            torch.save(self.muZero.state_dict(), f"Models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.game}/optimizer_{iteration}.pt")
