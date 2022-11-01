import numpy as np
import random
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
        self.mcts = MCTS(self.muZero, self.args)
        self.replayBuffer = ReplayBuffer(self.args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def self_play(self, game_idx):
        game_memory = []
        player = 1
        observation, valid_locations = self.game.get_initial_state()

        while True:
            encoded_observation = self.game.get_encoded_observation(observation)
            canonical_observation = self.game.get_canonical_state(encoded_observation, player)
            root = self.mcts.search(canonical_observation, valid_locations, player=1)

            action_probs = [0] * self.game.action_size
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            game_memory.append((root.state, player, action_probs))

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

            observation, valid_locations = self.game.step(observation, action, player)

            is_terminal, reward = self.game.check_terminal_and_value(observation, action)
            if is_terminal:
                return_memory = []
                for hist_state, hist_player, hist_action_probs in game_memory[:-1]:
                    return_memory.append((
                        hist_state, hist_action_probs, reward * ((-1) ** (hist_player != player)), 0, game_idx
                    ))
                return_memory.append((
                    game_memory[-1][0], game_memory[-1][2], reward * ((-1) ** (game_memory[-1][1] != player)), reward * ((-1) ** (game_memory[-1][1] != player)), game_idx
                ))
                return return_memory

            player = self.game.get_opponent_player(player)

    def train(self):
        batch = self.replayBuffer.sample(self.args['batch_size'])
        
    def run(self):
        for iteration in range(self.args['num_iterations']):
            print(f"iteration: {iteration}")
            self.replayBuffer.empty()

            self.model.eval()
            for train_game_idx in trange(self.args['num_train_games'], desc="train_game"):
                self.replayBuffer.add(self.self_play(train_game_idx + iteration * self.args['num_train_games']))

            self.model.train()
            for epoch in trange(self.args['num_epochs'], desc="epoch"):
                self.train()

            torch.save(self.muZero.state_dict(), f"Models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.game}/optimizer_{iteration}.pt")
