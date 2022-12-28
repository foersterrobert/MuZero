import torch
import gymnasium as gym
from . import MuZeroConfigBasic, DiscreteSupport
from ..Models import MuZeroLinearNetwork

class MuZeroConfigCartpole(MuZeroConfigBasic):
    def __init__(self):
        super().__init__(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_iterations=100,
            num_train_games=100,
            group_size=100,
            num_mcts_runs=100,
            num_epochs=100,
            batch_size=100,
            temperature=100,
            K=100,
            N=100,
            c_init=100,
            c_base=100,
            gamma=0.997,
            value_support=None,
            reward_support=None,
        )
        self.game = CartPole()
        self.model = MuZeroLinearNetwork()

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.action_size = self.env.action_space.n

    def __repr__(self):
        return 'CartPole-v1'

    def get_initial_state(self):
        observation, info = self.env.reset()
        valid_locations = self.action_size
        reward = 0
        is_terminal = False
        return observation, valid_locations, reward, is_terminal

    def step(self, action):
        observation, reward, is_terminal, _, _ = self.env.step(action)
        valid_locations = self.action_size
        if is_terminal:
            reward = 0
        return observation, valid_locations, reward, is_terminal

    def get_canonical_state(self, hidden_state, player):
        return hidden_state

    def get_encoded_observation(self, observation):
        return observation.copy()

    def get_opponent_player(self, player):
        return player

    def get_opponent_value(self, value):
        return value

