import gymnasium as gym

class CartPole:
    def __init__(self, render=False):
        self.env = gym.make('CartPole-v1', render_mode='human' if render else 'rgb_array')
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
