import numpy as np

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = 9

    def __repr__(self):
        return 'TicTacToe'

    def get_initial_state(self):
        observation = np.zeros((self.row_count, self.column_count), dtype=np.int8)
        valid_locations = self.get_valid_locations(observation)
        reward = 0
        terminal = False
        return observation, valid_locations, reward, terminal

    def is_position_a_winner(self, observation, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        mark = observation[row][column]
        
        return (
            np.sum(observation[row]) == mark * self.column_count # row
            or np.sum(observation[:, column]) == mark * self.row_count # column 
            or np.sum(np.diag(observation)) == mark * self.row_count # diagonal 
            or np.sum(np.diag(np.fliplr(observation))) == mark * self.row_count # flipped diagonal
        )

    def step(self, observation, action, player):
        row = action // self.column_count
        column = action % self.column_count
        observation[row][column] = player
        valid_locations = self.get_valid_locations(observation)
        is_terminal, reward = self.check_terminal_and_value(observation, action)
        return observation, valid_locations, reward, is_terminal

    def get_valid_locations(self, observation):
        return (observation.reshape(-1) == 0).astype(np.uint8)

    def get_canonical_state(self, hidden_state, player):
        if type(player) == list:
            for i in range(hidden_state.shape[0]):
                hidden_state[i] = self.get_canonical_state(hidden_state[i], player[i])
            return hidden_state

        return hidden_state if player == 1 else np.flip(hidden_state, axis=int(len(hidden_state.shape) == 4))

    def get_encoded_observation(self, observation, parallel=False):
        if parallel:
            encoded_observation = np.swapaxes(np.stack(
                ((observation == -1), (observation == 0), (observation == 1))), 0, 1
            ).astype(np.float32)

        else:
            encoded_observation = np.stack((
                (observation == -1),
                (observation == 0),
                (observation == 1)
            )).astype(np.float32)

        return encoded_observation

    def check_terminal_and_value(self, observation, action):
        if self.is_position_a_winner(observation, action):
            return (True, 1)
        if np.sum(self.get_valid_locations(observation)) == 0:
            return (True, 0)
        return (False, 0)
    
    def get_opponent_player(self, player):
        return player * -1

    def get_opponent_value(self, value):
        return value * -1
