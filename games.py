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
        return observation, valid_locations, 0

    def is_position_a_winner(self, observation, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        mark = observation[row][column]
        
        return (
            sum(observation[row]) == mark * self.column_count # row
            or sum(observation[:, column]) == mark * self.row_count # column 
            or sum(np.diag(observation)) == mark * self.row_count # diagonal 
            or sum(np.diag(np.fliplr(observation))) == mark * self.row_count # flipped diagonal
        )

    def step(self, observation, action, player):
        row = action // self.column_count
        column = action % self.column_count
        observation[row][column] = player
        valid_locations = self.get_valid_locations(observation)
        return observation, valid_locations, 0

    def get_valid_locations(self, observation):
        return (observation.reshape(-1) == 0).astype(np.uint8)

    def get_canonical_state(self, hidden_state, player):
        return hidden_state if player == 1 else hidden_state.flip(2)

    def get_encoded_observation(self, observation):
        encoded_observation = np.vstack((
            (observation == -1).reshape(1, self.row_count, self.column_count),
            (observation == 0).reshape(1, self.row_count, self.column_count),
            (observation == 1).reshape(1, self.row_count, self.column_count)
        )).astype(np.float32)
        return encoded_observation

    def check_terminal_and_value(self, observation, action):
        if self.is_position_a_winner(observation, action):
            return (True, 1)
        if sum(self.get_valid_locations(observation)) == 0:
            return (True, 0)
        return (False, 0)
