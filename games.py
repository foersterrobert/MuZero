import numpy as np

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = 9

    def __repr__(self):
        return 'TicTacToe'

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=np.int8)

    def is_position_a_winner(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        mark = state[row][column]
        
        return (
            sum(state[row]) == mark * self.column_count # row
            or sum(state[:, column]) == mark * self.row_count # column 
            or sum(np.diag(state)) == mark * self.row_count # diagonal 
            or sum(np.diag(np.fliplr(state))) == mark * self.row_count # flipped diagonal
        )

    def drop_piece(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row][column] = player
        return state

    def get_valid_locations(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def get_canonical_state(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.vstack((
            (state == -1).reshape(1, self.row_count, self.column_count),
            (state == 0).reshape(1, self.row_count, self.column_count),
            (state == 1).reshape(1, self.row_count, self.column_count)
        )).astype(np.float32)
        return encoded_state

    def get_opponent_value(self, score):
        return -1*score

    def get_opponent_player(self, player):
        return -1*player

    def check_terminal_and_value(self, state, action):
        if self.is_position_a_winner(state, action):
            return (True, 1)
        if sum(self.get_valid_locations(state)) == 0:
            return (True, 0)
        return (False, 0)
