import torch

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = 9
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        return 'TicTacToe'

    def get_initial_state(self):
        observation = torch.zeros((self.row_count, self.column_count), dtype=torch.int8, device=self.device)
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
            torch.sum(observation[row]) == mark * self.column_count # row
            or torch.sum(observation[:, column]) == mark * self.row_count # column 
            or torch.sum(torch.diag(observation)) == mark * self.row_count # diagonal 
            or torch.sum(torch.diag(torch.fliplr(observation))) == mark * self.row_count # flipped diagonal
        )

    def step(self, observation, action, player):
        row = action // self.column_count
        column = action % self.column_count
        observation[row][column] = player
        valid_locations = self.get_valid_locations(observation)
        is_terminal, reward = self.check_terminal_and_value(observation, action)
        return observation, valid_locations, reward, is_terminal

    def get_valid_locations(self, observation):
        return (observation.reshape(-1) == 0).int()

    def get_canonical_state(self, hidden_state, player):
        return hidden_state if player == 1 else hidden_state.flip(1)

    def get_encoded_observation(self, observation, parallel=False):
        if parallel:
            encoded_observation = torch.swapaxes(torch.stack(
                ((observation == -1), (observation == 0), (observation == 1))).float(), 0, 1
            )

        else:
            encoded_observation = torch.vstack((
                (observation == -1).reshape(1, self.row_count, self.column_count),
                (observation == 0).reshape(1, self.row_count, self.column_count),
                (observation == 1).reshape(1, self.row_count, self.column_count)
            )).float().unsqueeze(0)

        return encoded_observation

    def check_terminal_and_value(self, observation, action):
        if self.is_position_a_winner(observation, action):
            return (True, 1)
        if sum(self.get_valid_locations(observation)) == 0:
            return (True, 0)
        return (False, 0)
    
    def get_opponent_player(self, player):
        return player * -1

    def get_opponent_value(self, value):
        return value * -1
