import torch

class TicTacToe:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        return torch.zeros((self.row_count, self.column_count), device=self.device)

    def get_next_state(self, state, action, player):
        row, column = action // self.column_count, action % self.column_count
        state[row, column] = player
        return state

    def get_valid_actions(self, state):
        return (state.reshape(self.action_size) == 0).to(torch.float32)

    def check_win(self, state, action):
        row, column = action // self.column_count, action % self.column_count
        player = state[row, column].item()
        return (
            state[row, :].sum().item() == player * self.column_count
            or state[:, column].sum().item() == player * self.row_count
            or torch.diag(state).sum().item() == player * self.row_count
            or torch.diag(torch.flip(state, (0,))).sum().item() == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1.0, True
        if self.get_valid_actions(state).sum() == 0:
            return 0.0, True
        return 0.0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        return torch.stack((state == -1, state == 0, state == 1), dim=0).to(torch.float32)
