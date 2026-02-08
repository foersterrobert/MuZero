import torch

class TicTacToe:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        self.sequence_length = 3

    def get_initial_state(self):
        return torch.zeros((self.row_count, self.column_count), device=self.device)

    def get_next_state(self, state, action, player):
        action = action.item() if isinstance(action, torch.Tensor) and action.dim() == 0 else int(action)
        row, column = action // self.column_count, action % self.column_count
        if state.dim() == 2:
            state[row, column] = player
        else:
            state[:, row, column] = player
        return state

    def get_valid_moves(self, state):
        flat = state.reshape(-1, 9)
        return (flat == 0).to(torch.float32)

    def check_win(self, state, action):
        action = action.item() if isinstance(action, torch.Tensor) and action.dim() == 0 else int(action)
        row, column = action // self.column_count, action % self.column_count
        if state.dim() == 2:
            player = state[row, column].item()
            return (
                state[row, :].sum().item() == player * self.column_count
                or state[:, column].sum().item() == player * self.row_count
                or torch.diag(state).sum().item() == player * self.row_count
                or torch.diag(torch.flip(state, (0,))).sum().item() == player * self.row_count
            )
        else:
            player = state[:, row, column]
            row_win = (state[:, row, :].sum(dim=1) == player * self.column_count).any().item()
            col_win = (state[:, :, column].sum(dim=1) == player * self.row_count).any().item()
            diag1 = (torch.diagonal(state, dim1=1, dim2=2).sum(dim=1) == player * self.row_count).any().item()
            diag2 = (torch.diagonal(state.flip(1), dim1=1, dim2=2).sum(dim=1) == player * self.row_count).any().item()
            return row_win or col_win or diag1 or diag2

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1.0, True
        if self.get_valid_moves(state).sum() == 0:
            return 0.0, True
        return 0.0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        # (state == -1), (state == 0), (state == 1) -> channels first then batch
        encoded = torch.stack((state == -1, state == 0, state == 1), dim=0).to(torch.float32)
        if state.dim() == 3:
            encoded = encoded.permute(1, 0, 2, 3)  # (B, 3, 3, 3)
        return encoded
