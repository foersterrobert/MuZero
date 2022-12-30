import torch
import numpy as np
from .. import MuZeroConfigBasic
from .model import MuZeroResNet
from .game import TicTacToe

class MuZeroConfigTicTacToe(MuZeroConfigBasic):
    def __init__(
        self,
        cheatAvailableActions=False,
        cheatTerminalState=False,
        cheatDynamicsFunction=False,
        cheatRepresentationFunction=False,
    ):
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
        self.game = TicTacToe()
        self.model = MuZeroResNet({
            'predictionFunction': {
                'num_resBlocks': 4,
                'hidden_planes': 128
            },
            'dynamicsFunction': {
                'num_resBlocks': 4,
                'hidden_planes': 128
            },
            'representationFunction': {
                'num_resBlocks': 3,
                'hidden_planes': 64
            },
            'cheatAvailableActions': cheatAvailableActions,
            'cheatTerminalState': cheatTerminalState,
            'cheatDynamicsFunction': cheatDynamicsFunction,
            'cheatRepresentationFunction': cheatRepresentationFunction,
        })
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

