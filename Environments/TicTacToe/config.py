import torch
from ..baseConfig import MuZeroConfigBasic
from .model import MuZeroResNet, MuZeroResNetCheat
from .game import TicTacToe

class MuZeroConfigTicTacToe(MuZeroConfigBasic):
    def __init__(
        self,
        cheatAvailableActions=False,
        cheatTerminalState=False,
        cheatModel=False,
    ):
        super().__init__(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_iterations=20,
            num_train_games=500,
            group_size=100,
            num_mcts_runs=60,
            num_epochs=4,
            batch_size=64,
            temperature=1,
            K=3,
            N=None,
            c_init=2,
            c_base=19625,
            gamma=1,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            value_loss_weight=1,
            value_support=None,
            reward_support=None,
        )

        self.cheatAvailableActions = cheatAvailableActions
        self.cheatTerminalState = cheatTerminalState

        self.game = TicTacToe()

        if cheatModel:
            self.model = MuZeroResNetCheat({
                'predictionFunction': {
                    'num_resBlocks': 4,
                    'hidden_planes': 128,
                    'screen_size': 9,
                    'action_size': 9,
                    'value_support_size': 1,
                    'value_activation': 'tanh'
                },
            }).to(self.device)

        else:
            self.model = MuZeroResNet({
                'predictionFunction': {
                    'num_resBlocks': 3,
                    'hidden_planes': 64,
                    'screen_size': 9,
                    'action_size': 9,
                    'value_support_size': 1,
                    'value_activation': 'tanh'
                },
                'dynamicsFunction': {
                    'num_resBlocks': 4,
                    'hidden_planes': 64,
                    'predict_reward': False, 
                    'reward_support_size': 1
                },
                'representationFunction': {
                    'num_resBlocks': 3,
                    'hidden_planes': 32
                },
            }).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def __repr__(self):
        return 'TicTacToe'
