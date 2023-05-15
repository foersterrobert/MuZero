import torch
from ..baseConfig import MuZeroConfigBasic
from .model import MuZeroResNet
from .game import TicTacToe

class MuZeroConfigTicTacToe(MuZeroConfigBasic):
    def __init__(
        self,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_iterations=100,
        num_train_games=100,
        num_parallel_games=100,
        num_mcts_searches=60,
        num_epochs=3,
        batch_size=64,
        temperature=1,
        K=3,
        N=None,
        c_init=1.25,
        c_base=19652,
        discount=None,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.1,
        value_loss_weight=0.5,
        max_grad_norm=5,
        value_support=None,
        reward_support=None,
    ):
        super().__init__(
            device=device,
            num_iterations=num_iterations,
            num_train_games=num_train_games,
            num_parallel_games=num_parallel_games,
            num_mcts_searches=num_mcts_searches,
            num_epochs=num_epochs,
            batch_size=batch_size,
            temperature=temperature,
            K=K,
            N=N,
            c_init=c_init,
            c_base=c_base,
            discount=discount,
            dirichlet_epsilon=dirichlet_epsilon,
            dirichlet_alpha=dirichlet_alpha,
            value_loss_weight=value_loss_weight,
            max_grad_norm=max_grad_norm,
            value_support=value_support,
            reward_support=reward_support,
        )

        self.game = TicTacToe()

        self.model = MuZeroResNet({
            'predictionFunction': {
                'num_resBlocks': 2,
                'num_hidden': 16,
            },
            'dynamicsFunction': {
                'num_resBlocks': 2,
                'num_hidden': 16,
            },
            'representationFunction': {
                'num_resBlocks': 1,
                'num_hidden': 16,
            },
        }, self.game).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def __repr__(self):
        return 'TicTacToe'
