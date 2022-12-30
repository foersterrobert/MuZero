import torch
from ..baseConfig import MuZeroConfigBasic, DiscreteSupport
from .model import MuZeroLinearNet
from .game import CartPole

class MuZeroConfigCartpole(MuZeroConfigBasic):
    def __init__(self):
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
            value_support=DiscreteSupport(-20, 20),
            reward_support=DiscreteSupport(-5, 5)
        )
        self.game = CartPole()
        self.model = MuZeroLinearNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

