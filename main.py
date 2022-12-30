import torch
from torch.optim import Adam
import numpy as np
import random
# from trainer import Trainer
from trainerParallel import Trainer

# In training: scale hidden state ([0, 1])
# change ucb in mcts

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

LOAD = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ENVIRONMENT = 'TicTacToe'

if __name__ == '__main__':
    if ENVIRONMENT == 'CartPole':
        from Environments.CartPole.config import MuZeroConfigCartPole as Config

    elif ENVIRONMENT == 'TicTacToe':
        from Environments.TicTacToe.config import MuZeroConfigTicTacToe as Config
    
    config = Config()

    if LOAD:
        config.model.load_state_dict(torch.load(f'Models/{config.game}/model.pt', map_location=device))
        config.optimizer.load_state_dict(torch.load(f'Models/{config.game}/optimizer.pt', map_location=device))

    trainer = Trainer(config)
    trainer.run()
