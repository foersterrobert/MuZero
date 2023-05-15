import torch
import numpy as np
import random

# In training: scale hidden state ([0, 1])
# maybe opponent value should be 1 - value instead of -value

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

PARALLEL = True
LOAD = False

ENVIRONMENT = 'TicTacToe'

if __name__ == '__main__':
    if ENVIRONMENT == 'CartPole':
        from Environments.CartPole.config import MuZeroConfigCartpole as Config

    elif ENVIRONMENT == 'TicTacToe':
        from Environments.TicTacToe.config import MuZeroConfigTicTacToe as Config
    
    elif ENVIRONMENT == 'CarRacing':
        from Environments.CarRacing.config import MuZeroConfigCarRacing as Config
    
    config = Config()

    if PARALLEL:
        from muZeroParallel import MuZeroParallel as MuZero
    else:
        from muZero import MuZero

    if LOAD:
        config.model.load_state_dict(torch.load(f'Environments/{config}/Models/{config.model}.pt', map_location=config.device))
        config.optimizer.load_state_dict(torch.load(f'Environments/{config}/Models/{config.model}_optimizer.pt', map_location=config.device))

    muzero = MuZero(config)
    muzero.run()
