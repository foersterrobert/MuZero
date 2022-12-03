import torch
from torch.optim import Adam
import numpy as np
import random
from games import TicTacToe
from models import MuZero
from trainer import Trainer
# from trainerParallel import Trainer

# In training: scale hidden state ([0, 1])
# change ucb in mcts

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

LOAD = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    # args = {
    #     'num_iterations': 48,             # number of highest level iterations
    #     'num_train_games': 500,           # number of self-play games to play within each iteration
    #     'num_simulation_games': 800,      # number of mcts simulations when selecting a move within self-play
    #     'num_training_steps': 32,         # number of epochs for training on self-play data for each iteration
    #     'batch_size': 128,                # batch size for training
    #     'temperature': 1,                 # temperature for the softmax selection of moves
    #     'K': 5,                           # unroll K steps of the dynamics function when training
    #     'c1': 1.25,                       # the value of the constant policy
    #     'c2': 19652,                      # the value of the constant policy
    #     'n': 10,                          # steps to unroll for reward prediction
    #     'discount': 0.997
    # }
    args = {
        'num_iterations': 8,              # number of highest level iterations
        'num_train_games': 500,           # number of self-play games to play within each iteration
        'group_size': 500,                # group size for parallel training
        'num_mcts_runs': 60,              # number of mcts simulations when selecting a move within self-play
        'num_epochs': 4,                  # number of epochs for training on self-play data for each iteration
        'batch_size': 64,                 # batch size for training
        'temperature': 1,                 # temperature for the softmax selection of moves
        'K': 0, # Cheat!                  # unroll K steps of the dynamics function when training
        'c': 2,                           # the value of the constant policy
        'c1': 1.25,                       # the value of the constant policy
        'c2': 19652,                      # the value of the constant policy
        'n': 10,                          # steps to unroll for reward prediction
        'dirichlet_alpha': 0.3,           # dirichlet noise for exploration
        'dirichlet_epsilon': 0.125,       # dirichlet noise for exploration
        'discount': 0.997,
        'value_loss_weight': 1, # 0.25,
        'dynamicsFunction': {
            'num_resBlocks': 4,
            'hidden_planes': 128
        },
        'predictionFunction': {
            'num_resBlocks': 4,
            'hidden_planes': 128
        },
        'representationFunction': {
            'num_resBlocks': 4,
            'hidden_planes': 128
        },
        'cheatAvailableActions': True,
        'cheatTerminalState': True,
    }
    game = TicTacToe()
    muZero = MuZero(game, args).to(device)
    optimizer = Adam(muZero.parameters(), lr=0.001, weight_decay=0.0001)
    if LOAD:
        muZero.load_state_dict(torch.load(f'Models/{game}/model.pt', map_location=device))
        optimizer.load_state_dict(torch.load(f'Models/{game}/optimizer.pt', map_location=device))

    trainer = Trainer(muZero, optimizer, game, args)
    trainer.run()
