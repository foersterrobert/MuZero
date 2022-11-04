import torch
from torch.optim import Adam
import random
import numpy as np
from games import TicTacToe
from models import MuZero
from trainer import Trainer

# Don't understand: backpropagation + training
# In training: scale hidden state ([0, 1])
# In training: scale loss 1/k

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        'num_iterations': 10,             # number of highest level iterations
        'num_train_games': 50,            # number of self-play games to play within each iteration
        'num_simulation_games': 32,       # number of mcts simulations when selecting a move within self-play
        'num_training_steps': 6,          # number of epochs for training on self-play data for each iteration
        'batch_size': 96,                 # batch size for training
        'temperature': 1,                 # temperature for the softmax selection of moves
        'K': 3,                           # unroll K steps of the dynamics function when training
        'c1': 1.25,                       # the value of the constant policy
        'c2': 19652,                      # the value of the constant policy
        'n': 10,                          # steps to unroll for reward prediction
        'discount': 0.997,
        'value_loss_weight': 0.25,
        'dynamicsFunction': {
            'num_resBlocks': 5,
            'hidden_planes': 128
        },
        'predictionFunction': {
            'num_resBlocks': 5,
            'hidden_planes': 128
        },
        'representationFunction': {
            'num_resBlocks': 4,
            'hidden_planes': 128
        }
    }
    game = TicTacToe()
    muZero = MuZero(game, args).to(device)
    optimizer = Adam(muZero.parameters(), lr=0.001, weight_decay=0.0001)

    trainer = Trainer(muZero, optimizer, game, args)
    trainer.run()
