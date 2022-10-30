
import torch
from torch.optim import Adam
import random
import numpy as np
from games import TicTacToe
from models import PredictionFunction, DynamicsFunction, RepresentationFunction
from trainer import Trainer

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = {
        'num_iterations': 48,             # number of highest level iterations
        'num_train_games': 500,           # number of self-play games to play within each iteration
        'num_simulation_games': 800,      # number of mcts simulations when selecting a move within self-play
        'num_epochs': 4,                  # number of epochs for training on self-play data for each iteration
        'batch_size': 128,                # batch size for training
        'temperature': 1,                 # temperature for the softmax selection of moves
        'c1': 1.25,                       # the value of the constant policy
        'c2': 19652,                      # the value of the constant policy
        'discount': 0.997
    }
    game = TicTacToe()
    representationFunction = RepresentationFunction(5, 64)
    dynamicsFunction = DynamicsFunction(5, 64)
    predictionFunction = PredictionFunction(5, 128)
    optimizer = Adam(predictionFunction.parameters(), lr=0.001, weight_decay=0.0001)

    trainer = Trainer(representationFunction, dynamicsFunction, predictionFunction, optimizer, game, args)
    trainer.run()
