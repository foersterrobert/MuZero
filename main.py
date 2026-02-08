from mcts import MCTS
from train import Trainer
from config import *
from replayBuffer import *
from model import MuZero
from env import TicTacToe
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TicTacToe()

model = MuZero(env, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trainer = Trainer(model, optimizer, env)
trainer.learn()