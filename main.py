from config import *

import torch
from env import TicTacToe
from model import MuZero
from train import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TicTacToe()

model = MuZero(env, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trainer = Trainer(model, optimizer, env)
trainer.learn()
