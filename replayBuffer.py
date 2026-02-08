from config import *
import torch
from dataclasses import dataclass

@dataclass
class Transition:
    state: torch.Tensor
    player: int
    action: int
    action_probs: torch.Tensor
    reward: float
    value: float

@dataclass
class Sequence:
    state: torch.Tensor   # (3, 3)
    actions: list       # List of int
    action_probs: list      # List of torch.Tensor
    values: list        # List of float
    rewards: list       # List of float

class ReplayBuffer:
    def __init__(self, env):
        self.env = env
        self.trajectories = [] # store whole env trajectories from self-play, each trajectory is a list of (state, action, action_probs, reward) tuples
        self.sequences = [] # store k-step sequences for training, each sequence is a list of (state, action, action_probs, value, reward) tuples

    def empty(self):
        self.trajectories = []
        self.sequences = []

    def build_sequences(self):
        for trajectory in self.trajectories:
            for i in range(len(trajectory)):
                state, action, action_probs, reward, value = trajectory[i].state, trajectory[i].action, trajectory[i].action_probs, trajectory[i].reward, trajectory[i].value
                action_list, action_probs_list, value_list, reward_list = [action], [action_probs], [value], [reward]

                for k in range(1, K + 1):
                    if i + k < len(trajectory):
                        action_k, action_probs_k, reward_k, value_k = trajectory[i + k].action, trajectory[i + k].action_probs, trajectory[i + k].reward, trajectory[i + k].value
                        action_list.append(action_k)
                        action_probs_list.append(action_probs_k)
                        value_list.append(value_k)
                        reward_list.append(reward_k)

                    else:
                        action_list.append(torch.randint(self.env.action_size, (1,)).item())
                        action_probs_list.append(torch.full((self.env.action_size,), 1 / self.env.action_size))
                        reward_list.append(0)
                        value_list.append(0)

                self.sequences.append(Sequence(
                    state=state,
                    actions=action_list,
                    action_probs=action_probs_list,
                    values=value_list,
                    rewards=reward_list,
                ))
