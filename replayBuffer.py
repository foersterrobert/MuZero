from config import *
import numpy as np

class ReplayBuffer:
    def __init__(self, env):
        self.memory = []
        self.trajectories = []
        self.env = env

    def __len__(self):
        return len(self.trajectories)

    def empty(self):
        self.memory = []
        self.trajectories = []

    def build_trajectories(self):
        for i in range(len(self.memory)):
            observation, action, policy, reward, _, game_idx = self.memory[i]
            policy_list, action_list, value_list, reward_list = [policy], [action], [], [reward]

            # value bootstrap for N-step return
            # value starts at root value n steps ahead
            if i + N + 1 < len(self.memory) and self.memory[i + N + 1][5] == game_idx:
                value = self.memory[i + N + 1][4] * GAMMA ** N
            else:
                value = 0
            # add discounted rewards until end of game or N steps
            for n in range(2, N + 2):
                if i + n < len(self.memory) and self.memory[i + n][5] == game_idx:
                    _, _, _, reward, _, _ = self.memory[i + n]
                    value += reward * GAMMA ** (n - 2)
                else:
                    break
            value_list.append(value)

            for k in range(1, K + 1):
                if i + k < len(self.memory) and self.memory[i + k][5] == game_idx:
                    _, action, policy, reward, _, _ = self.memory[i + k]
                    action_list.append(action)
                    policy_list.append(policy)
                    reward_list.append(reward)

                    if i + k + N + 1 < len(self.memory) and self.memory[i + k + N + 1][5] == game_idx:
                        value = self.memory[i + k + N + 1][4] * GAMMA ** N
                    else:
                        value = 0
                    for n in range(2, N + 2):
                        if i + k + n < len(self.memory) and self.memory[i + k + n][5] == game_idx:
                            _, _, _, reward, _, _ = self.memory[i + k + n]
                            value += reward * GAMMA ** (n - 2)
                        else:
                            break
                    value_list.append(value)

                else:
                    action_list.append(np.random.choice(self.env.action_size))
                    policy_list.append(np.full(self.env.action_size, 1 / self.env.action_size))
                    value_list.append(0)
                    reward_list.append(0)

            policy_list = np.stack(policy_list)
            self.trajectories.append((observation, action_list, policy_list, value_list, reward_list))