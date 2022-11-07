import numpy as np

class ReplayBuffer:
    def __init__(self, args, game):
        self.memory = []
        self.trajectories = []
        self.args = args
        self.game = game

    def __len__(self):
        return len(self.memory)

    def empty(self):
        self.memory = []

    def add(self, game_memory):
        self.memory.extend(game_memory)
    
    def build_trajectories(self):
        self.trajectories = []
        for i in range(len(self.memory)):
            observation, action, policy, value, reward, game_idx, is_terminal = self.memory[i]
            if is_terminal:
                action = np.random.choice(self.game.action_size)
            policy_list, action_list, value_list, reward_list = [policy], [action], [value], [reward]

            for k in range(1, self.args['K'] + 1):
                if i + k < len(self.memory) and self.memory[i + k][5] == game_idx:
                    _, next_action, next_policy, next_value, next_reward, _, next_terminal = self.memory[i + k]
                    if next_terminal:
                        next_action = np.random.choice(self.game.action_size)
                    action_list.append(next_action)
                    policy_list.append(next_policy)
                    value_list.append(next_value)
                    reward_list.append(next_reward)

                else:
                    action_list.append(np.random.choice(self.game.action_size))
                    policy_list.append(policy_list[-1])
                    value_list.append(-1 * value_list[-1])
                    reward_list.append(-1 * reward_list[-1])

            self.trajectories.append((observation, action_list, policy_list, value_list, reward_list))