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
        self.trajectories = []

    def build_trajectories(self):
        for i in range(len(self.memory)):
            observation, action, policy, value, reward, game_idx, is_terminal = self.memory[i]
            if is_terminal:
                action = np.random.choice(self.game.action_size)
            policy_list, action_list, value_list, reward_list = [policy], [action], [value], [reward]

            for k in range(1, self.args['K'] + 1):
                if i + k < len(self.memory) and self.memory[i + k][5] == game_idx:
                    _, action, policy, value, reward, _, is_terminal = self.memory[i + k]
                    if is_terminal:
                        action = np.random.choice(self.game.action_size)
                    action_list.append(action)
                    policy_list.append(policy)
                    value_list.append(value)
                    reward_list.append(reward)

                else:
                    action_list.append(np.random.choice(self.game.action_size))
                    policy_list.append(policy_list[-1])
                    value_list.append(self.game.get_opponent_value(1) * value_list[-1])
                    reward_list.append(self.game.get_opponent_value(1) * reward_list[-1])

            policy_list = np.stack(policy_list)
            self.trajectories.append((observation, action_list, policy_list, value_list, reward_list))
