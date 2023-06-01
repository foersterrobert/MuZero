import numpy as np

class ReplayBuffer:
    def __init__(self, args, game):
        self.memory = []
        self.trajectories = []
        self.args = args
        self.game = game

    def __len__(self):
        return len(self.trajectories)
    
    def empty(self):
        self.memory = []
        self.trajectories = []

    def build_trajectories(self):
        for i in range(len(self.memory)):
            observation, action, policy, value, game_idx = self.memory[i]
            policy_list, action_list, value_list = [policy], [action], [value]

            for k in range(1, self.args['K'] + 1):
                if i + k < len(self.memory) and self.memory[i + k][4] == game_idx:
                    _, action, policy, value, _ = self.memory[i + k]
                    policy_list.append(policy)
                    action_list.append(action)
                    value_list.append(value)

                else:
                    policy_list.append(np.ones(self.game.action_size) / self.game.action_size)
                    action_list.append(np.random.choice(self.game.action_size))
                    value_list.append(0)
                    # value_list.append(self.game.get_opponent_value(value_list[-1]))

            policy_list = np.stack(policy_list)
            self.trajectories.append((observation, policy_list, action_list, value_list))