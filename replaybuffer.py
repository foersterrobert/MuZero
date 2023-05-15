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
            observation, action, policy, value, game_idx, is_terminal = self.memory[i]
            if not is_terminal:
                policy_list, action_list, value_list = [policy], [action], [value]

                for k in range(1, self.args['K'] + 1):
                    if i + k < len(self.memory) and self.memory[i + k][4] == game_idx:
                        _, action, policy, value, _, is_terminal = self.memory[i + k]
                        if is_terminal:
                            action = np.random.choice(self.game.action_size)
                        policy_list.append(policy)
                        action_list.append(action)
                        value_list.append(value)

                    else:
                        policy_list.append(np.zeros(self.game.action_size, dtype=np.float32))
                        action_list.append(np.random.choice(self.game.action_size))
                        value_list.append(self.game.get_opponent_value(value_list[-1]))

                policy_list = np.stack(policy_list)
                self.trajectories.append((observation, policy_list, action_list, value_list))
