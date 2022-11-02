import numpy as np

class ReplayBuffer:
    def __init__(self, args):
        self.buffer = []
        self.args = args

    def empty(self):
        self.buffer = []

    def add(self, game_memory):
        self.buffer.extend(game_memory)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        trajectories = []
        for i in indices:
            observation, action, action_probs, final_reward, reward, game_idx = self.buffer[i]
            action_probs_list, action_list, final_reward_list, reward_list = [action_probs], [action], [final_reward * self.args['discount'] ** self.args['n']], [reward]

            for k in range(self.args['K'] - 1):
                if i + k + 1 < len(self.buffer) and self.buffer[i + k + 1][4] == game_idx:
                    _, next_action, next_action_probs, next_final_reward, next_reward, _ = self.buffer[i + k + 1]
                    action_list.append(next_action)
                    action_probs_list.append(next_action_probs)
                    final_reward_list.append(next_final_reward * self.args['discount'] ** self.args['n'])
                    reward_list.append(next_reward)

                else:
                    action_list.append(action)
                    action_probs_list.append(action_probs)
                    final_reward_list.append(final_reward * self.args['discount'] ** self.args['n'])
                    reward_list.append(reward)

            trajectories.append((observation, action_list, action_probs_list, final_reward_list, reward_list))
        
        return trajectories                        
