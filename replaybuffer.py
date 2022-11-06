import numpy as np

class ReplayBuffer:
    def __init__(self, args, game):
        self.buffer = []
        self.args = args
        self.game = game

    def empty(self):
        self.buffer = []

    def add(self, game_memory):
        self.buffer.extend(game_memory)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        trajectories = []
        for i in indices:
            observation, action, action_probs, final_reward, reward, game_idx, is_terminal = self.buffer[i]
            if is_terminal:
                action = np.random.choice(self.game.action_size)
            action_probs_list, action_list, final_reward_list, reward_list = [action_probs], [action], [final_reward], [reward]

            for k in range(1, self.args['K'] + 1):
                if i + k < len(self.buffer) and self.buffer[i + k][5] == game_idx:
                    _, next_action, next_action_probs, next_final_reward, next_reward, _, next_terminal = self.buffer[i + k]
                    if next_terminal:
                        next_action = np.random.choice(self.game.action_size)
                    action_list.append(next_action)
                    action_probs_list.append(next_action_probs)
                    final_reward_list.append(next_final_reward)
                    reward_list.append(next_reward)

                else:
                    action_list.append(np.random.choice(self.game.action_size))
                    action_probs_list.append(action_probs_list[-1])
                    final_reward_list.append(-1 * final_reward_list[-1])
                    reward_list.append(-1 * reward_list[-1])

            trajectories.append((observation, action_list, action_probs_list, final_reward_list, reward_list))
        
        return trajectories                        
