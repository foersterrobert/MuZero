import torch
import math
import numpy as np

class MinMaxStats:
    def __init__(self, known_bounds):
        self.maximum = known_bounds['max'] if known_bounds else -float('inf')
        self.minimum = known_bounds['min'] if known_bounds else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, state, reward, prior, muZero, args, game, parent=None, action_taken=None, visit_count=0):
        self.state = state
        self.reward = reward
        self.children = []
        self.parent = parent
        self.total_value = 0
        self.visit_count = visit_count # Should start at 1 for root node
        self.prior = prior
        self.muZero = muZero
        self.action_taken = action_taken
        self.args = args
        self.game = game

    @torch.no_grad()
    def expand(self, action_probs):
        actions = [a for a in range(self.game.action_size) if action_probs[a] > 0]
        expand_state = self.state.copy()
        expand_state = np.expand_dims(expand_state, axis=0).repeat(len(actions), axis=0)

        expand_state, reward = self.muZero.dynamics(
            torch.tensor(expand_state, dtype=torch.float32, device=self.muZero.device), actions)
        expand_state = expand_state.cpu().numpy()
        reward = self.muZero.inverse_reward_transform(reward).cpu().numpy().flatten()
        
        for i, a in enumerate(actions):
            child = Node(
                expand_state[i],
                reward[i],
                action_probs[a],
                self.muZero,
                self.args,
                self.game,
                parent=self,
                action_taken=a,
            )
            self.children.append(child)

    def backpropagate(self, value, minMaxStats):
        self.total_value += value
        self.visit_count += 1
        minMaxStats.update(self.value())
        if self.parent is not None:
            value = self.reward + self.args['gamma'] * value
            self.parent.backpropagate(value, minMaxStats)

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def select_child(self, minMaxStats):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            ucb_score = self.get_ucb_score(child, minMaxStats)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def get_ucb_score(self, child, minMaxStats):
        pb_c = math.log((self.visit_count + self.args["pb_c_base"] + 1) /
                  self.args["pb_c_base"]) + self.args["pb_c_init"]
        pb_c *= math.sqrt(self.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = minMaxStats.normalize(child.reward + self.args['gamma'] * child.value())
        else:
            # value_score = 0
            value_score = minMaxStats.normalize(child.reward)
        return prior_score + value_score

class MCTS:
    def __init__(self, muZero, game, args):
        self.muZero = muZero
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self, state, reward):
        minMaxStats = MinMaxStats(self.args['known_bounds'])
        hidden_state = self.muZero.represent(
            torch.tensor(state, dtype=torch.float32, device=self.muZero.device).unsqueeze(0)
        )
        action_probs, _ = self.muZero.predict(hidden_state)
        hidden_state = hidden_state.cpu().numpy().squeeze(0)
        
        root = Node(hidden_state, reward, 0, self.muZero, self.args, self.game, visit_count=1)

        action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().squeeze(0)
        action_probs = action_probs * (1 - self.args['dirichlet_epsilon']) + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        action_probs /= np.sum(action_probs)

        root.expand(action_probs)

        for simulation in range(self.args['num_mcts_runs']):
            node = root

            while node.is_expanded():
                node = node.select_child(minMaxStats)

            action_probs, value = self.muZero.predict(
                torch.tensor(node.state, dtype=torch.float32, device=self.muZero.device).unsqueeze(0)
            )
            action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().squeeze(0)
            value = self.muZero.inverse_value_transform(value).item()

            node.expand(action_probs)
            node.backpropagate(value, minMaxStats)

        return root