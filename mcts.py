import numpy as np
import math

class Node:
    def __init__(self, state, reward, player, prior, dynamicsFunction, args, parent=None, action_taken=None):
        self.state = state
        self.reward = reward
        self.player = player
        self.children = []
        self.parent = parent
        self.total_value = 0
        self.visit_count = 0
        self.prior = prior
        self.dynamicFunction = dynamicsFunction
        self.action_taken = action_taken
        self.args = args

    def expand(self, action_probs):
        for a, prob in enumerate(action_probs):
            if prob != 0:
                child_state = self.state.copy()
                child_state, reward = self.dynamicFunction.predict(child_state, a)
                child = Node(
                    child_state,
                    reward,
                    -1 * self.player,
                    prob,
                    self.dynamicFunction,
                    self.args,
                    parent=self,
                    action_taken=a,
                )
                self.children.append(child)

    def backpropagate(self, value):
        self.total_value += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(-1 * value)

    def is_expandable(self):
        return len(self.children) > 0

    def select_child(self):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            ucb_score = self.get_ucb_score(child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def get_ucb_score(self, child):
        prior_score = child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count) * (self.args.c1 + math.log((self.visit_count + self.args.c2 + 1) / self.args.c2))
        if child.visit_count == 0:
            return prior_score
        return prior_score - (child.total_value / child.visit_count)

class MCTS:
    def __init__(self, representationFunction, dynamicsFunction, predictionFunction, game, args):
        self.representationFunction = representationFunction
        self.dynamicsFunction = dynamicsFunction
        self.predictionFunction = predictionFunction
        self.game = game
        self.args = args

    def search(self, observation, available_actions, player=1):
        hidden_state, reward = self.representationFunction(observation)
        root = Node(hidden_state, reward, player, 0, self.dynamicsFunction, self.args)
        action_probs, value = self.predictionFunction(hidden_state)
        action_probs = action_probs * available_actions
        action_probs = action_probs / np.sum(action_probs)
        root.expand(action_probs)

        for simulation in range(self.args['num_simulation_games']):
            node = root

            while node.is_expandable():
                node = node.select_child()

            canonical_hidden_state = self.game.get_canonical_state(node.state, node.player)
            action_probs, value = self.predictionFunction(canonical_hidden_state)
            node.expand(action_probs)
            node.backpropagate(value)

        return root