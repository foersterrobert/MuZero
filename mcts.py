import torch
import numpy as np
import math

class Node:
    def __init__(self, state, reward, player, prior, muZero, args, game, parent=None, action_taken=None):
        self.state = state
        self.reward = reward
        self.player = player
        self.children = []
        self.parent = parent
        self.total_value = 0
        self.visit_count = 0
        self.prior = prior
        self.muZero = muZero
        self.action_taken = action_taken
        self.args = args
        self.game = game

    @torch.no_grad()
    def expand(self, action_probs):
        for a, prob in enumerate(action_probs):
            if prob != 0:
                child_state = self.state.detach().clone()
                child_state = self.game.get_canonical_state(child_state, self.player)
                child_state, reward = self.muZero.dynamics(child_state, a)
                child_state = self.game.get_canonical_state(child_state, self.player)
                child = Node(
                    child_state,
                    reward,
                    self.game.get_opponent_player(self.player),
                    prob,
                    self.muZero,
                    self.args,
                    self.game,
                    parent=self,
                    action_taken=a,
                )
                self.children.append(child)

    def backpropagate(self, value):
        self.total_value += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(self.game.get_opponent_value(value))

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
        # prior_score = child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count) * (self.args['c1'] + math.log((self.visit_count + self.args['c2'] + 1) / self.args['c2']))
        prior_score = self.args['c'] * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        if child.visit_count == 0:
            return prior_score
        return prior_score - (child.total_value / child.visit_count)

class MCTS:
    def __init__(self, muZero, game, args):
        self.muZero = muZero
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self, observation, reward, available_actions, player=1):
        hidden_state = self.muZero.represent(observation)
        root = Node(hidden_state, reward, player, 0, self.muZero, self.args, self.game)

        action_probs, value = self.muZero.predict(hidden_state)
        action_probs = torch.softmax(action_probs, dim=1).squeeze(0)
        value = value.item()

        action_probs = action_probs * available_actions
        action_probs = action_probs / torch.sum(action_probs)

        root.expand(action_probs)

        for simulation in range(self.args['num_simulation_games']):
            node = root

            while node.is_expandable():
                node = node.select_child()

            canonical_hidden_state = self.game.get_canonical_state(node.state, node.player)
            action_probs, value = self.muZero.predict(canonical_hidden_state)
            action_probs = torch.softmax(action_probs, dim=1).squeeze(0)
            value = value.item()

            node.expand(action_probs)
            node.backpropagate(value)

        return root