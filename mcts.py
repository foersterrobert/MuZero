import torch
import numpy as np
import math

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
    def __init__(self, state, reward, prior, model, config, game, parent=None, action_taken=None, visit_count=0):
        self.state = state
        self.reward = reward
        self.children = []
        self.parent = parent
        self.total_value = 0
        self.visit_count = visit_count
        self.prior = prior
        self.model = model
        self.action_taken = action_taken
        self.config = config
        self.game = game

    @torch.no_grad()
    def expand(self, action_probs):
        actions = [a for a in range(self.game.action_size) if action_probs[a] > 0]
        expand_state = self.state.copy()
        expand_state = np.expand_dims(expand_state, axis=0).repeat(len(actions), axis=0)

        expand_state, reward = self.model.dynamics(
            torch.tensor(expand_state, dtype=torch.float32, device=self.config.device), actions)
        expand_state = expand_state.detach().cpu().numpy()
        expand_state = self.game.get_canonical_state(expand_state, -1).copy()
        
        for i, a in enumerate(actions):
            child = Node(
                expand_state[i],
                reward,
                action_probs[a],
                self.model,
                self.config,
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

    def is_expanded(self):
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
        prior_score = self.config.c_init + math.log((self.visit_count + self.config.c_base + 1) / self.config.c_base)
        prior_score *= math.sqrt(self.visit_count) / (1 + child.visit_count)
        prior_score *= child.prior

        if child.visit_count == 0:
            return prior_score
        return prior_score + self.game.get_opponent_value(child.total_value / child.visit_count)

class MCTS:
    def __init__(self, model, game, config):
        self.model = model
        self.game = game
        self.config = config

    @torch.no_grad()
    def search(self, hidden_state, reward, available_actions):
        hidden_state = torch.tensor(hidden_state, dtype=torch.float32, device=self.config.device).unsqueeze(0)
        hidden_state = self.model.represent(hidden_state)
        action_probs, _ = self.model.predict(hidden_state)
        hidden_state = hidden_state.cpu().numpy().squeeze(0)

        root = Node(hidden_state, reward, 0, self.model, self.config, self.game, visit_count=1)

        action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().squeeze(0)
        action_probs = (1 - self.config.dirichlet_epsilon) * action_probs + self.config.dirichlet_epsilon * np.random.dirichlet([self.config.dirichlet_alpha] * self.game.action_size)
        action_probs *= available_actions
        action_probs /= np.sum(action_probs)

        root.expand(action_probs)

        for simulation in range(self.config.num_mcts_runs):
            node = root

            while node.is_expanded():
                node = node.select_child()

            if self.config.cheatAvailableActions or self.config.cheatTerminalState:
                unencoded_state = node.state.copy()
                unencoded_state = (
                    unencoded_state * np.array([-1, 0, 1]).repeat(9).reshape(3, 3, 3)
                ).sum(axis=0)

                if self.config.cheatTerminalState:
                    is_terminal, value = self.game.check_terminal_and_value(unencoded_state, node.action_taken)
                    value = self.game.get_opponent_value(value)

            if not self.config.cheatTerminalState or not is_terminal:
                action_probs, value = self.model.predict(
                    torch.tensor(node.state, dtype=torch.float32, device=self.config.device).unsqueeze(0)
                )
                action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().squeeze(0)
                value = value.item()

                if self.config.cheatAvailableActions:
                    available_actions = self.game.get_valid_locations(unencoded_state)
                    action_probs *= available_actions
                    action_probs /= np.sum(action_probs)

                node.expand(action_probs)
            node.backpropagate(value)

        return root