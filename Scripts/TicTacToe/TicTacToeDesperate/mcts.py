import torch
import numpy as np
import math

class Node:
    def __init__(self, muZero, game, args, state, parent=None, action_taken=None, prior=None, visit_count=0):
        self.muZero = muZero
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    @torch.no_grad()
    def expand(self, policy):
        actions = [a for a in range(self.game.action_size) if policy[a] > 0]
        child_state = self.state.copy()
        child_state = np.expand_dims(child_state, axis=0).repeat(len(actions), axis=0)

        child_state = self.muZero.dynamics(
            torch.tensor(child_state, dtype=torch.float32, device=self.muZero.device), actions)
        child_state = child_state.cpu().numpy()
        
        for i, action in enumerate(actions):
            child = Node(
                self.muZero,
                self.game,
                self.args,
                state=child_state[i],
                parent=self,
                action_taken=action,
                prior=policy[action],
            )
            self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, muZero, game, args):
        self.muZero = muZero
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self, state, valid_moves):
        hidden_state = self.muZero.represent(
            torch.tensor(state, dtype=torch.float32, device=self.muZero.device).unsqueeze(0)
        )
        policy, _ = self.muZero.predict(hidden_state)
        hidden_state = hidden_state.cpu().numpy().squeeze(0)
        
        root = Node(self.muZero, self.game, self.args, hidden_state, visit_count=1)

        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)

        for search in range(self.args['num_mcts_searches']):
            node = root

            while node.is_expanded():
                node = node.select()

            policy, value = self.muZero.predict(
                torch.tensor(node.state, dtype=torch.float32, device=self.muZero.device).unsqueeze(0)
            )
            policy = torch.softmax(policy, dim=1).squeeze().cpu().numpy()
            value = value.item()

            node.expand(policy)
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size, dtype=np.float32)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs