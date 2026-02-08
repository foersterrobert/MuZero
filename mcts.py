import numpy as np
import math
import torch

class Node:
    def __init__(self, env, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.env = env
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
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.clone()
                child_state = self.env.get_next_state(child_state, action, 1)
                child_state = self.env.change_perspective(child_state, player=-1)

                child = Node(self.env, self.args, child_state, self, action, prob)
                self.children.append(child)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            value = self.env.get_opponent_value(value)
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, model, env, args):
        self.model = model
        self.env = env
        self.args = args
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.env, self.args, state, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.env.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.env.action_size)
        
        valid_moves = self.env.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_mcts_searches']):
            node = root
            
            while node.is_expanded():
                node = node.select()
                
            value, is_terminal = self.env.get_value_and_terminated(node.state, node.action_taken)
            value = self.env.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.env.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.env.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
        action_probs = np.zeros(self.env.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs