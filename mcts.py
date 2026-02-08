import math
import torch
import torch.distributions as dist
from config import *

class Node:
    def __init__(self, env, model, hidden_state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.env = env
        self.model = model
        self.hidden_state = hidden_state
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
        best_ucb = -float('inf')
        
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
        return q_value + C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            prob_value = prob.item()
            if prob_value > 0:
                child_hidden_state = self.hidden_state.clone()
                child_hidden_state = self.model.dynamics(child_hidden_state, [action])[0]

                child = Node(self.env, self.model, child_hidden_state, self, action, prob_value)
                self.children.append(child)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            value = self.env.get_opponent_value(value)
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.noise = dist.Dirichlet(torch.ones(self.env.action_size) * DIRICHLET_ALPHA)
        
    @torch.no_grad()
    def search(self, encoded_state):
        hidden_state = self.model.represent(encoded_state.unsqueeze(0).to(device=self.model.device))
        root = Node(self.env, self.model, hidden_state, visit_count=1)
        
        policy, _ = self.model.predict(
            hidden_state.to(device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu()
        policy = (1 - DIRICHLET_EPSILON) * policy + DIRICHLET_EPSILON * self.noise.sample()
        policy = policy / policy.sum()
        root.expand(policy)
        
        for i in range(NUM_MCTS_SEARCHES):
            node = root
            
            while node.is_expanded():
                node = node.select()
                
            policy, value = self.model.predict(
                node.hidden_state.to(device=self.model.device)
            )
            policy = torch.softmax(policy, axis=1).squeeze(0).cpu()
            value = self.env.get_opponent_value(value.item())
            
            node.expand(policy)    
            node.backpropagate(value)    
            
        action_probs = torch.zeros(self.env.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= action_probs.sum()
        return action_probs