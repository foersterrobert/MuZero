import math
import torch
import torch.distributions as dist

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
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            prob_value = prob.item() if isinstance(prob, torch.Tensor) else prob
            if prob_value > 0:
                child_state = self.state.clone()
                child_state = self.env.get_next_state(child_state, action, 1)
                child_state = self.env.change_perspective(child_state, player=-1)

                child = Node(self.env, self.args, child_state, self, action, prob_value)
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
            self.env.get_encoded_state(state).to(device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu()
        dirichlet_noise = dist.Dirichlet(torch.ones(self.env.action_size) * self.args['dirichlet_alpha']).sample()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * dirichlet_noise
        
        valid_actions = self.env.get_valid_actions(state)
        policy = policy * valid_actions
        policy = policy / policy.sum()
        root.expand(policy)
        
        for i in range(self.args['num_mcts_searches']):
            node = root
            
            while node.is_expanded():
                node = node.select()
                
            value, is_terminal = self.env.get_value_and_terminated(node.state, node.action_taken)
            value = self.env.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    self.env.get_encoded_state(node.state).to(device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu()
                valid_actions = self.env.get_valid_actions(node.state)
                policy = policy * valid_actions
                policy = policy / policy.sum()
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
        action_probs = torch.zeros(self.env.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= action_probs.sum()
        return action_probs