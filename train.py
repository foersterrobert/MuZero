import random
import torch
import torch.nn.functional as F
from tqdm import trange
from mcts import MCTS
from config import *

class Trainer:
    def __init__(self, model, optimizer, env):
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.mcts = MCTS(model, env)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.env.get_initial_state()
        
        while True:
            neutral_state = self.env.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            
            temperature_action_probs = action_probs ** (1 / TEMPERATURE)
            temperature_action_probs = temperature_action_probs / temperature_action_probs.sum()
            action = torch.distributions.Categorical(temperature_action_probs).sample().item()
            
            state = self.env.get_next_state(state, action, player)
            
            value, is_terminal = self.env.get_value_and_terminated(state, action)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.env.get_opponent_value(value)
                    returnMemory.append((
                        self.env.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = self.env.get_opponent(player)
                
    def train(self, memory):
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), BATCH_SIZE):
            sample = memory[batch_idx:batch_idx+BATCH_SIZE]
            state, policy_targets, value_targets = zip(*sample)
            
            state = torch.stack(state).to(self.model.device)
            policy_targets = torch.stack(policy_targets).to(self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device).reshape(-1, 1)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        for iteration in range(NUM_ITERATIONS):
            memory = []
            
            self.model.eval()
            for i in trange(NUM_SELF_PLAY_ITERATIONS):
                memory += self.selfPlay()
                
            self.model.train()
            for i in trange(NUM_EPOCHS):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"Models/{self.env}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.env}/optimizer_{iteration}.pt")