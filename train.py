import random
import torch
import torch.nn.functional as F
from tqdm import trange
from mcts import MCTS
from config import *
from replayBuffer import *

class Trainer:
    def __init__(self, model, optimizer, env):
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.buffer = ReplayBuffer(env)
        self.mcts = MCTS(model, env)
        
    def selfPlay(self):
        trajectory = []
        player = 1
        state = self.env.get_initial_state()
        
        while True:
            neutral_state = self.env.change_perspective(state, player)
            encoded_state = self.env.get_encoded_state(neutral_state)
            action_probs = self.mcts.search(encoded_state)
            

            temperature_action_probs = action_probs ** (1 / TEMPERATURE)
            temperature_action_probs = temperature_action_probs / temperature_action_probs.sum()
            valid_actions = self.env.get_valid_actions(state)
            temperature_action_probs = temperature_action_probs * valid_actions
            temperature_action_probs = temperature_action_probs / temperature_action_probs.sum()
            
            action = torch.distributions.Categorical(temperature_action_probs).sample().item()
            
            state = self.env.get_next_state(state, action, player)
            
            value, is_terminal = self.env.get_value_and_terminated(state, action)
            
            trajectory.append(Transition(encoded_state, player, action, action_probs, value, value=None))
            
            if is_terminal:
                for i in range(len(trajectory)):
                    trajectory[i].value = value if trajectory[i].player == player else self.env.get_opponent_value(value)
                return trajectory   

            player = self.env.get_opponent(player)

    def train(self):
        random.shuffle(self.buffer.sequences)
        for batchIdx in range(0, len(self.buffer.sequences), BATCH_SIZE): 
            state, action, policy, value, reward = zip(*[
                (seq.state, seq.actions, seq.action_probs, seq.values, seq.rewards)
                for seq in self.buffer.sequences[batchIdx:batchIdx+BATCH_SIZE]
            ])

            state = torch.stack(state).to(self.model.device)
            action = torch.tensor(action, dtype=torch.long, device=self.model.device)
            policy = torch.stack([torch.stack(p) for p in policy]).to(self.model.device)
            value = torch.stack([torch.tensor(v, dtype=torch.float32) for v in value]).to(self.model.device)
            reward = torch.stack([torch.tensor(r, dtype=torch.float32) for r in reward]).to(self.model.device)

            state = self.model.represent(state)
            out_policy, out_value = self.model.predict(state)

            policy_loss = F.cross_entropy(out_policy, policy[:, 0]) 
            value_loss = F.mse_loss(out_value, value[:, 0].unsqueeze(1))
            reward_loss = torch.zeros(value_loss.shape, device=self.model.device)

            for k in range(1, K + 1):
                state, out_reward = self.model.dynamics(state, action[:, k - 1])
                reward_loss += F.mse_loss(out_reward.squeeze(1), reward[:, k])
                state.register_hook(lambda grad: grad * 0.5)

                out_policy, out_value = self.model.predict(state)

                policy_loss += F.cross_entropy(out_policy, policy[:, k])
                value_loss += F.mse_loss(out_value, value[:, k].unsqueeze(1))

            loss = (value_loss * VALUE_LOSS_WEIGHT + policy_loss + reward_loss).mean()
            loss.register_hook(lambda grad: grad * 1 / K)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

    def learn(self):
        for iteration in range(NUM_ITERATIONS):
            self.buffer.empty()
            
            self.model.eval()
            for i in trange(NUM_SELF_PLAY_ITERATIONS):
                trajectory = self.selfPlay()
                self.buffer.trajectories.append(trajectory)
            
            self.buffer.build_sequences()

            self.model.train()
            for i in trange(NUM_EPOCHS):
                self.train()
            
            torch.save(self.model.state_dict(), f"Models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/optimizer_{iteration}.pt")