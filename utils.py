import torch
import numpy as np

class KaggleAgent:
    def __init__(self, model, game):
        self.model = model
        self.game = game

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        observation = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        observation[observation==2] = -1
        observation = torch.tensor(observation, dtype=torch.int8, device=self.game.device)
        valid_moves = self.game.get_valid_locations(observation)
        
        encoded_observation = self.game.get_encoded_observation(observation)
        canonical_observation = self.game.get_canonical_state(encoded_observation, player)

        with torch.no_grad():
            policy, _ = self.model.predict(canonical_observation)
            policy = torch.softmax(policy, dim=1).squeeze(0)
            policy = policy * valid_moves
        action = torch.argmax(policy).item()
        return action
