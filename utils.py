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
        valid_moves = self.game.get_valid_locations(observation)
        
        encoded_observation = self.game.get_encoded_observation(observation)
        canonical_observation = self.game.get_canonical_state(encoded_observation, player).copy()

        with torch.no_grad():
            canonical_observation = torch.tensor(canonical_observation, dtype=torch.float32, device=self.model.device)
            policy, _ = self.model.predict(canonical_observation.unsqueeze(0))
            policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            policy = policy * valid_moves
        action = int(np.argmax(policy))
        return action
