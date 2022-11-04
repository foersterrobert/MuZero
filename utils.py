import numpy as np
import torch

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
        canonical_observation = self.game.get_canonical_observation(encoded_observation, player)
        hidden_state = torch.from_numpy(canonical_observation).unsqueeze(0)

        policy, value = self.model.predict(hidden_state)
        policy = policy.detach().cpu().numpy()[0]
        policy = policy * valid_moves
        action = int(np.argmax(policy))
        return action
