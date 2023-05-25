import numpy as np
import torch
from mcts import MCTS
from kaggle_environments import make, evaluate

class KaggleAgent:
    def __init__(self, muZero, game, args, name='MuZero'):
        self.muZero = muZero
        self.game = game
        self.args = args
        self.name = name
        if self.args['search']:
            self.mcts = MCTS(self.muZero, self.game, self.args)

    def __repr__(self):
        return self.name

    def __call__(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        observation = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        observation[observation==2] = -1
        valid_moves = self.game.get_valid_moves(observation)
        
        neutral_observation = self.game.change_perspective(observation, player).copy()
        encoded_observation = self.game.get_encoded_observation(neutral_observation)

        with torch.no_grad():
            if self.args['search']:
                policy = self.mcts.search(encoded_observation, valid_moves)

            else:
                hidden_state = torch.tensor(encoded_observation, dtype=torch.float32, device=self.muZero.device).unsqueeze(0)
                hidden_state = self.muZero.represent(hidden_state)
                print(hidden_state)

                policy, _ = self.muZero.predict(hidden_state)
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()

        policy *= valid_moves
        policy /= np.sum(policy)

        if self.args['temperature'] == 0:
            action = int(np.argmax(policy))
        elif self.args['temperature'] == float('inf'):
            action = np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
        else:
            policy = policy ** (1 / self.args['temperature'])
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy)

        return action
    
def evaluateKaggle(gameName, players, num_iterations=1):
    if num_iterations == 1:
        env = make(gameName, debug=True)
        env.run(players)
        return env.render(mode="ipython")

    results = np.array(evaluate(gameName, players, num_episodes=num_iterations))[:, 0]
    print(f"""
{players[0]} | Wins: {np.sum(results == 1)} | Draws: {np.sum(results == 0)} | Losses: {np.sum(results == -1)}
{players[1]} | Wins: {np.sum(results == -1)} | Draws: {np.sum(results == 0)} | Losses: {np.sum(results == 1)}
    """)
    return results