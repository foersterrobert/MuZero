import torch
import numpy as np
import gymnasium as gym
from mcts import MCTS
from kaggle_environments import make, evaluate

class KaggleAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        if self.args['search']:
            self.mcts = MCTS(self.model, self.game, self.args['config'])

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        observation = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        observation[observation==2] = -1
        valid_moves = self.game.get_valid_locations(observation)
        
        encoded_observation = self.game.get_encoded_observation(observation)
        canonical_observation = self.game.get_canonical_state(encoded_observation, player).copy()

        with torch.no_grad():
            if self.args['search']:
                root = self.mcts.search(canonical_observation, 0, valid_moves)

                policy = [0] * self.game.action_size
                for child in root.children:
                    policy[child.action_taken] = child.visit_count
                policy /= np.sum(policy)

            else:
                hidden_state = torch.tensor(canonical_observation, dtype=torch.float32, device=self.args['device']).unsqueeze(0)
                hidden_state = self.model.represent(hidden_state)

                policy, _ = self.model.predict(hidden_state)
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

class GymAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        if self.args['search']:
            self.mcts = MCTS(self.model, self.game, self.args)

    @torch.no_grad()
    def predict(self, observation):
        encoded_observation = self.game.get_encoded_observation(observation)

        if self.args['search']:
            root = self.mcts.search(encoded_observation)

            policy = [0] * self.game.action_size
            for child in root.children:
                policy[child.action_taken] = child.visit_count
            policy /= np.sum(policy)

        else:
            policy, _ = self.model.predict(encoded_observation, augment=self.args['augment'])

        valid_moves = self.game.get_valid_locations(observation)
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
Player 1 | Wins: {np.sum(results == 1)} | Draws: {np.sum(results == 0)} | Losses: {np.sum(results == -1)}
Player 2 | Wins: {np.sum(results == -1)} | Draws: {np.sum(results == 0)} | Losses: {np.sum(results == 1)}
    """)

def evaluateGym(gameName, agent, num_iterations=1):
    if num_iterations == 1:
        env = gym.make(gameName, render_mode="human")
    else:
        env = gym.make(gameName)
    
    results = []
    for i in range(num_iterations):
        counter = 0
        observation, info = env.reset()
        while True:
            action = agent.predict(observation)
            observation, reward, done, info = env.step(action)

            if done:
                results.append(counter)
                break
            counter += 1
    
    print(f"""
Average number of moves: {sum(results) / len(results)}
    """)