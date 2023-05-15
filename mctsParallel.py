from .mcts import Node, MinMaxStats
import numpy as np
import torch

class MCTS:
    def __init__(self, muZero, game, args):
        self.muZero = muZero
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self, observations, valid_moves, spGames):
        hidden_states = self.muZero.represent(
            torch.tensor(observations, dtype=torch.float32, device=self.muZero.device)
        )
        policy, _ = self.muZero.predict(hidden_states)
        
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        policy *= valid_moves
        policy /= np.sum(policy, axis=1, keepdims=True)

        hidden_states = hidden_states.cpu().numpy()

        for i, g in enumerate(spGames):
            g.root = Node(
                self.muZero, self.game, self.args, 
                hidden_states[i], visit_count=1)
            g.root.expand(policy[i])

        for search in range(self.args['num_mcts_searches']):
            for g in spGames:
                node = g.root

                while node.is_expanded():
                    node = node.select()

                g.node = node

            hidden_states = np.stack([g.node.state for g in spGames])
            policy, value = self.muZero.predict(
                torch.tensor(hidden_states, dtype=torch.float32, device=self.muZero.device)
            )
            policy = torch.softmax(policy, dim=1).cpu().numpy()
            value = value.cpu().numpy().reshape(-1)

            for i, g in enumerate(spGames):
                g.node.expand(policy[i])
                g.node.backpropagate(value[i])