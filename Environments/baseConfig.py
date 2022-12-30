
class MuZeroConfigBasic:
    def __init__(
        self,
        device,
        num_iterations,
        num_train_games,
        group_size,
        num_mcts_runs,
        num_epochs,
        batch_size,
        temperature,
        K,
        N,
        c_init,
        c_base,
        gamma,
        dirichlet_alpha,
        dirichlet_epsilon,
        value_loss_weight,
        value_support,
        reward_support,
    ):
        self.device = device

        # Args
        self.num_iterations = num_iterations # number of highest level iterations
        self.num_train_games = num_train_games # number of self-play games to play within each iteration
        self.group_size = group_size # group size for parallel training
        self.num_mcts_runs = num_mcts_runs # number of mcts simulations when selecting a move within self-play
        self.num_epochs = num_epochs # number of epochs for training on self-play data for each iteration
        self.batch_size = batch_size # batch size for training
        self.temperature = temperature # temperature for the softmax selection of moves
        self.K = K # unroll K steps of the dynamics function when training | Set to 0 when cheating
        self.N = N # steps to unroll for reward prediction
        self.c_init = c_init # the value of the constant policy
        self.c_base = c_base # the value of the constant policy
        self.gamma = gamma # discount factor
        self.dirichlet_alpha = dirichlet_alpha # dirichlet noise for exploration
        self.dirichlet_epsilon = dirichlet_epsilon # dirichlet noise for exploration
        self.value_loss_weight = value_loss_weight # weight for value loss

        #     'num_iterations': 48,             # number of highest level iterations
        #     'num_train_games': 500,           # number of self-play games to play within each iteration
        #     'num_simulation_games': 800,      # number of mcts simulations when selecting a move within self-play
        #     'num_training_steps': 32,         # number of epochs for training on self-play data for each iteration
        #     'batch_size': 128,                # batch size for training
        #     'temperature': 1,                 # temperature for the softmax selection of moves
        #     'K': 5,                           # unroll K steps of the dynamics function when training
        #     'c1': 1.25,                       # the value of the constant policy
        #     'c2': 19652,                      # the value of the constant policy
        #     'n': 10,                          # steps to unroll for reward prediction
        #     'discount': 0.997

        # Support
        self.value_support = value_support
        self.reward_support = reward_support

class DiscreteSupport:
    def __init__(self, min, max):
        assert min < max
        self.min = min
        self.max = max
        self.range = range(min, max + 1)
        self.size = len(self.range)