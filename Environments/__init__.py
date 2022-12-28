
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
        value_support,
        reward_support,
    ):
        self.device = device

        # Args
        self.num_iterations = num_iterations
        self.num_train_games = num_train_games
        self.group_size = group_size
        self.num_mcts_runs = num_mcts_runs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self.K = K
        self.N = N
        self.c_init = c_init
        self.c_base = c_base
        self.gamma = gamma

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