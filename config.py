# Number of highest level iterations
NUM_ITERATIONS = 48

# Number of self-play games to play within each iteration
NUM_SELF_PLAY_ITERATIONS = 10

# Number of games to play in parallel
NUM_PARALLEL_GAMES = 100

# Number of mcts simulations when selecting a move within self-play
NUM_MCTS_SEARCHES = 600

# Number of epochs for training on self-play data for each iteration
NUM_EPOCHS = 4

# Batch size for training
BATCH_SIZE = 128

# Temperature for the softmax selection of moves
TEMPERATURE = 1.25

# The value of the constant policy
C = 2

# Whether to augment the training data with flipped states
AUGMENT = False

# The value of the dirichlet alpha
DIRICHLET_ALPHA = 0.3

# The value of the dirichlet epsilon
DIRICHLET_EPSILON = 0.25

# The number of steps to look ahead for value bootstrapping
N = 10

# The number of steps to look ahead for policy targets
K = 5

# The discount factor for future rewards
GAMMA = 0.997

# The weight of the value loss in the total loss
VALUE_LOSS_WEIGHT = 0.25

# The maximum norm for gradient clipping
MAX_GRAD_NORM = 5
