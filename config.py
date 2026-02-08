import os
import random

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional outside project environments
    np = None

# Must be set before the first CUDA context is created for deterministic CUDA matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

# Number of highest level iterations
NUM_ITERATIONS = 40

# Global seed for reproducibility across runs
SEED = 42


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    # This does not affect the current process hash seed, but keeps child processes consistent.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

# Number of self-play games to play within each iteration
NUM_SELF_PLAY_ITERATIONS = 200

# Number of mcts simulations when selecting a move within self-play
NUM_MCTS_SEARCHES = 60

# Number of epochs for training on self-play data for each iteration
NUM_EPOCHS = 4

# Batch size for training
BATCH_SIZE = 128

# Temperature for the softmax selection of moves
TEMPERATURE = 1.25

# The value of the constant policy
C = 2

# The value of the dirichlet alpha
DIRICHLET_ALPHA = 0.3

# The value of the dirichlet epsilon
DIRICHLET_EPSILON = 0.25

# The number of steps to look ahead for policy targets
K = 5

# The weight of the value loss in the total loss
VALUE_LOSS_WEIGHT = 0.25

# The maximum norm for gradient clipping
MAX_GRAD_NORM = 5


set_global_seed(SEED)
