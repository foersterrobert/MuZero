args = {
            'num_iterations': 48,             # number of highest level iterations
            'num_train_games': 500,           # number of self-play games to play within each iteration
            'num_simulation_games': 800,      # number of mcts simulations when selecting a move within self-play
            'num_epochs': 4,                  # number of epochs for training on self-play data for each iteration
            'batch_size': 128,                # batch size for training
            'temperature': 1,                 # temperature for the softmax selection of moves
            'c1': 1.25,                       # the value of the constant policy
            'c2': 19652,                      # the value of the constant policy
            'discount': 0.997
        }