from evolution_strategy import evolution_strategy, reward_function
from .ann import ANN
from .env import Env
import numpy as np
import sys

env = Env()

# hyperparameters
HISTORY_LENGTH = 1
D = len(env.reset()) * HISTORY_LENGTH
M = 50  # hidden layer size
K = 2  # output size

if __name__ == '__main__':
    model = ANN(D, M, K)

    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        # play with a saved model
        j = np.load('es_flappy_results.npz')
        best_params = np.concatenate(
            [j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])

        # in case initial shapes are not correct
        D, M = j['W1'].shape
        K = len(j['b2'])
        model.D, model.M, model.K = D, M, K
    else:
        # train and save
        # env.set_display(True)
        model.init()
        params = model.get_params()
        best_params, rewards = evolution_strategy(
            f=reward_function,
            population_size=50,
            sigma=0.1,
            lr=0.03,
            initial_params=params,
            num_iters=300,
        )

        model.set_params(best_params)
        np.savez(
            'es_flappy_results.npz',
            train=rewards,
            **model.get_params_dict()
        )

    # play 5 test episodes
    env.set_display(True)
    for _ in range(5):
        print("Test:", reward_function(best_params))
