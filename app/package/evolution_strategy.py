from datetime import datetime
from .env import Env
from .ann import ANN
import numpy as np

# when HL = 1 it means that state is equal to a current observation
# when HL > 1 then we concatenate current observation with the previous one to obtain the state
HISTORY_LENGTH = 1

# make a global environment to be used throughout the script
env = Env()

# hyperparameters
# input size = dimensionality of the data
D = len(env.reset()) * HISTORY_LENGTH
M = 50  # hidden layer size
K = 2  # output size


def evolution_strategy(
        f,  # f = reward_function
        population_size,
        sigma,  # noise standard deviation that gets added to the parameter for each offspring
        lr,
        initial_params,
        num_iters):
    # assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)

    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)  # generates the noise

        R = np.zeros(population_size)  # stores the reward

        # loop through each "offspring"
        for j in range(population_size):
            params_try = params + sigma * N[j]  # N[j] = noise for this offspring
            R[j] = f(params_try)

        m = R.mean()
        s = R.std()
        if s == 0:
            # we can't apply the following equation
            print("Skipping")
            continue

        A = (R - m) / s  # standardizing reward
        reward_per_iteration[t] = m
        params = params + lr / (population_size * sigma) * np.dot(N.T, A)

        # update the learning rate
        lr *= 0.992354
        # sigma *= 0.99

        print("Iter:", t, "Avg Reward: %.3f" % m, "Max:",
              R.max(), "Duration:", (datetime.now() - t0))

    return params, reward_per_iteration


# using a neural network policy to play episode of the game and return the reward
def reward_function(params):
    model = ANN(D, M, K)
    model.set_params(params)

    # play one episode and return the total reward
    episode_reward = 0
    episode_length = 0
    done = False
    obs = env.reset()
    obs_dim = len(obs)
    if HISTORY_LENGTH > 1:
        state = np.zeros(HISTORY_LENGTH * obs_dim)  # current state
        state[-obs_dim:] = obs
    else:
        state = obs
    while not done:
        # get the action
        action = model.sample_action(state)

        # perform the action
        obs, reward, done = env.step(action)

        # update total reward
        episode_reward += reward
        episode_length += 1

        # update state
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:] = obs
        else:
            state = obs
    return episode_reward
