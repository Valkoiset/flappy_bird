from .activation_functions import relu, softmax
import numpy as np


class ANN:
    def __init__(self, D, M, K, f=relu):
        self.D = D
        self.M = M
        self.K = K
        self.f = f

    # initialize neural network's weights
    def init(self):
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) / np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M)
        self.b2 = np.zeros(K)

    def forward(self, X):
        """Returns a list of probabilities."""
        Z = self.f(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2)

    def sample_action(self, x):
        # assume input is a single state of size (D,)
        # first make it (N, D) to fit ML conventions
        X = np.atleast_2d(x)
        P = self.forward(X)
        p = P[0]
        return np.argmax(p)

    def get_params(self):
        """Returns all parameters of a neural network as a 1D array."""
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        """Returns a dictionary of all the neural network's weights."""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        }

    def set_params(self, params):
        """
        Takes 1D array of parameters, shapes them back into neural network weights
        and then assigns them to the neural network.
        """
        D, M, K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M:D * M + M]
        self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]
