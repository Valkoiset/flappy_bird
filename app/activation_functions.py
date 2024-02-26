import numpy as np


def relu(x):
    return x * (x > 0)


def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    # subtracting the max of each column since we don't like exponentiating the large numbers
    e = np.exp(a - c)
    # then divide it by the sum of the exponentiated values to give us the model output probabilities
    return e / e.sum(axis=-1, keepdims=True)
