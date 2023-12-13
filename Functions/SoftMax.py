import numpy as np


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    return exp_a / np.sum(exp_a)
