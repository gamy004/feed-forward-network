import numpy as np


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    @staticmethod
    def derivative_sigmoid(x):
        return np.multiply(x, (1.0 - x))

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def derivative_relu(x):
        x[x <= 0] = 0
        x[x > 0] = 1

        return x
