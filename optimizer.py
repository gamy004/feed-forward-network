import numpy as np


class LossCalculator:
    @staticmethod
    def binary_crossentropy(y, y_h, W, lambd):
        n = y.shape[0]
        y_h = y_h.T
        error_type_1 = np.multiply(y, np.log(y_h))
        error_type_2 = np.multiply((1 - y), np.log((1 - y_h)))

        sum_error = np.sum(error_type_1 + error_type_2)
        error = (-1 / n) * sum_error

        L2_regularization_cost = []

        for level in range(len(W)):
            L2_regularization_W = np.sum(np.square(W[level]))
            L2_regularization_cost.append(L2_regularization_W)

        total_L2_regularization_cost = np.sum(L2_regularization_cost) * (lambd / (2 * n))

        total_error = np.squeeze(error + total_L2_regularization_cost)

        return total_error
