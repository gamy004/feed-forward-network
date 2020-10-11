import pickle
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import sleep
from activation import Activation
from optimizer import LossCalculator


def save(fp, output):
    with open(fp, "wb") as p:
        pickle.dump(output, p)


class FeedForwardNetwork:
    def __init__(self, n_input, n_output, learning_rate=1e-2, regularization=1e-3, predict_ratio=0.5):
        self.n_input = n_input
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.predict_ratio = predict_ratio
        self.activation = Activation()
        self.loss_calculator = LossCalculator()

        self.W = []  # Weights
        self.B = []  # Biases
        self.A = []  # Biases + Weights*Input
        self.Z = []  # Activation(A)
        self.D = []  # Delta
        self.G_W = []  # Gradients of W
        self.G_B = []  # Gradients of B

        self.hidden = list([])
        self.loss = None
        self.L = []  # Loss history
        self.V_L = []  # Val Loss history
        self.V_ACC = []  # ValAccuracy history

    def add(self, n_units, activation='relu'):
        self.hidden.append([n_units, activation])

        return self

    def compile(self, loss='binary_crossentropy'):
        self.loss = loss

        self.init_weight()

        return self

    def init_weight(self):
        input_to_first_hidden = self.hidden[0]
        first_hidden_units = input_to_first_hidden[0]

        self.add_layer(self.n_input, first_hidden_units)

        for hidden_to_hidden in self.hidden[1:]:
            current_n_units = hidden_to_hidden[0]
            previous_B = self.B[-1]
            previous_n_units = previous_B.shape[0]

            self.add_layer(previous_n_units, current_n_units)

        return self

    def add_layer(self, x_dim, y_dim):
        self.W.append(np.random.rand(y_dim, x_dim))
        self.B.append(np.random.rand(y_dim, 1))
        self.G_W.append(np.zeros(shape=(y_dim, x_dim)))
        self.G_B.append(np.zeros(shape=(y_dim, 1)))
        self.A.append(np.zeros(shape=(y_dim, 1)))
        self.Z.append(np.zeros(shape=(y_dim, 1)))
        self.D.append(np.zeros(shape=(y_dim, 1)))

        return self

    def fit(self, X, y, epochs=100, batch_size=None, validation_data=None, shuffle=False):
        input_samples = X.shape[0]

        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                num_chunks = 1
                train_Xs = X.copy()
                train_ys = y.copy()

                if shuffle:
                    shuffle_df = pd.DataFrame(np.concatenate((X, y), axis=1)).sample(frac=1)
                    shuffle_df_columns = shuffle_df.columns.to_list()
                    shuffle_df_column_length = len(shuffle_df_columns)
                    train_Xs = shuffle_df[
                        shuffle_df_columns[0: shuffle_df_column_length - 1]
                    ].to_numpy()
                    train_ys = shuffle_df[
                        shuffle_df_columns[shuffle_df_column_length - 1: shuffle_df_column_length]
                    ].to_numpy()
                    del shuffle_df, shuffle_df_columns, shuffle_df_column_length

                if batch_size is not None:
                    num_chunks = math.ceil(input_samples / batch_size)

                train_Xs = np.array_split(train_Xs, num_chunks)
                train_ys = np.array_split(train_ys, num_chunks)

                for train_X, train_y in zip(train_Xs, train_ys):
                    y_h = self.feed_forward(train_X)

                    training_loss = self.apply_loss_function(train_y, y_h)
                    self.L.append(training_loss)

                    self.back_propagate(train_X, train_y)

                    message = "Training Loss: {}".format(training_loss)

                    if validation_data is not None:
                        val_loss, val_accuracy = self.evaluate(validation_data[0], validation_data[1])
                        self.V_L.append(val_loss)
                        self.V_ACC.append(val_accuracy)

                        message = "{}, Validation Loss: {}, Validation Accuracy: {}".format(
                            message,
                            val_loss,
                            val_accuracy
                        )

                    pbar.set_description(message)

                sleep(0.1)
                pbar.update(1)

        return self

    def predict(self, X):
        y_h = self.feed_forward(X)

        return y_h > self.predict_ratio

    def evaluate(self, X, y):
        y_h = self.feed_forward(X)
        val_loss = self.apply_loss_function(y, y_h)

        prediction = y_h > self.predict_ratio
        prediction = prediction.T.astype('int8')

        compare = prediction == y
        total = len(compare)
        correct = compare.sum()
        accuracy = round(correct / total, 2)

        return val_loss, accuracy

    def save(self, name="model"):
        save("{}.sav".format(name), self)

        return self

    def feed_forward(self, X):
        for level in range(len(self.hidden)):
            hidden_info = self.hidden[level]
            h_activation = hidden_info[1]
            h_W = self.W[level]
            h_B = self.B[level]

            if level == 0:
                prev_Z = X.copy()
            else:
                prev_Z = self.Z[level - 1]

            h_Z = h_W.dot(prev_Z.T) + h_B
            h_A = self.apply_activation(h_Z, h_activation)

            self.Z[level] = h_Z.T
            self.A[level] = h_A

        y_h = self.A[-1]

        return y_h

    def back_propagate(self, X, y):
        output_level = len(self.hidden) - 1
        last_hidden_level = output_level - 1

        self.calculate_output_gradient(y)

        for level in range(last_hidden_level, -1, -1):
            self.calculate_hidden_gradient(level, X)

        self.update_gradient()

        return self

    def calculate_output_gradient(self, y):
        n = y.shape[0]
        output_level = len(self.hidden) - 1

        o_W = self.W[output_level]
        o_A = self.A[output_level]
        p_A = self.A[output_level - 1]

        d_Z = o_A - y.T
        d_B = (1 / n) * np.sum(d_Z, axis=1, keepdims=True)
        d_W = (1 / n) * np.dot(d_Z, p_A.T) + (self.regularization/n)*o_W

        self.D[output_level] = d_Z
        self.G_B[output_level] = d_B
        self.G_W[output_level] = d_W

        return self

    def calculate_hidden_gradient(self, hidden_level, X):
        n = X.shape[0]
        next_level = hidden_level+1
        hidden_info = self.hidden[hidden_level]
        hidden_activation = hidden_info[1]

        n_D = self.D[next_level]
        n_W = self.W[next_level]

        if hidden_level == 0:
            p_A = X.copy()
        else:
            p_A = self.A[hidden_level - 1]

        h_Z = self.Z[hidden_level]
        h_W = self.W[hidden_level]
        a_Z = self.apply_derivative_activation(h_Z, hidden_activation)
        D = np.dot(n_W.T, n_D)

        d_Z = np.multiply(D, a_Z.T)
        d_B = (1 / n) * np.sum(d_Z, axis=1, keepdims=True)
        d_W = (1 / n) * np.dot(d_Z, p_A) + (self.regularization/n)*h_W

        self.D[hidden_level] = d_Z
        self.G_B[hidden_level] = d_B
        self.G_W[hidden_level] = d_W

        return self

    def update_gradient(self):
        for level in range(len(self.hidden)):
            W = self.W[level]
            G_W = self.G_W[level]
            B = self.B[level]
            G_B = self.G_B[level]

            d_W = W - self.learning_rate * G_W
            d_B = B - self.learning_rate * G_B

            self.W[level] = d_W
            self.B[level] = d_B

        return self

    def apply_activation(self, x, activation):
        return getattr(self.activation, activation)(x)

    def apply_derivative_activation(self, x, activation):
        return getattr(self.activation, 'derivative_{}'.format(activation))(x)

    def apply_loss_function(self, y, y_h):
        return getattr(self.loss_calculator, self.loss)(y, y_h, self.W, self.regularization)
