import pickle
from time import sleep

import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

seed = 1234
num_hidden_units = 2

np.random.seed(seed)

class LossCalculator:
    @staticmethod
    def binary_crossentropy(y, y_h, W, lambd):
        n = y.shape[0]
        error_type_1 = np.multiply(y, np.log(y_h))
        error_type_2 = np.multiply((1 - y), np.log((1 - y_h)))

        error = (-1/n) * np.sum(error_type_1 + error_type_2)

        L2_regularization_cost = []

        for level in range(len(W)):
            L2_regularization_W = np.sum(np.square(W[level]))
            L2_regularization_cost.append(L2_regularization_W)

        total_L2_regularization_cost = np.sum(L2_regularization_cost)*(lambd/(2*n))

        total_error = np.squeeze(error + total_L2_regularization_cost)

        return total_error


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-1.0 * x))

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
        self.W.append(np.random.rand(y_dim, x_dim) * 0.01)
        self.B.append(np.random.rand(y_dim, 1) * 0.01)
        self.G_W.append(np.zeros(shape=(y_dim, x_dim)))
        self.G_B.append(np.zeros(shape=(y_dim, 1)))
        self.A.append(np.zeros(shape=(y_dim, 1)))
        self.Z.append(np.zeros(shape=(y_dim, 1)))
        self.D.append(np.zeros(shape=(y_dim, 1)))

        return self

    def fit(self, X, y, epochs=100, batch_size=None):
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                fit_X = X.copy()
                fit_y = y.copy()

                if batch_size is not None:
                    selected_idx = np.random.randint(fit_X.shape[0], size=batch_size)
                    fit_X = fit_X[selected_idx]
                    fit_y = fit_y[selected_idx]

                y_h = self.feed_forward(fit_X)
                training_loss = self.apply_loss_function(fit_y, y_h)
                self.L.append(training_loss)

                pbar.set_description("Training Loss: {}".format(training_loss))

                self.back_propagate(fit_X, fit_y)
                sleep(0.1)
                pbar.update(1)

        return self

    def predict(self, X):
        y_h = self.feed_forward(X)

        return y_h > self.predict_ratio

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


def q1(df, label="class"):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df[label])
    classes = label_encoder.classes_

    df_dict = {}

    for class_name in classes:
        class_df = df.copy()
        class_df[class_name] = 0

        labels = class_df[class_name].copy()
        labels.loc[class_df[label] == class_name] = 1
        class_df[class_name] = labels
        class_df.drop(label, axis=1, inplace=True)
        df_dict.update({
            class_name: class_df
        })

    return df_dict, classes


def main():
    # X, y = make_multilabel_classification(n_samples=500,
    #                                       n_features=4,
    #                                       n_classes=1,
    #                                       n_labels=1,
    #                                       random_state=seed)
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.33,
    #     random_state=seed
    # )

    df_dict, classes = q1(pandas.read_csv("./dataset.csv"), label="species")

    def save(fp, output):
        with open(fp, "wb") as p:
            pickle.dump(output, p)

    for class_name in classes[: 1]:
        df = df_dict.get(class_name)
        X = df.copy().drop(columns=[class_name]).to_numpy()
        Y = df.copy()[class_name].to_numpy()

    Y = Y.reshape(len(Y), 1)

    n_input = X.shape[1]
    n_output = Y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.33,
        random_state=seed
    )

    model = FeedForwardNetwork(n_input, n_output)
    model.add(2, activation='relu')
    model.add(1, activation='sigmoid')
    model.compile()

    model.fit(X_train, y_train, epochs=100, batch_size=24)

    predictions = model.predict(X_test)
    print(predictions)


if __name__ == '__main__':
    main()
