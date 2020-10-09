import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

seed = 1234
num_hidden_units= 2


def make_model(n_input, n_output):
    W = [np.random.rand(num_hidden_units, n_input), np.random.rand(n_output, num_hidden_units)]

    return W


def main():
    X, y = make_multilabel_classification(n_samples=10,
                                          n_features=4,
                                          n_classes=1,
                                          n_labels=1,
                                          random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=seed
    )

    n_input = X.shape[1]
    n_output = y.shape[1]

    model = make_model(n_input, n_output)


if __name__ == '__main__':
    main()
