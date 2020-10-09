import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

seed=1234


def feed_forward(x):
    print(x)


def main():
    X, y = make_classification(n_samples=100,
                               n_features=4,
                               n_clusters_per_class=1,
                               n_classes=2,
                               random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=seed
    )

    print(X_train, X_test, y_train, y_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    # x_train, y_train, x_val, y_val, x_test, y_test = split_()
    # network = feed_forward()

