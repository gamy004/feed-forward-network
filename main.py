import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

from model import FeedForwardNetwork

seed = 1234

np.random.seed(seed)


def save(fp, output):
    with open(fp, "wb") as p:
        pickle.dump(output, p)


def sampling(X, Y, sample_frac=1):
    sample_X = pd.DataFrame(X).sample(frac=sample_frac)
    sample_index = sample_X.index.to_list()
    sample_Y = Y[sample_index]

    return [sample_X.to_numpy(), sample_Y]


def main():
    X, Y = make_multilabel_classification(n_samples=300,
                                          n_features=4,
                                          n_classes=1,
                                          n_labels=1,
                                          random_state=seed)

    n_input = X.shape[1]
    n_output = Y.shape[1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.33,
        random_state=seed
    )

    save("test_set.sav", [X_test, Y_test])

    X_train, X_validate, Y_train, Y_validate = train_test_split(
        X_train,
        Y_train,
        test_size=0.11,
        random_state=seed
    )

    save("train_set.sav", [X_train, Y_train])
    save("val_set.sav", [X_validate, Y_validate])

    try:
        with open("model.sav", "rb") as p:
            model = pickle.load(p)
    except FileNotFoundError:
        model = FeedForwardNetwork(n_input, n_output)
        model.add(2, activation='relu')
        model.add(1, activation='sigmoid')
        model.compile()

        model.fit(X_train, Y_train,
                  epochs=5000,
                  batch_size=32,
                  validation_data=(X_validate, Y_validate),
                  shuffle=True)

        model.save()


if __name__ == '__main__':
    main()
