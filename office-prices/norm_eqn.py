import io
import sys

import numpy as np

from itertools import combinations_with_replacement


def prepend_column_of_ones(m):
    return np.hstack((np.ones((m.shape[0], 1)), m))


def create_polynomial_features(m, grade):
    new_features = []
    idx = list(range(m.shape[1]))
    for r in range(1, grade + 1):
        for new_feature in combinations_with_replacement(idx, r):
            new_features.append(new_feature)

    new_matrix = []
    for new_feature in new_features:
        new_feature_matrix = np.prod(m[:, new_feature], axis=1)[np.newaxis].T
        new_matrix.append(new_feature_matrix)

    return np.hstack(new_matrix)


# noinspection PyPep8Naming, PyShadowingNames
def norm_eqn(X, y, lam=1e-4):
    L = np.eye(X.shape[1])
    L[0, 0] = 0

    # noinspection PyTypeChecker
    return np.linalg.solve(X.T.dot(X) + lam * L, X.T.dot(y))


# noinspection PyShadowingNames
def error(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


if __name__ == '__main__':
    sys.stdin = io.FileIO("input.txt", "r")

    # F -- the number of observed features
    # N -- the number of train rows
    F, N = map(int, next(sys.stdin).strip().split())
    train = []
    for _ in range(N):
        train.append(list(map(float, next(sys.stdin).strip().split())))

    # T -- the number of test rows
    T = int(next(sys.stdin).strip())
    test = []
    # noinspection PyRedeclaration
    for _ in range(T):
        test.append(list(map(float, next(sys.stdin).strip().split())))

    train = np.array(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_train = create_polynomial_features(X_train, grade=3)
    X_test = create_polynomial_features(np.array(test), grade=3)

    X_train = prepend_column_of_ones(X_train)
    X_test = prepend_column_of_ones(X_test)

    theta = norm_eqn(X_train, y_train)
    y_pred = X_test.dot(theta)

    with open("output.txt", "r") as fp:
        y_test = np.array(list(map(float, fp.read().split())))

    print("error: {}".format(error(y_test, y_pred)))

    for y in y_pred:
        print(y)
