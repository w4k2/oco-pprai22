import numpy as np
from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt


class OCO():
    def __init__(self):
        pass

    def fit_resample(self, X, y):
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        min_X = X[y == 1, :]
        min_y = y[y == 1]

        maj_X = X[y == 0, :]

        ocsvm = OneClassSVM(nu=0.1, tol=0.01, shrinking=False)
        ocsvm.fit(min_X, min_y)

        mean_min = min_X.mean(axis=0)
        std_min = min_X.std(axis=0)
        min_gauss = np.random.normal(loc=mean_min, scale=std_min, size=(self.n_samples, self.n_features))

        y_pred = ocsvm.decision_function(min_gauss)
        new_min = min_gauss[y_pred > 0]

        res_min_X = np.concatenate((min_X, new_min), axis=0)

        res_X = np.concatenate((res_min_X, maj_X), axis=0)
        res_y = len(res_min_X)*[1] + len(maj_X)*[0]

        return(res_X, res_y)
