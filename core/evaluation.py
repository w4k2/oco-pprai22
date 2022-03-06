from sklearn.model_selection import RepeatedStratifiedKFold
from .metrics import binary_confusion_matrix
from .load_data import load_csv
import numpy as np
import os

from joblib import Parallel, delayed


def eval_cv(methods, datasets, random_state, experiment_name, n_splits=2, n_repeats=8, n_jobs=-1):

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for data_path in datasets:
        print(data_path)
        X, y = load_csv(data_path)
        for method_name, method in zip(methods.keys(), methods.values()):

            directory = "results/raw_conf/"+experiment_name+"/"+data_path.split("/")[2].split(".")[0]
            filepath = directory+"/%s.csv" % method_name

            out = Parallel(n_jobs=n_jobs)(delayed(compute)(method, X, y, train_index, test_index) for train_index, test_index in rskf.split(X, y))
            conf_mats = [d['conf_mat'] for d in out]

            if not os.path.exists(directory):
                os.makedirs(directory)
            np.savetxt(fname=filepath, fmt="%d, %d, %d, %d", X=conf_mats)


def compute(method, X, y, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    method.fit(X_train, y_train)
    y_pred = method.predict(X_test)
    conf_mat = binary_confusion_matrix(y_test, y_pred)
    return dict(conf_mat=conf_mat)
