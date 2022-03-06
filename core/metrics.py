from tqdm import tqdm
import numpy as np
import os


def binary_confusion_matrix(y_true, y_pred):
    # tn, fp, fn, tp
    return tuple([np.sum((y_pred == i % 2) * (y_true == i // 2)) for i in range(4)])


def accuracy(tn, fp, fn, tp):
    return np.nan_to_num((tp+tn)/(tn+fp+fn+tp))


def recall(tn, fp, fn, tp):
    return np.nan_to_num(tp/(tp+fn))


def specificity(tn, fp, fn, tp):
    return np.nan_to_num(tn/(tn+fp))


def precision(tn, fp, fn, tp):
    return np.nan_to_num(tp/(tp+fp))


def f1_score(tn, fp, fn, tp):
    prc = precision(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num(2*(prc*rec)/(prc+rec))


def balanced_accuracy(tn, fp, fn, tp):
    spc = specificity(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num((1/2)*(spc+rec))


def g_mean(tn, fp, fn, tp):
    spc = specificity(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num(np.sqrt(spc*rec))


def mcc(tn, fp, fn, tp):
    return np.nan_to_num(((tp*tn)-(fp*fn))/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))


def calculate_metrics(methods, datasets, metrics, experiment_name, recount=False):
    data = {}
    metrics_func = {
                    "accuracy": accuracy,
                    "recall": recall,
                    "specificity": specificity,
                    "precision": precision,
                    "f1_score": f1_score,
                    "balanced_accuracy": balanced_accuracy,
                    "g_mean": g_mean,
                    "mcc": mcc,
    }

    for data_name in datasets:
        for clf_name in methods:
            try:
                filename = "results/raw_conf/%s/%s/%s.csv" % (experiment_name, data_name, clf_name)
                data[data_name, clf_name] = np.genfromtxt(filename, delimiter=',', dtype=np.int16)
            except Exception:
                data[data_name, clf_name] = None
                print("Error in loading data", "results/raw_conf/%s/%s/%s.csv" % (experiment_name, data_name, clf_name), clf_name)

    for data_name in tqdm(datasets, "Metrics %s" % experiment_name):
        for clf_name in methods:
            if data[data_name, clf_name] is None:
                continue
            for metric in metrics:
                if os.path.exists("results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, data_name, metric, clf_name)) and not recount:
                    continue
                tn = data[data_name, clf_name][:, 0]
                fp = data[data_name, clf_name][:, 1]
                fn = data[data_name, clf_name][:, 2]
                tp = data[data_name, clf_name][:, 3]

                result = metrics_func[metric](tn, fp, fn, tp)

                filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, data_name, metric, clf_name)

                if not os.path.exists("results/raw_metrics/%s/%s/%s/" % (experiment_name, data_name, metric)):
                    os.makedirs("results/raw_metrics/%s/%s/%s/" % (experiment_name, data_name, metric))

                np.savetxt(fname=filename, fmt="%f", X=result)
