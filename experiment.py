from core import eval_cv
import os
from methods import OCO, SC
from core import calculate_metrics
from core import mean_ranks
from sklearn.svm import SVC
from imblearn import over_sampling


random_state = 1234

methods = {
    "OCO": SC(OCO),
    "BSMOTE": SC(over_sampling.BorderlineSMOTE),
    "SMOTE": SC(over_sampling.SMOTE),
    "ROS": SC(over_sampling.RandomOverSampler),
    "NO": SVC(),
}

experiment_name = "simple"

dir = "./datasets/"
datasets = ["%s%s" % (dir, file) for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]
datasets.sort()

eval_cv(methods, datasets, random_state, experiment_name, n_splits=2, n_repeats=8, n_jobs=-1)

dir = "./results/raw_conf/%s/" % experiment_name
datasets = ["%s" % (file) for file in os.listdir(dir) if not os.path.isfile(os.path.join(dir, file))]

methods = {
    "OCO": SC(OCO),
    "BSMOTE": SC(over_sampling.BorderlineSMOTE),
    "SMOTE": SC(over_sampling.SMOTE),
    "ROS": SC(over_sampling.RandomOverSampler),
    "NO": SVC(),
}

metrics = {
    "g_mean":        "Gmean$_s$",
}

calculate_metrics(methods.keys(), datasets, metrics.keys(), experiment_name, recount=True)
mean_ranks(methods.keys(), datasets, metrics.keys(), experiment_name, metrics_alias=metrics.values())
