import numpy as np
import os
from matplotlib import rcdefaults
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from scipy.stats import ttest_rel


def mean_ranks(methods, streams, metrics, experiment_name, methods_alias=None, metrics_alias=None):

    streams.sort()
    rcdefaults()

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}
    mean_data = np.zeros(shape=(len(streams), len(methods), len(metrics)))

    for i, stream_name in enumerate(streams):
        for j, clf_name in enumerate(methods):
            for k, metric in enumerate(metrics):
                try:
                    filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                    data[stream_name, clf_name, metric] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    mean_data[i, j, k] = np.mean(np.genfromtxt(filename, delimiter=',', dtype=np.float32))
                except Exception:
                    data[stream_name, clf_name, metric] = None
                    print("Error in loading data", stream_name, clf_name, metric)

    ir_array = []
    nf_array = []
    ns_array = []
    nc_array = []

    for i, stream_name in enumerate(streams):
        data2, meta = arff.loadarff("datasets/%s.dat" % stream_name)
        main_df = pd.DataFrame(data2)
        new_df = main_df
        ind = 0
        for i in range(len(main_df.columns)-1):
            try:
                feat = main_df.iloc[:, i].values.astype(float)
            except ValueError:
                feat = main_df.iloc[:, i].values.astype(str)
                name = main_df.columns[i]
                le = preprocessing.LabelEncoder()
                feat = le.fit_transform(feat)
                ohe = preprocessing.OneHotEncoder(sparse=False)
                feat = ohe.fit_transform(feat.reshape(len(feat), 1))
                new_df = new_df.drop(columns=main_df.columns[i])
                for j in range(feat.shape[1]):
                    ind += 1
                    new_df.insert(ind-1, name+"_%s" % j, feat[:, j])

        X = new_df.iloc[:, 0:-1].values.astype(float)
        new_df.replace(b'positive', 1, inplace=True)
        new_df.replace(b'negative', 0, inplace=True)
        y = new_df.iloc[:, -1].values.astype(int)

        nc_array.append(len(main_df.iloc[:, -1].unique()))
        nf_array.append(main_df.shape[1])
        ns_array.append(len(main_df.index))

        unique, counts = np.unique(y, return_counts=True)
        if len(counts) == 1:
            raise ValueError("Only one class in procesed data. Use bigger data chunk")
        elif counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        ir_array.append(majority.shape[0]/minority.shape[0])

    for k, metric in enumerate(metrics):
        text = "\\begin{table}[ht]\n\\centering\n\\scalebox{0.5}{\n\\begin{tabular}{lcccccccc}\n\\toprule\n\\textbf{Dataset} & \\textbf{Features} & \\textbf{Samples} & \\textbf{Imb. Ratio} & "
        for id, m in enumerate(methods_alias):
            text += f" \\textbf{{{m}}} ({id+1}) &"
        text = text[:-1]
        text += " \\\\ \\toprule\n"

        for i, stream_name in enumerate(streams):
            best = np.argmax(mean_data[i, :, k])

            stream_name_ = stream_name.replace('_', '\\_')
            text += f"\\multirow{{2}}{{*}}{{\\textit{{{stream_name_}}}}}"
            text += f" & \\multirow{{2}}{{*}}{{{nf_array[i]:d}}}"
            text += f" & \\multirow{{2}}{{*}}{{{ns_array[i]:d}}}"
            text += f" & \\multirow{{2}}{{*}}{{{ir_array[i]:0.1f}}}"
            for j, (clf_name, method_a) in list(enumerate(zip(methods, methods_alias))):
                if data[stream_name, clf_name, metric] is None:
                    continue
                mn = np.mean(data[stream_name, clf_name, metric])
                std = np.std(data[stream_name, clf_name, metric])
                if best == j:
                    text += f" & \\textbf{{{mn:0.3f} $\\pm$ {std:0.3f}}}"
                else:
                    text += f" & {mn:0.3f} $\\pm$ {std:0.3f}"
            text += " \\\\\n & & & "
            for j, (clf_name1, method_a) in list(enumerate(zip(methods, methods_alias))):
                text += " & \\textit{"
                for h, (clf_name2, method_a) in list(enumerate(zip(methods, methods_alias))):
                    a = data[stream_name, clf_name1, metric]
                    b = data[stream_name, clf_name2, metric]
                    stat, pv = ttest_rel(a, b)
                    if stat > 0 and pv < 0.05:
                        text += f"{h+1} "
                text += "}"

            text += " \\\\ \\bottomrule\n"
        text += "\\end{tabular}}\n\\end{table}"

        if not os.path.exists("results/tables/"):
            os.makedirs("results/tables/")

        with open(f"results/tables/{metric}.tex", 'w+') as file:
            file.write(text)
