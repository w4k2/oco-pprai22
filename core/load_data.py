import pandas as pd
from scipy.io import arff
from sklearn import preprocessing


def load_csv(filename):

    data, meta = arff.loadarff(filename)
    main_df = pd.DataFrame(data)

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

    return X, y
