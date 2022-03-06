from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.svm import SVC


class SC(BaseEstimator, ClassifierMixin):
    def __init__(self, oversampling, classifier=SVC()):
        self.oversampling = oversampling
        self.classifier = classifier

    def fit(self, X, y):
        res_X, res_y = self.oversampling().fit_resample(X, y)

        self.classifier.fit(res_X, res_y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)
