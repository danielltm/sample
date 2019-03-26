import typing as tp
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, RegressorMixin

CLF = tp.Union[ClassifierMixin, ClusterMixin, RegressorMixin]

class CLfSwitcher(BaseEstimator):
    """
    A Custom Base Estimator
    """

    def __init__(self, estimator: CLF=None):
        """
        :param estimator: sklern classifier, cluster, regression
        """
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predic(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight)