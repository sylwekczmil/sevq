from functools import partial

import numpy as np
import xgboost as xgb
from evq.algorithm import EVQ
from neupy.algorithms.competitive.lvq import LVQ, LVQ2, LVQ21, LVQ3
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier, \
    AdditiveExpertEnsembleClassifier, OzaBaggingClassifier, \
    DynamicWeightedMajorityClassifier
from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, \
    ExtremelyFastDecisionTreeClassifier

from sevq.algorithm import SEVQ

l1_norm = partial(np.linalg.norm, ord=1, axis=-1)


class SimplifiedFuzzyARTMAP:
    """
    Simplified Fuzzy ARTMAP algorithm
    https://github.com/AIOpenLab/art/blob/master/fuzzy_art.py

    """

    def __init__(self, alpha=0., rho=0.9, gamma=0.0001, epsilon=0.0001,
                 complement_coding=True):
        """

        :param alpha: (default=0 - fast learning)
        :param rho: vigilance
        """
        self.alpha = alpha
        self.beta = 1 - alpha  # learning rate
        self.gamma = gamma
        self.rho = rho  # vigilance
        self.epsilon = epsilon
        self.complement_coding = complement_coding

        self.w = None
        self.out_w = None
        self.n_classes = 0

    def _init_weights(self, x, y):
        self.w = np.atleast_2d(x)
        self.out_w = np.zeros((1, self.n_classes))
        self.out_w[0, y] = 1

    def _complement_code(self, x):
        if self.complement_coding:
            return np.hstack((x, 1 - x))
        else:
            return x

    def _add_category(self, x, y):
        self.w = np.vstack((self.w, x))
        self.out_w = np.vstack((self.out_w, np.zeros(self.n_classes)))
        self.out_w[-1, y] = 1

    def _match_category(self, x, y=None):
        _rho = self.rho
        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm + (1 - self.gamma) * (l1_norm(x) + l1_norm(self.w))
        norms = fuzzy_norm / l1_norm(x)

        threshold = norms >= _rho
        while not np.all(threshold == False):
            y_ = np.argmax(scores * threshold.astype(int))

            if y is None or self.out_w[y_, y] == 1:
                return y_
            else:
                _rho = norms[y_] + self.epsilon
                norms[y_] = 0
                threshold = norms >= _rho
        return -1

    def fit(self, x, y, epochs=10):
        samples = self._complement_code(np.atleast_2d(x))
        self.n_classes = len(set(y))

        if self.w is None:
            self._init_weights(samples[0], y[0])

        idx = np.arange(len(samples), dtype=np.uint32)

        for epoch in range(epochs):
            idx = np.random.permutation(idx)
            for sample, label in zip(samples[idx], y[idx]):
                category = self._match_category(sample, label)
                if category == -1:
                    self._add_category(sample, label)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
        return self

    def predict(self, x):
        samples = self._complement_code(np.atleast_2d(x))

        labels = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            category = self._match_category(sample)
            labels[i] = np.argmax(self.out_w[category])
        return labels


class XGBoost:

    def __init__(self, max_depth=5):
        self.bst = None
        self.max_depth = max_depth

    def fit(self, x, y, param=None, feature_names=None):
        if param is None:
            param = {
                'max_depth': self.max_depth,  # the maximum depth of each tree
                'objective': 'multi:softprob',  # error evaluation for multiclass training
                'num_class': len(set(y))
            }
        dtrain = xgb.DMatrix(x, label=y, feature_names=feature_names)
        self.bst = xgb.train(param, dtrain)

    def predict(self, x):
        return np.argmax(self.bst.predict(xgb.DMatrix(x)), axis=1)


CLASSIFIERS = [
    ('SEVQ', lambda ni, nc: SEVQ()),
    ('SVC', lambda ni, nc: SVC()),
    ('XGB', lambda ni, nc: XGBoost()),
    ('NC', lambda ni, nc: NearestCentroid()),
    ('KNN', lambda ni, nc: KNeighborsClassifier(3)),
    ('DT', lambda ni, nc: DecisionTreeClassifier(max_depth=5)),
    ('RF', lambda ni, nc: RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ('MLP', lambda ni, nc: MLPClassifier(alpha=1, max_iter=10000)),
    ('AB', lambda ni, nc: AdaBoostClassifier()),
    ('GNB', lambda ni, nc: GaussianNB()),
    ('QDA', lambda ni, nc: QuadraticDiscriminantAnalysis())
]

CLASSIFIERS_FOR_NORMALIZED = [
    ('SEVQ', lambda ni, nc: SEVQ()),
    ('EVQ', lambda n_inputs, n_classes: EVQ(number_of_classes=n_classes, vigilance=0.2)),
    ('LVQ', lambda ni, nc: LVQ(n_inputs=ni, n_classes=nc, shuffle_data=True, step=0.01)),
    ('LVQ2', lambda ni, nc: LVQ2(n_inputs=ni, n_classes=nc, shuffle_data=True, step=0.01)),
    ('LVQ2.1', lambda ni, nc: LVQ21(n_inputs=ni, n_classes=nc, shuffle_data=True, step=0.01)),
    ('LVQ3', lambda ni, nc: LVQ3(n_inputs=ni, n_classes=nc, shuffle_data=True, step=0.001)),  # 0.01 has low results
    ('SFAM', lambda ni, nc: SimplifiedFuzzyARTMAP()),
    ('HT', lambda ni, nc: HoeffdingTreeClassifier()),
    ('HAT', lambda ni, nc: HoeffdingAdaptiveTreeClassifier()),
    ('EFDT', lambda ni, nc: ExtremelyFastDecisionTreeClassifier()),
    ('DWM', lambda ni, nc: DynamicWeightedMajorityClassifier()),
    ('NB', lambda ni, nc: NaiveBayes()),
    ('KNNI', lambda ni, nc: KNNClassifier()),
    ('ARF', lambda ni, nc: AdaptiveRandomForestClassifier()),
    ('AEE', lambda ni, nc: AdditiveExpertEnsembleClassifier()),
    ('OB', lambda ni, nc: OzaBaggingClassifier())
]
