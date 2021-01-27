"""
inspired by 
https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53696
"""
from collections import Counter


import numpy as np
import pandas as pd

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

import xgboost as xgb

import lightgbm as lgbm

from hyperopt.pyll.base import scope

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}

def gini(truth, predictions):
    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth))], dtype=np.float)
    g = g[np.lexsort((g[:, 2], -1 * g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)


def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)


def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True


def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)


gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)

RFC_base = {
    'const_params': {
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    'space': {
        'n_estimators': scope.int(hp.quniform('n_estimators', 25, 500, 25)),
        'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1))
    }
}

xgb_base = {
    'const_params': {
        'n_jobs': -1,
        # 'learning_rate': 0.05
    },
    'space': {
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 50)),
        'learning_rate': hp.quniform('learning_rate', .01, .1, .01),
        'max_depth': scope.int(hp.quniform('max_depth', 2, 8, 1)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'gamma': hp.uniform('gamma', 0.0, 0.5),
    }
}

lgbm_base = {
    'const_params': {
    },
    'space': {
        'n_estimators': scope.int(hp.quniform('n_estimators', 32, 64, 8)),
        'learning_rate': hp.quniform('learning_rate', .01, .1, .01),
        'num_leaves': scope.int(hp.quniform('num_leaves', 8, 128, 16)),
        'colsample_bytree': hp.uniform('colsample_bytree', .1, .9),
    }
}


class ClassifierOpt:
    def __init__(self, classifier, score, const_params, space):
        self.classifier = classifier
        self.score = score
        self.const_params = const_params
        self.space = space

    def optimize(self, X, Y, max_evals=50):
        def objective(params):
            clf = self.classifier(**self.const_params, **params)
            score = cross_val_score(clf, X, Y, scoring=self.score, cv=StratifiedKFold(n_splits=5)).mean()
            print("Score {:.3f} params {}".format(score, params))
            return score

        best = fmin(fn=objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=max_evals)

        return best
