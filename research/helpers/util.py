import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def auc(y_true: np.ndarray, y_pred: np.ndarray, *, average=None, sample_weight=None,
        max_fpr=None, multi_class=None, labels: np.ndarray = None):
    lb = preprocessing.LabelBinarizer()
    lb.fit(labels)
    y_score = lb.transform(y_pred)
    try:
        return roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight, max_fpr=max_fpr,
                             multi_class=multi_class, labels=labels)
    except Exception as e:
        return 0.5


def to_latex(df: pd.DataFrame, **kwargs):
    tex = df.to_latex(**kwargs)
    tex = tex.replace('_', ' ').replace('\\toprule\n', '').replace('\\midrule\n', '').replace('\\bottomrule\n', '')

    lines = tex.split('\n')
    lines.insert(1, '\\footnotesize')
    lines.insert(6, '\\hline')
    lines.insert(8, '\\hline')
    lines.insert(-3, '\\hline')
    tex = '\n'.join(lines)

    tex = tex.replace('\\begin{table}', '\\begin{table}[H]') \
        .replace(' accuracy ', ' ACC ').replace(' auc ', ' AUC ').replace(' algorithm ', ' Algorithm ') \
        .replace(' precision ', ' Pre ').replace(' recall ', ' Sen ').replace(' f1 ', ' F1 ')
    return tex
