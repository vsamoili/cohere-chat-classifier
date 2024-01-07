from typing import Literal

import numpy as np
from cohere.responses import Classifications
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report,accuracy_score

from theydo.data_model import Dataset


METRICS = {
    'prec': precision_score,
    'rec': recall_score,
    'f1': f1_score,
    'auc': roc_auc_score
}


def cast_pred_label_as_numeric(label: Literal['positive', 'negative']):
    return int(label == 'positive')


def prepare_ys(pred_data: Classifications, eval_data: Dataset):
    y_pred = np.asarray([
        cast_pred_label_as_numeric(item.prediction)
        for item in pred_data
    ])
    y_true = np.asarray(eval_data.numeric_labels)

    return y_true, y_pred


def calculate_metric(pred_data: Classifications, eval_data: Dataset, metric: str):
    y_true, y_pred = prepare_ys(pred_data, eval_data)
    func = METRICS[metric]
    return func(y_true=y_true, y_pred=y_pred)


def print_classification_report(pred_data: Classifications, eval_data: Dataset):
    y_true, y_pred = prepare_ys(pred_data, eval_data)
    return classification_report(y_true=y_true, y_pred=y_pred)
