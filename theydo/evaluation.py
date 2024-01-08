from typing import Literal

import numpy as np
from cohere.responses import Classifications
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, accuracy_score

from theydo.data_model import Dataset


METRICS = {
    'acc': accuracy_score,
    'prec': precision_score,
    'rec': recall_score,
    'f1': f1_score,
    # 'auc': roc_auc_score
}


def cast_pred_label_as_numeric(label: Literal['positive', 'negative']) -> int:
    """
    Convert a string label to a numeric label.

    Takes a string label ('positive' or 'negative') and converts it into a numeric format.
    'positive' is mapped to 1 and 'negative' is mapped to 0.

    :param label: The label to convert. Should be either 'positive' or 'negative'.
    :return: The numeric representation of the label. 1 for 'positive' and 0 for 'negative'.
    """
    return int(label == 'positive')


def prepare_ys(pred_data: Classifications, eval_data: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare the true and predicted labels for evaluation.

    This function takes the predicted data and evaluation data, and converts them into numpy arrays
    of true labels (y_true) and predicted labels (y_pred), which are suitable for use in further
    metric calculations.

    :param pred_data: Predicted data as returned from the Cohere model.
    :param eval_data: Evaluation data containing the true labels.
    :return: A tuple containing two numpy arrays, the true labels and the predicted labels.
    """
    y_pred = np.asarray([
        cast_pred_label_as_numeric(item.prediction)
        for item in pred_data
    ])
    y_true = np.asarray(eval_data.numeric_labels)

    return y_true, y_pred


def calculate_metric(pred_data: Classifications, eval_data: Dataset, metric: str) -> float:
    """
    Calculate a specified metric based on the prediction and evaluation data.

    This function computes a specified metric (such as accuracy, precision, recall, or F1 score)
    using the provided predicted and true labels.

    :param pred_data: Predicted data containing the predictions.
    :param eval_data: Evaluation data containing the true labels.
    :param metric: The metric to calculate. Options are 'acc', 'prec', 'rec', 'f1'.
    :return: The calculated metric value.
    """
    y_true, y_pred = prepare_ys(pred_data, eval_data)
    func = METRICS[metric]
    return func(y_true=y_true, y_pred=y_pred)


def get_classification_report(pred_data: Classifications, eval_data: Dataset) -> str:
    """
    Generate a classification report for the prediction and evaluation data.

    This function creates a detailed classification report, which includes metrics such as
    precision, recall, F1-score, and support for each class.

    :param pred_data: Predicted data containing the predictions.
    :param eval_data: Evaluation data containing the true labels.
    :return: A text report showing the main classification metrics.
    """
    y_true, y_pred = prepare_ys(pred_data, eval_data)
    return classification_report(y_true=y_true, y_pred=y_pred)
