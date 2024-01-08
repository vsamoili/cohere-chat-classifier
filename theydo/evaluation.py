from typing import Literal

import numpy as np
import pandas as pd
from cohere.responses import Classifications
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, accuracy_score

from theydo.data_model import Dataset


METRICS = {
    'accuracy_avg': accuracy_score,
    'precision_avg': precision_score,
    'recall_avg': recall_score,
    'f1_avg': f1_score,
    # 'auc': roc_auc_score
}


def make_dataset_from_clf_data(clf_data: Classifications) -> Dataset:
    """
    Create a Dataset object from classification data.

    This function takes classification data (Classifications) and converts it into a pandas DataFrame,
    which is then used to create a Dataset object. The DataFrame contains two columns: 'text' and 'label',
    where 'text' is the input text and 'label' is the classification prediction.

    :param clf_data: The classification data to be converted into a dataset.
    :return: A Dataset object containing the data from the classifications.
    """

    df = pd.DataFrame({
        'text': [item. input for item in clf_data],
        'label': [item.prediction for item in clf_data]
    })

    return Dataset(df=df)


def make_dataset_from_request_data(request_data: list[tuple[str, str]]) -> Dataset:
    """
    Create a Dataset object from request data.

    This function takes request data in the form of a list of tuples. Each tuple contains two elements:
    the input text and its associated label. The data is converted into a pandas DataFrame with two
    columns ('text' and 'label') and then used to create a Dataset object.

    :param request_data: The request data to be converted into a dataset. Each tuple in the list should
                         contain two strings: the text and its corresponding label.
    :return: A Dataset object containing the data from the request.
    """

    df = pd.DataFrame({
        'text': [item[0] for item in request_data],
        'label': [item[1] for item in request_data]
    })

    return Dataset(df=df)


def prepare_ys(pred_data: Dataset, eval_data: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare the true and predicted labels for evaluation.

    This function takes the predicted data and evaluation data, and converts them into numpy arrays
    of true labels (y_true) and predicted labels (y_pred), which are suitable for use in further
    metric calculations.

    :param pred_data: Predicted data as returned from the Cohere model.
    :param eval_data: Evaluation data containing the true labels.
    :return: A tuple containing two numpy arrays, the true labels and the predicted labels.
    """

    y_pred = np.asarray(pred_data.numeric_labels)
    y_true = np.asarray(eval_data.numeric_labels)

    return y_true, y_pred


def calculate_metric(pred_data: Dataset, eval_data: Dataset, metric: str) -> float:
    """
    Calculate a specified metric based on the prediction and evaluation data.

    This function computes a specified metric (such as mean accuracy, mean precision, mean recall, or mean F1 score)
    using the provided predicted and true labels.

    :param pred_data: Predicted data containing the predictions.
    :param eval_data: Evaluation data containing the true labels.
    :param metric: The metric to calculate. See METRICS keys for options.
    :return: The calculated metric value.
    """

    y_true, y_pred = prepare_ys(pred_data, eval_data)
    print(y_true, y_pred)
    func = METRICS[metric]
    return func(y_true=y_true, y_pred=y_pred)


def get_classification_report(pred_data: Dataset, eval_data: Dataset) -> str:
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


def calculate_all(pred_data: Dataset, eval_data: Dataset) -> dict[str, float]:
    """
    Calculate all specified metrics for the given prediction and evaluation data.

    This function iterates through all the metrics defined in the METRICS dictionary and computes
    each metric using the provided predicted and evaluation data. The results are returned as a
    dictionary where keys are the metric names and values are the corresponding metric scores.

    :param pred_data: Predicted data containing the predictions.
    :param eval_data: Evaluation data containing the true labels.
    :return: A dictionary containing metric names as keys and their computed scores as values.
    """

    return {
        metric_name: calculate_metric(pred_data=pred_data, eval_data=eval_data, metric=metric_name)
        for metric_name in METRICS.keys()
    }
