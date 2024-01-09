import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from theydo.data_model import Dataset


METRICS = {
    'accuracy_avg': accuracy_score,
    'precision_avg': precision_score,
    'recall_avg': recall_score,
    'f1_avg': f1_score
}


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
    func = METRICS[metric]
    return func(y_true=y_true, y_pred=y_pred)


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
