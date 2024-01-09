import pandas as pd
from cohere.responses import Classifications

from theydo.data_model import Dataset


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


def make_dataset_from_chat_response(json_response: list[dict[str, str]]) -> Dataset:
    """
    Create a Dataset object from a JSON chat response.

    This function takes a JSON response typically obtained from a chat interface or a similar API.
    The JSON response is expected to be a list of dictionaries, with each dictionary containing at least
    two keys: 'text' and 'label'. The 'text' key should map to the input text, and the 'label' key should
    map to its associated label. The function converts this JSON response into a pandas DataFrame and then
    uses it to create a Dataset object.

    :param json_response: The JSON response data to be converted into a dataset. Each dictionary in the
                          list should contain 'text' and 'label' keys.
    :return: A Dataset object containing the data from the JSON response.
    """
    df = pd.DataFrame({
        'text': [item['text'] for item in json_response],
        'label': [item['label'] for item in json_response]
    })

    return Dataset(df=df)
