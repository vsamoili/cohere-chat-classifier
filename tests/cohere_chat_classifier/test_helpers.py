from unittest.mock import MagicMock

import pandas as pd

from theydo.data_model import Dataset
from theydo.helpers import make_dataset_from_request_data, make_dataset_from_chat_response, make_dataset_from_clf_data


def test_make_dataset_from_clf_data():
    # Create mock objects for Classifications
    mock_classifications_1 = MagicMock()
    mock_classifications_1.input = 'text1'
    mock_classifications_1.prediction = 'label1'

    mock_classifications_2 = MagicMock()
    mock_classifications_2.input = 'text2'
    mock_classifications_2.prediction = 'label2'

    # List of mocked Classifications objects
    clf_data = [mock_classifications_1, mock_classifications_2]

    # Call the function under test
    dataset = make_dataset_from_clf_data(clf_data)

    # Create expected DataFrame for comparison
    expected_df = pd.DataFrame({
        'text': ['text1', 'text2'],
        'label': ['label1', 'label2']
    })

    # Verify DataFrame structure and content
    pd.testing.assert_frame_equal(dataset.df, expected_df)
    assert isinstance(dataset, Dataset)


def test_make_dataset_from_request_data():
    request_data = [('text1', 'label1'), ('text2', 'label2')]

    dataset = make_dataset_from_request_data(request_data)

    # Verify DataFrame structure and content
    assert list(dataset.df.columns) == ['text', 'label']
    assert all(dataset.df['text'] == ['text1', 'text2'])
    assert all(dataset.df['label'] == ['label1', 'label2'])
    assert isinstance(dataset, Dataset)


def test_make_dataset_from_chat_response():
    json_response = [{'text': 'text1', 'label': 'label1'}, {'text': 'text2', 'label': 'label2'}]

    dataset = make_dataset_from_chat_response(json_response)

    # Verify DataFrame structure and content
    assert list(dataset.df.columns) == ['text', 'label']
    assert all(dataset.df['text'] == ['text1', 'text2'])
    assert all(dataset.df['label'] == ['label1', 'label2'])
    assert isinstance(dataset, Dataset)
