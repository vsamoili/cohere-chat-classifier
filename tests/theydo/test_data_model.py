from unittest.mock import patch

import pandas as pd
import pytest

from theydo.exceptions import LabelFormatError
from theydo.helpers import *
from theydo.data_model import Dataset


def test_dataset_initialization_with_df():
    df = pd.DataFrame({'text': ['sample1'], 'label': [1]})
    dataset = Dataset(df=df)
    assert dataset.df.equals(df)


def test_dataset_initialization_without_df():
    dataset = Dataset()
    assert dataset.df is None


@patch('theydo.data_model.pd.read_parquet')
def test_load(mock_read_parquet):
    mock_df = pd.DataFrame({'text': ['sample1'], 'label': [1]})
    mock_read_parquet.return_value = mock_df
    dataset = Dataset(full_dir='path/to/data')
    dataset.load()
    assert dataset.df.equals(mock_df)
    mock_read_parquet.assert_called_with('path/to/data', engine='pyarrow')


@pytest.fixture
def sample_dataset():
    df = pd.DataFrame({'text': ['text1', 'text2'], 'label': [0, 1]})
    return Dataset(df=df)


def test_text_property(sample_dataset):
    expected_series = pd.Series(['text1', 'text2'], name='text')
    pd.testing.assert_series_equal(sample_dataset.text, expected_series)


def test_labels_property_numeric(sample_dataset):
    sample_dataset.df['label'] = [0, 1]
    expected_series = pd.Series(['negative', 'positive'], name='label')
    pd.testing.assert_series_equal(sample_dataset.labels, expected_series)


def test_numeric_labels_property(sample_dataset):
    sample_dataset.df['label'] = ['negative', 'positive']
    expected_series = pd.Series([0, 1], name='label')
    pd.testing.assert_series_equal(sample_dataset.numeric_labels, expected_series)


def test_labels_property_string(sample_dataset):
    sample_dataset.df['label'] = ['negative', 'positive']
    expected_series = pd.Series(['negative', 'positive'], name='label')
    pd.testing.assert_series_equal(sample_dataset.labels, expected_series)


def test_labels_property_unrecognized_format(sample_dataset):
    sample_dataset.df['label'] = ['unknown', 'unknown']
    with pytest.raises(LabelFormatError):
        _ = sample_dataset.labels


def test_text_to_list_property(sample_dataset):
    expected_list = ['text1', 'text2']
    assert sample_dataset.text_to_list == expected_list


def test_labels_to_list_property_numeric(sample_dataset):
    sample_dataset.df['label'] = [0, 1]
    expected_list = ['negative', 'positive']
    assert sample_dataset.labels_to_list == expected_list


def test_labels_to_list_property_string(sample_dataset):
    sample_dataset.df['label'] = ['negative', 'positive']
    expected_list = ['negative', 'positive']
    assert sample_dataset.labels_to_list == expected_list
