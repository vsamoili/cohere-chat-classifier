from pathlib import Path
from typing import List

import pandas as pd

from theydo import config
from theydo.exceptions import DatasetDoesNotExist


class Dataset:

    def __init__(self, full_dir: str | Path):
        self.df = None
        self.full_dir = full_dir

    def load(self):
        df = pd.read_parquet(self.full_dir, engine='pyarrow')
        self.df = df

    @property
    def text(self) -> pd.Series:
        return self.df.text

    @property
    def labels(self) -> pd.Series:
        # Labels in Cohere model must be strings
        return self.df.label.apply(lambda x: ['negative', 'positive'][x == 1])

    @property
    def numeric_labels(self) -> pd.Series:
        # Labels in Cohere model must be strings
        return self.df.label

    @property
    def text_to_list(self) -> List[str]:
        return self.text.to_list()

    @property
    def labels_to_list(self) -> List[int]:
        return self.labels.to_list()


def get_training_dir():
    try:
        return Path(__file__).resolve().parent.parent / Path(config.get('training_dir'))
    except FileNotFoundError:
        raise DatasetDoesNotExist("Training dataset does not exist. Please check config file and try again.")


def get_test_dir():
    try:
        return Path(__file__).resolve().parent.parent / Path(config.get('test_dir'))
    except FileNotFoundError:
        raise DatasetDoesNotExist("Training dataset does not exist. Please check config file and try again.")


if __name__ == "__main__":

    train_df = Dataset(get_training_dir())
    train_df.load()
    test_df = Dataset(get_test_dir())
    test_df.load()
    print('something')
