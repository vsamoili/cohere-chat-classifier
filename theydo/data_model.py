from pathlib import Path
from typing import List, Optional

import pandas as pd

from theydo import config
from theydo.exceptions import DatasetDoesNotExist, LabelFormatError


class Dataset:

    def __init__(self, df: Optional[pd.DataFrame] = None, full_dir: Optional[str | Path] = None):
        self.df = df
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
        if set(self.df.label) == {0, 1}:
            return self.df.label.apply(lambda x: ['negative', 'positive'][x == 1])
        elif set(self.df.label) == {'positive', 'negative'}:
            return self.df.label
        else:
            raise LabelFormatError("Unrecognized label formatting. Should be either [0, 1] or ['positive', 'negative']")

    @property
    def numeric_labels(self) -> pd.Series:
        # Labels in Cohere model must be strings
        return self.labels.apply(lambda x: [0, 1][x == 'positive'])

    @property
    def text_to_list(self) -> List[str]:
        return self.text.to_list()

    @property
    def labels_to_list(self) -> List[str]:
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

    train_df = Dataset(full_dir=get_training_dir())
    train_df.load()
    test_df = Dataset(full_dir=get_test_dir())
    test_df.load()
    print('something')
