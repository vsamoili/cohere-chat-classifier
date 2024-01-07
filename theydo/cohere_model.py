import os
from typing import List, Tuple

import cohere
from cohere.responses.classify import Example, Classifications
from cohere.responses.generation import Generations

from theydo import config
from theydo.data_model import get_training_dir, get_test_dir, Dataset


class CohereModel:

    def __init__(self):
        self.model = cohere.Client(os.getenv('COHERE_API_KEY'))
        self._clf_model = config['classification_model']
        self._gen_model = config['generation_model']

    @property
    def clf_model(self):
        return self._clf_model

    @clf_model.setter
    def clf_model(self, value):
        if value.lower() in ('small', 'medium', 'large'):
            self._clf_model = value
        else:
            raise ValueError("Invalid model size specification. Must be one of 'small', 'medium', 'large'.")

    @property
    def gen_model(self):
        return self._gen_model

    @gen_model.setter
    def gen_model(self, value):
        if value.lower() in ('command', 'command-nightly', 'command-light', 'command-light-nightly'):
            self._gen_model = value
        else:
            raise ValueError("Invalid model size specification. Must be one of 'small', 'medium', 'large'.")

    def generate(self, prompt: str) -> Generations:
        return self.model.generate(prompt=prompt, model=self.gen_model)

    def classify(self, inputs: List[str], examples: List[Example]) -> Classifications:
        return self.model.classify(
            model=self.clf_model,
            inputs=inputs,
            examples=examples
        )

    def evaluate(self):
        pass


def make_examples(inputs: List[Tuple[str, str]]) -> List[Example]:
    return [Example(*item) for item in inputs]


if __name__ == "__main__":
    model = CohereModel()
    training_set = Dataset(full_dir=get_training_dir())
    test_set = Dataset(full_dir=get_test_dir())
    training_set.load()
    test_set.load()
    training_set.df = training_set.df.sample(frac=1)
    test_set.df = test_set.df.sample(frac=1)

    training_set.df = training_set.df.iloc[:100, :]
    test_set.df = test_set.df.iloc[:10, :]

    results = model.classify(
        inputs=test_set.text_to_list,
        examples=make_examples(test_set)
    )
    print('kati')

    # response = model.generate(training_set.text_to_list[0])
    # print('response')
