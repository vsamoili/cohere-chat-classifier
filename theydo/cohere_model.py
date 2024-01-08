import os
from typing import List, Tuple

import cohere
from cohere.responses.classify import Example, Classifications
from cohere.responses.generation import Generations

from theydo import config
from theydo.data_model import get_training_dir, get_test_dir, Dataset
from theydo.evaluation import calculate_metric, get_classification_report


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

    training_set.df = training_set.df.iloc[:200, :]
    test_set.df = test_set.df.iloc[:90, :]

    examples = make_examples([(text, label) for text, label in zip(training_set.text_to_list, training_set.labels_to_list)])

    results = model.classify(
        inputs=test_set.text_to_list,
        examples=examples
    )

    metrics = {
        'acc': calculate_metric(pred_data=results, eval_data=test_set, metric='acc'),
        'prec': calculate_metric(pred_data=results, eval_data=test_set, metric='prec'),
        'rec': calculate_metric(pred_data=results, eval_data=test_set, metric='rec'),
        'f1': calculate_metric(pred_data=results, eval_data=test_set, metric='f1'),
        # 'auc': calculate_metric(pred_data=results, eval_data=test_set, metric='auc')
    }

    print(get_classification_report(results, test_set))

    for i, (pred, true) in enumerate(zip(results, test_set.labels_to_list)):
        if pred.prediction != true:
            print(i, pred.confidence)
    # response = model.generate(training_set.text_to_list[0])
    # print('response')
