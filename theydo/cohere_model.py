import os
from typing import List, Tuple, Optional
import re
import json

import cohere
from cohere.responses.classify import Example, Classifications
from cohere.responses.chat import Chat

from theydo import config
from theydo.data_model import get_training_dir, get_test_dir, Dataset
from theydo.evaluation import calculate_all
from theydo.helpers import make_dataset_from_chat_response


class CohereClassifier:

    def __init__(self):
        self.model = cohere.Client(os.getenv('COHERE_API_KEY'))
        self._clf_model = config['classification_model']

    @property
    def clf_model(self):
        return self._clf_model

    @clf_model.setter
    def clf_model(self, value):
        if value.lower() in ('small', 'medium', 'large'):
            self._clf_model = value
        else:
            raise ValueError("Invalid model size specification. Must be one of 'small', 'medium', 'large'.")

    def classify(self, inputs: List[str], examples: List[Example]) -> Classifications:
        return self.model.classify(
            model=self.clf_model,
            inputs=inputs,
            examples=examples
        )


class CohereChat:

    def __init__(self):
        self.model = cohere.Client(os.getenv('COHERE_API_KEY'))
        self._clf_model = config['classification_model']
        self._gen_model = config['generation_model']
        self._gen_temperature = None

        self.chat_delimiter = '###'
        self.chat_base_message = f"""You are a sentiment analysis classifier. You will be provided a list of movie 
reviews delimited with '{self.chat_delimiter}'. Classify each movie review as 'positive' or 'negative' based on the 
sentiment. Provide your output in a json format with two keys: 'text' and 'label' where 'text' is the original review 
to be classified and 'label' is one of 'positive' or 'negative'. Here is the list:
"""

        self.chat_tuned_message = ''

    @property
    def gen_model(self):
        return self._gen_model

    @gen_model.setter
    def gen_model(self, value):
        if value.lower() in ('command', 'command-nightly', 'command-light', 'command-light-nightly'):
            self._gen_model = value
        else:
            raise ValueError("Invalid model size specification. Must be one of 'small', 'medium', 'large'.")

    def chat(self, msg: str) -> Chat:
        return self.model.chat(message=msg, model=self.gen_model)

    @property
    def gen_temperature(self):
        return self._gen_temperature

    @gen_temperature.setter
    def gen_temperature(self, value):
        if value < 0.0 or value > 2.0:
            raise ValueError("Temperature must be a value between 0.0 and 2.0.")
        self._gen_temperature = value

    def initialize_chat(self):
        self.model.chat(

        )

    def classify_with_prompt(
            self,
            inputs: List[str],
    ) -> list[dict[str, str]]:

        complete_message = self.create_review_prompt(inputs)
        response = model.chat(complete_message)
        results = self.parse_chat_response(response.text)

        return results

    def create_review_prompt(self, texts: List[str]) -> str:
        return self.chat_base_message + '\n\n' + f'\n{self.chat_delimiter}\n'.join(texts)

    @staticmethod
    def parse_chat_response(response: str) -> Optional[List[dict[str, str]]]:
        # Use regex to find the JSON-like structure in the string
        json_str = re.search(r'\[.*\{.*\}.*\]', response, re.DOTALL)

        if json_str:
            try:
                json_obj = json.loads(json_str.group())
            except json.JSONDecodeError:
                return None
            return json_obj
        else:
            return None


def make_examples(inputs: List[Tuple[str, str]]) -> List[Example]:
    return [Example(*item) for item in inputs]


if __name__ == "__main__":
    model = CohereChat()
    training_set = Dataset(full_dir=get_training_dir())
    test_set = Dataset(full_dir=get_test_dir())
    training_set.load()
    test_set.load()
    training_set.df = training_set.df.sample(frac=1)
    test_set.df = test_set.df.sample(frac=1)

    training_set.df = training_set.df.iloc[:100, :]
    test_set.df = test_set.df.iloc[:3, :]
    print('debug')

    # complete_message = model.create_complete_prompt(test_set.text_to_list)
    # response = model.chat(complete_message)
    # results = model.parse_chat_response(response.text)
    #
    # pred_data = make_dataset_from_chat_response(results)
    # eval_data = test_set
    # metrics = calculate_all(pred_data=pred_data, eval_data=eval_data)
    # print('response')
