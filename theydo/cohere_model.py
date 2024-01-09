import os
from typing import List, Tuple, Optional
import re
import json
import uuid

import cohere
from cohere.responses.classify import Example, Classifications
from cohere.responses.chat import Chat
import logging

from theydo import config
from theydo.data_model import get_training_dir, get_test_dir, Dataset
from theydo.evaluation import calculate_all
from theydo.helpers import make_dataset_from_chat_response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    def __init__(self, base_message: Optional[str] = None):
        self.model = cohere.Client(os.getenv('COHERE_API_KEY'))
        self._clf_model = config['classification_model']
        self._gen_model = config['generation_model']
        self._gen_temperature = 0.3     # default temperature

        self.start_review_token = '<START>'
        self.end_review_token = '<END>'

        self.reviews_to_parse_at_once = 3

        if not base_message:
            self.chat_base_message = f"""You are a sentiment analysis classifier. You will be provided a list of movie 
reviews each starting with a start token "{self.start_review_token}" and ending with an end token "{self.end_review_token}". Classify each movie review as 
'positive' or 'negative' based on its underlying sentiment. Provide your output in a JSON format with two keys: 'text' and 'label' 
where 'text' is the original review  to be classified and 'label' is one of 'positive' or 'negative'. Here is the list:
"""
        else:
            self.chat_base_message = base_message

    @property
    def gen_model(self):
        return self._gen_model

    @gen_model.setter
    def gen_model(self, value):
        if value.lower() in ('command', 'command-nightly', 'command-light', 'command-light-nightly'):
            self._gen_model = value
        else:
            raise ValueError("Invalid model size specification. Must be one of 'small', 'medium', 'large'.")

    def chat(self, msg: str, conversation_id: str) -> Chat:
        return self.model.chat(
            message=msg,
            model=self.gen_model,
            conversation_id=conversation_id,
            temperature=self.gen_temperature,
            prompt_truncation='AUTO'
        )

    @property
    def gen_temperature(self):
        return self._gen_temperature

    @gen_temperature.setter
    def gen_temperature(self, value):
        if value < 0.0 or value > 2.0:
            raise ValueError("Temperature must be a value between 0.0 and 2.0.")
        self._gen_temperature = value

    def classify_with_prompt(self, inputs: List[str]) -> list[dict[str, str]]:
        """
        Classify a list of inputs using a prompt-based approach.

        This function classifies a list of inputs by generating and sending prompts to a chat interface
        in batches. Each batch of inputs is processed by creating a review prompt (using 'create_review_prompt'),
        sending this prompt to the chat interface, and then parsing the chat response. The function handles
        the inputs in chunks of size 'self.reviews_to_parse_at_once' and maintains a conversation flow with
        the chat interface using a unique 'conversation_id'.

        If any batch does not yield classification results, a warning is logged. The function accumulates all
        successful results across batches and returns them.

        :param inputs: A list of text inputs to be classified.
        :return: A list of dictionaries containing classification results. Each dictionary corresponds to a
                 classified input.
        """

        conversation_id = str(uuid.uuid4())
        all_results = []

        i = 0
        while i < len(inputs):

            logger.info(f"Classifying {i+self.reviews_to_parse_at_once} out of {len(inputs)} reviews...")
            # Iterate through inputs n at a time
            current_inputs = inputs[i:i+self.reviews_to_parse_at_once]
            continuation_prompt = False if i == 0 else True

            complete_message = self.create_review_prompt(current_inputs, continuation_prompt=continuation_prompt)
            response = self.chat(complete_message, conversation_id=conversation_id)
            results = self.parse_chat_response(response.text)
            i += self.reviews_to_parse_at_once

            if not results:
                logger.warning("No classification results found in the latest generation. Please try again.")
            else:
                # Collect results
                all_results.extend(results)

        return all_results

    def create_review_prompt(self, texts: List[str], continuation_prompt: bool = False) -> str:
        """
        Create a review prompt string from a list of text reviews.

        This function generates a prompt string for review purposes from a given list of text reviews.
        The prompt can either be a continuation of a previous prompt or a new one, based on the
        'continuation_prompt' flag. If 'continuation_prompt' is True, the function creates a prompt
        instructing to do the same for the given number of reviews, ensuring the response is a parsable JSON.
        If 'continuation_prompt' is False, it starts with a base message defined in 'self.chat_base_message'.
        Each review text is encapsulated between 'start_review_token' and 'end_review_token' defined in the class.

        :param texts: A list of text reviews to be included in the prompt.
        :param continuation_prompt: A flag to indicate whether the prompt is a continuation of a previous one.
                                    Defaults to False.

        :return: A string containing the constructed review prompt.
        """
        if continuation_prompt:
            prompt = f'Do the same for the {len(texts)} following reviews. Remember that your whole answer must be exactly a parsable JSON and nothing else:'
        else:
            prompt = self.chat_base_message
        for text in texts:
            prompt += f'\n{self.start_review_token}\n' + text + f'\n{self.end_review_token}\n'
        return prompt

    @staticmethod
    def parse_chat_response(response: str) -> list[Optional[dict[str, str]]]:
        """
        Parse a chat response string to extract JSON-like structures.

        This function analyzes a string response, typically from a chat interface, to find and extract
        JSON-like structures. It uses regular expressions to identify patterns that resemble JSON objects
        and attempts to parse them. If the parsing is successful, the JSON object is added to the result list.
        If a JSON decoding error occurs, a warning is logged, and None is appended to the list instead.

        Note: This function assumes that the response string contains zero or more JSON-like structures
        encapsulated within curly braces '{}'.

        :param response: The chat response string containing potential JSON-like structures.
        :return: A list of dictionaries parsed from the JSON-like structures in the response string.
                 If a structure is unparsable, None is included in its place in the list.
        """
        # Use regex to find the JSON-like structure in the string
        matches = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
        found_objs = []

        for match in matches:
            try:
                json_obj = json.loads(match)
                found_objs.append(json_obj)
            except json.JSONDecodeError:
                logger.warning(f"Unparsable JSON found for pattern: {match}")
                found_objs.append({"text": "None", "label": "positive"})

        return found_objs


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
    test_set.df = test_set.df.iloc[:10, :]

    results = model.classify_with_prompt(test_set.text_to_list)
    print('debug')

    pred_data = make_dataset_from_chat_response(results)
    eval_data = test_set
    metrics = calculate_all(pred_data=pred_data, eval_data=eval_data)
    print('response')
