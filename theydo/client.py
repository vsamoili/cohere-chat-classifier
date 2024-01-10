from typing import Optional

import requests

from theydo.data_model import get_test_dir, Dataset, get_training_dir

BASE_URL = "http://127.0.0.1:8000"


def get_status():
    response = requests.get(f"{BASE_URL}/status")
    return response.json()


def classify_prompt_request(inputs: list[str], training_data: list[dict[str, str]]):
    data = {"training_data": training_data, "inputs": inputs}
    response = requests.post(url=f"{BASE_URL}/classify", json=data)
    return response.json()


def classify_prompt_request_with_evaluation(inputs: list[str], test_data: Optional[list[dict[str, str]]] = None):
    data = {"training_data": None, "inputs": inputs, "test_data": test_data}
    response = requests.post(url=f"{BASE_URL}/classify_with_prompt", json=data)
    return response.json()


# Example usage for Chat component
if __name__ == "__main__":

    # Load test set from configured directory
    test_set = Dataset(full_dir=get_test_dir())
    test_set.load()

    # Shuffle data
    test_set.df = test_set.df.sample(frac=1)

    # Take small subsets
    test_set.df = test_set.df.iloc[:3, :]

    # Get server status
    status = get_status()
    print("Server Status:", status)

    # Classify a request with evaluation metrics using the Chat endpoint
    test_data = [{'text': text, 'label': label} for text, label in zip(test_set.text_to_list, test_set.labels_to_list)]
    input_data = test_set.text_to_list

    # results = classify_prompt_request(input_data, training_data)
    results_with_eval = classify_prompt_request_with_evaluation(inputs=input_data, test_data=test_data)
    print("Classification Response:", results_with_eval)
