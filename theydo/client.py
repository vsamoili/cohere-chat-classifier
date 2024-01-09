import requests

from theydo.data_model import get_test_dir, Dataset, get_training_dir

BASE_URL = "http://127.0.0.1:80"


def get_status():
    response = requests.get(f"{BASE_URL}/status")
    return response.json()


def generate_response(prompt: str):
    response = requests.post(f"{BASE_URL}/generate/{prompt}")
    return response.json()


def classify_request(inputs: list[str], training_data: list[dict[str, str]]):
    data = {"training_data": training_data, "inputs": inputs}
    response = requests.post(url=f"{BASE_URL}/classify", json=data)
    return response.json()


def classify_request_with_evaluation(inputs: list[str], training_data: list[dict[str, str]], test_data: list[dict[str, str]]):
    data = {"training_data": training_data, "inputs": inputs, "test_data": test_data}
    response = requests.post(url=f"{BASE_URL}/classify", json=data)
    return response.json()


# Example usage
if __name__ == "__main__":

    training_set = Dataset(full_dir=get_training_dir())
    test_set = Dataset(full_dir=get_test_dir())
    training_set.load()
    test_set.load()
    training_set.df = training_set.df.sample(frac=1)
    test_set.df = test_set.df.sample(frac=1)

    training_set.df = training_set.df.iloc[:100, :]
    test_set.df = test_set.df.iloc[:90, :]

    # Get server status
    status = get_status()
    print("Server Status:", status)

    # # Generate a response
    # prompt = "Hello, World!"
    # generated_response = generate_response(prompt)
    # print("Generated Response:", generated_response)

    # Classify a request
    training_data = [{'text': text, 'label': label} for text, label in zip(training_set.text_to_list, training_set.labels_to_list)]
    test_data = [{'text': text, 'label': label} for text, label in zip(test_set.text_to_list, test_set.labels_to_list)]
    input_data = test_set.text_to_list
    # results = classify_request(input_data, training_data)
    results_with_eval = classify_request_with_evaluation(input_data, training_data, test_data)
    print("Classification Response:", results_with_eval)
