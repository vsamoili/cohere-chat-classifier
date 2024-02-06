from unittest.mock import patch, MagicMock

import pytest

from theydo import config
from theydo.cohere_model import CohereChat


@patch('cohere_chat_classifier.cohere_model.cohere.Client')
@patch('cohere_chat_classifier.cohere_model.os.getenv', return_value='dummy_api_key')
def test_init(mock_getenv, mock_client):
    # Test initialization with default base_message
    chat_default = CohereChat()
    mock_getenv.assert_called_with('COHERE_API_KEY')
    mock_client.assert_called_with('dummy_api_key')
    assert chat_default.model is not None
    assert chat_default._clf_model == config['classification_model']
    assert chat_default._gen_model == config['generation_model']
    assert chat_default._gen_temperature == 0.3
    assert chat_default.start_review_token == '<START>'
    assert chat_default.end_review_token == '<END>'
    assert chat_default.reviews_to_parse_at_once == 3
    assert chat_default.chat_base_message.startswith("You are a sentiment analysis classifier.")

    custom_message = "Custom base message"
    chat_custom = CohereChat(base_message=custom_message)
    assert chat_custom.chat_base_message == custom_message


def test_gen_model_getter():
    chat = CohereChat()
    assert chat.gen_model == chat._gen_model


def test_gen_model_setter_valid():
    chat = CohereChat()
    chat.gen_model = 'command'
    assert chat.gen_model == 'command'


def test_gen_model_setter_invalid():
    chat = CohereChat()
    with pytest.raises(ValueError):
        chat.gen_model = 'invalid_model'


def test_gen_temperature_getter():
    chat = CohereChat()
    assert chat.gen_temperature == 0.3


def test_gen_temperature_setter_valid():
    chat = CohereChat()
    chat.gen_temperature = 1.0
    assert chat.gen_temperature == 1.0


def test_gen_temperature_setter_invalid():
    chat = CohereChat()
    with pytest.raises(ValueError):
        chat.gen_temperature = -1


@patch('cohere_chat_classifier.cohere_model.cohere.Client.chat', return_value=MagicMock())
def test_chat(mock_chat):
    chat = CohereChat()
    response = chat.chat("message", "conv_id")
    mock_chat.assert_called_with(
        message="message", model=chat.gen_model, conversation_id="conv_id",
        temperature=chat.gen_temperature, prompt_truncation='AUTO'
    )
    assert response is not None


def test_parse_chat_response_valid():
    response = '{"text": "review", "label": "positive"}'
    result = CohereChat.parse_chat_response(response)
    assert result[0]["label"] == "positive"


@patch('cohere_chat_classifier.cohere_model.re.findall', return_value=['invalid_json'])
def test_parse_chat_response_invalid(mock_findall):
    response = 'invalid_json'
    result = CohereChat.parse_chat_response(response)
    assert result[0]["text"] == "None"


def test_create_review_prompt_continuation():
    chat = CohereChat()
    texts = ["review1", "review2"]

    # Test with continuation_prompt set to True
    continuation_prompt = chat.create_review_prompt(texts, continuation_prompt=True)
    expected_continuation_prompt = f"Do the same for the {len(texts)} following reviews. Remember that your whole answer must be exactly a parsable JSON and nothing else:"
    for text in texts:
        expected_continuation_prompt += f'\n{chat.start_review_token}\n' + text + f'\n{chat.end_review_token}\n'

    assert continuation_prompt == expected_continuation_prompt


def test_create_review_prompt_new():
    chat = CohereChat()
    texts = ["review1", "review2"]

    # Test with continuation_prompt set to False
    new_prompt = chat.create_review_prompt(texts, continuation_prompt=False)
    expected_new_prompt = chat.chat_base_message
    for text in texts:
        expected_new_prompt += f'\n{chat.start_review_token}\n' + text + f'\n{chat.end_review_token}\n'

    assert new_prompt == expected_new_prompt


def test_create_review_prompt_empty():
    chat = CohereChat()
    texts = []

    # Test with an empty list of texts
    prompt_empty = chat.create_review_prompt(texts, continuation_prompt=False)
    assert prompt_empty == chat.chat_base_message


def test_create_review_prompt_single_review():
    chat = CohereChat()
    texts = ["single_review"]

    # Test with a single review
    single_review_prompt = chat.create_review_prompt(texts, continuation_prompt=False)
    expected_single_prompt = chat.chat_base_message + f'\n{chat.start_review_token}\n' + texts[0] + f'\n{chat.end_review_token}\n'

    assert single_review_prompt == expected_single_prompt


@patch('cohere_chat_classifier.cohere_model.CohereChat.parse_chat_response', return_value=[])
@patch('cohere_chat_classifier.cohere_model.CohereChat.chat', return_value=MagicMock(text="chat_response"))
@patch('cohere_chat_classifier.cohere_model.logger')
def test_classify_with_prompt_retry_logic(mock_logger, mock_chat, mock_parse_chat_response):
    chat = CohereChat()
    chat.reviews_to_parse_at_once = 1
    inputs = ["review1"]
    max_retries_per_batch = 2

    chat.classify_with_prompt(inputs, max_retries_per_batch)

    # Assertions
    assert mock_chat.call_count == max_retries_per_batch
    mock_logger.warning.assert_called_once()


@patch('cohere_chat_classifier.cohere_model.CohereChat.parse_chat_response')
@patch('cohere_chat_classifier.cohere_model.CohereChat.chat')
@patch('cohere_chat_classifier.cohere_model.CohereChat.create_review_prompt')
@patch('cohere_chat_classifier.cohere_model.uuid.uuid4', return_value='test_uuid')
@patch('cohere_chat_classifier.cohere_model.logger')
def test_classify_with_prompt_batch_processing(mock_logger, mock_uuid, mock_create_review_prompt, mock_chat, mock_parse_chat_response):
    chat = CohereChat()
    chat.reviews_to_parse_at_once = 2
    inputs = ["review1", "review2", "review3"]
    max_retries_per_batch = 3

    # Mocking chat response and parse_chat_response
    mock_chat.return_value = MagicMock(text="chat_response")
    mock_parse_chat_response.side_effect = [
        [{"result": "output1"}, {"result": "output2"}],  # First batch response
        [],  # Second batch first try (empty results)
        [{"result": "output3"}]  # Second batch second try (successful results)
    ]

    # Execute the method
    results = chat.classify_with_prompt(inputs, max_retries_per_batch)

    # Assertions
    assert mock_create_review_prompt.call_count == 2
    assert mock_chat.call_count == 3  # 2 batches, with 1 retry in the second batch
    assert mock_parse_chat_response.call_count == 3
    assert len(results) == 3
    mock_logger.info.assert_called()
    mock_uuid.assert_called()
    mock_logger.warning.assert_not_called()  # No warnings as all retries were successful


@patch('cohere_chat_classifier.cohere_model.CohereChat.parse_chat_response', return_value=[])
@patch('cohere_chat_classifier.cohere_model.CohereChat.chat', return_value=MagicMock(text="chat_response"))
@patch('cohere_chat_classifier.cohere_model.logger')
def test_classify_with_prompt_no_results_warning(mock_logger, mock_chat, mock_parse_chat_response):
    chat = CohereChat()
    inputs = ["review1"]
    max_retries_per_batch = 3

    chat.classify_with_prompt(inputs, max_retries_per_batch)

    # Assertions
    assert mock_chat.call_count == max_retries_per_batch
    mock_logger.warning.assert_called_with("No classification results found in the latest generation. Please try again.")
