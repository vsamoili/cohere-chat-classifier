from unittest.mock import patch, MagicMock
import pytest
from theydo.cohere_model import CohereChat
import uuid


@patch('theydo.cohere_model.cohere.Client')
@patch('theydo.cohere_model.os.getenv', return_value='dummy_api_key')
def test_init(mock_getenv, mock_client):
    chat = CohereChat()
    mock_getenv.assert_called_with('COHERE_API_KEY')
    mock_client.assert_called_with('dummy_api_key')
    assert chat._gen_temperature == 0.3
    # Add more assertions here to test the rest of the initialization logic


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


@patch('theydo.cohere_model.cohere.Client.chat', return_value=MagicMock())
def test_chat(mock_chat):
    chat = CohereChat()
    response = chat.chat("message", "conv_id")
    mock_chat.assert_called_with(
        message="message", model=chat.gen_model, conversation_id="conv_id",
        temperature=chat.gen_temperature, prompt_truncation='AUTO'
    )
    assert response is not None


@patch('theydo.cohere_model.CohereChat.parse_chat_response')
@patch('theydo.cohere_model.CohereChat.chat')
@patch('theydo.cohere_model.CohereChat.create_review_prompt')
def test_classify_with_prompt(mock_create_review_prompt, mock_chat, mock_parse_chat_response):
    # Mock responses for the chat method
    mock_chat.return_value = MagicMock(text="chat_response")

    # Mock responses for parsing chat response
    mock_parse_chat_response.side_effect = [
        [{"text": "review1", "label": "positive"}],
        [{"text": "review2", "label": "negative"}],
        [{"text": None, "label": "positive"}]  # Simulate no results for a batch
    ]

    chat = CohereChat()
    chat.reviews_to_parse_at_once = 2

    inputs = ["review1", "review2", "review3"]

    # Call the method under test
    results = chat.classify_with_prompt(inputs)

    # Assertions for create_review_prompt
    assert mock_create_review_prompt.call_count == 2

    # Assertions for chat method
    assert mock_chat.call_count == 2

    # Assertions for parse_chat_response
    assert mock_parse_chat_response.call_count == 2

    # Assertions for the result
    expected_results = [
        {"text": "review1", "label": "positive"},
        {"text": "review2", "label": "negative"}
    ]
    assert results == expected_results

    # Test with empty input list
    results_empty = chat.classify_with_prompt([])
    assert results_empty == []


def test_parse_chat_response_valid():
    response = '{"text": "review", "label": "positive"}'
    result = CohereChat.parse_chat_response(response)
    assert result[0]["label"] == "positive"


@patch('theydo.cohere_model.re.findall', return_value=['invalid_json'])
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
