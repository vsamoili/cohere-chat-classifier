from typing import Optional

from cohere.responses.classify import Classifications
from cohere.responses.generation import Generations
from fastapi import FastAPI
from pydantic import BaseModel

from theydo.cohere_model import CohereChat, make_examples
from theydo.evaluation import calculate_all
from theydo.helpers import make_dataset_from_request_data, make_dataset_from_clf_data, make_dataset_from_chat_response

app = FastAPI()
model = CohereChat()


class Data(BaseModel):
    text: str | None = None
    label: str | None = None


class ClassificationRequest(BaseModel):
    inputs: list[str]
    training_data: list[Data]
    test_data: Optional[list[Data]] = None


class Prediction(BaseModel):
    input: str
    prediction: str
    confidence: Optional[float] = None


class ClassificationResponse(BaseModel):
    predictions: list[Prediction]
    metrics: dict[str, float]


def format_classification_response(
        clf_results: Classifications,
        metrics: Optional[dict[str, float]] = None
) -> ClassificationResponse:
    return ClassificationResponse(
        predictions=[
            Prediction(
                input=item.input,
                prediction=item.prediction,
                confidence=item.confidence
            ) for item in clf_results
        ],
        metrics=metrics
    )


def format_classification_with_prompt_response(
        clf_results: list[dict[str, str]],
        metrics: Optional[dict[str, float]] = None
) -> ClassificationResponse:
    return ClassificationResponse(
        predictions=[
            Prediction(
                input=item['text'],
                prediction=item['label'],
                confidence=None
            ) for item in clf_results
        ],
        metrics=metrics
    )


@app.get("/status")
async def read_root():
    return {"Status": "OK"}


@app.post("/classify")
async def classify(request_data: ClassificationRequest) -> ClassificationResponse:
    inputs = request_data.inputs
    training_data = [(item.text, item.label) for item in request_data.training_data]
    clf_results = model.classify(inputs=inputs, examples=make_examples(training_data))

    if request_data.test_data:
        test_data = [(item.text, item.label) for item in request_data.test_data]
        metrics = calculate_all(
            pred_data=make_dataset_from_clf_data(clf_results),
            eval_data=make_dataset_from_request_data(test_data))
    else:
        metrics = None
    return format_classification_response(clf_results, metrics)


@app.post("/classify_with_prompt")
async def classify_with_prompt(request_data: ClassificationRequest) -> ClassificationResponse:
    inputs = request_data.inputs
    if request_data.training_data:
        few_shot_examples = [(item.text, item.label) for item in request_data.training_data][:5]
    else:
        few_shot_examples = None

    clf_results = model.classify_with_prompt(inputs=inputs, few_shot_examples=few_shot_examples)

    if request_data.test_data:
        test_data = [(item.text, item.label) for item in request_data.test_data]
        metrics = calculate_all(
            pred_data=make_dataset_from_chat_response(clf_results),
            eval_data=make_dataset_from_request_data(test_data))
    else:
        metrics = None
    return format_classification_with_prompt_response(clf_results, metrics)
