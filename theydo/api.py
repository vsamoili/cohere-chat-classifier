from typing import Optional

from cohere.responses.classify import Classifications
from cohere.responses.generation import Generations
from fastapi import FastAPI
from pydantic import BaseModel

from theydo.cohere_model import CohereModel, make_examples
from theydo.evaluation import calculate_all, make_dataset_from_request_data, make_dataset_from_clf_data

app = FastAPI()
model = CohereModel()


class Data(BaseModel):
    text: str | None = None
    label: str | None = None


class ClassificationRequest(BaseModel):
    inputs: list[str]
    training_data: list[Data]
    test_data: Optional[list[Data]] = None


class GenerationResponse(BaseModel):
    pass


class Prediction(BaseModel):
    input: str
    prediction: str
    confidence: float


class ClassificationResponse(BaseModel):
    predictions: list[Prediction]
    metrics: dict[str, float]


# def format_generation_response(gen_results: Generations) -> GenerationResponse:
#     pass


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


@app.get("/status")
async def read_root():
    return {"Status": "OK"}


# @app.post("/generate/{prompt}")
# async def generate_response(prompt: str):
#     return model.generate(prompt)
#

@app.post("/classify")
async def classification_response(request_data: ClassificationRequest) -> ClassificationResponse:
    inputs = request_data.inputs
    training_data = [(item.text, item.label) for item in request_data.training_data]
    test_data = [(item.text, item.label) for item in request_data.test_data]
    clf_results = model.classify(inputs=inputs, examples=make_examples(training_data))

    if request_data.test_data:
        metrics = calculate_all(
            pred_data=make_dataset_from_clf_data(clf_results),
            eval_data=make_dataset_from_request_data(test_data))
    else:
        metrics = None
    return format_classification_response(clf_results, metrics)
