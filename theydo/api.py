from cohere.responses.classify import Classifications
from cohere.responses.generation import Generations
from fastapi import FastAPI
from pydantic import BaseModel

from theydo.cohere_model import CohereModel, make_examples

app = FastAPI()
model = CohereModel()


class Example(BaseModel):
    text: str | None = None
    label: str | None = None


class GenerationResponse(BaseModel):
    pass


class ClassificationResponse(BaseModel):
    input: str
    prediction: str
    confidence: float


class ClassificationData(BaseModel):
    inputs: list[str]
    examples: list[Example]


def format_generation_response(gen_results: Generations) -> GenerationResponse:
    pass


def format_classification_response(clf_results: Classifications) -> list[ClassificationResponse]:
    return [
        ClassificationResponse(
            input=item.input,
            prediction=item.prediction,
            confidence=item.confidence
        ) for item in clf_results
    ]


@app.get("/status")
async def read_root():
    return {"Status": "OK"}


@app.post("/generate/{prompt}")
async def generate_response(prompt: str):
    return model.generate(prompt)


@app.post("/classify")
async def classification_response(data: ClassificationData) -> list[ClassificationResponse]:
    inputs = data.inputs
    examples = [(example.text, example.label) for example in data.examples]
    results = model.classify(inputs=inputs, examples=make_examples(examples))
    return format_classification_response(results)
