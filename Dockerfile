FROM python:3.10-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install poetry
RUN poetry config virtualenvs.create false

ADD pyproject.toml .
ADD poetry.lock .
RUN poetry install --no-dev

EXPOSE 8000

# Run the application
ENTRYPOINT ["uvicorn", "cohere_chat_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
