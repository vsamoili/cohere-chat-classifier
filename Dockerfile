FROM python:3.10-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install poetry
RUN poetry config virtualenvs.create false

ADD pyproject.toml .
ADD poetry.lock .
RUN poetry install --no-dev

# Make port available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run the application
ENTRYPOINT ["poetry", "run", "uvicorn", "theydo.api:app", "--host", "0.0.0.0", "--port", "80"]
