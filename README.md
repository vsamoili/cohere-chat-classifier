# theydo

A Python service that uses the open source Cohere LLM to perform sentiment analysis classification on the IMDB movie review dataset.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Docker](#docker)

## Overview

`theydo` is a Python service designed to leverage the open-source Cohere LLM (Language Model) for sentiment analysis classification on the IMDB movie review dataset. This service provides a user-friendly interface to analyze and classify the sentiment of movie reviews as "positive" or "negative". It has two main POST endpoints, one that uses the Cohere Classifier and another one for the requirement of this assignment, which is to use a prompt-response type of model to perform classifications. For the latter, the Cohere Chat component is used and manipulated given a suitable prompt and structure to enable its classification task. Both models can perform basic metrics calculations (accuracy, precision, recall, f1 score averages) if given a test set with ground truth labels. 

## Prerequisites

System prerequisites for running the app:

- Python 3.10+
- Poetry (for dependency management)
- Docker (for containerization)

## Getting Started

Follow these instructions to install:

1. Clone this repository using SSH:

   ```bash
   git clone git@github.com:<username>/theydo.git
   ```

2. Navigate to the project directory:

   ```bash
   cd theydo
   ```

3. Install project dependencies using Poetry:

   ```bash
   poetry install
   ```

4. Start the FastAPI application:

   ```bash
   uvicorn theydo.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

### API Endpoints

- `GET /status`: Service status.
- `POST /classify`: Use the Cohere Classification model to perform classification on input data by providing a few examples of training data. If test data are provided, it will also produce standard classification metrics on the given test dataset.
Classification includes confidence level of the LLM.
- `POST /classify_with_prompt`: Use the Cohere Chat model to perform zero shot classification on input data. If test data are provided, it will also produce standard classification metrics on the given test dataset.

Use the `client.py` to perform the above requests to the service. 

### Example Request

```Python
classify_request_with_evaluation(inputs=<input_data>, test_data=<test_data>)
```
See `client.py` for input types.

### Example Response

```json
{
   "predictions":[
      {
         "input":"Ugh, bad, bad, bad, [...] getting to know you time but whatever.",
         "prediction":"negative",
         "confidence":"None"
      },
      {
         "input":"Wow.this is a touching story! First i saw 'Rescue Dawn'. [...] The horror doesn't get more real than in the words of Dieter Dengler himself.He totally succeeds in painting the picture.",
         "prediction":"positive",
         "confidence":"None"
      },
      {
         "input":"We found this movie nearly impossible to watch. [...] This was supposed to be a television movie, guys, not Books on Tape.",
         "prediction":"negative",
         "confidence":"None"
      }
   ],
   "metrics":{
      "accuracy_avg":1.0,
      "precision_avg":1.0,
      "recall_avg":1.0,
      "f1_avg":1.0
   }
}
```

## Project Structure

Explain the structure of your project's directory and important files. For example:

- `theydo/`: Application logic.
- `tests/`: Unit tests with the pytest framework.
- `poetry.lock` and `pyproject.toml`: Poetry files for dependency management.
- `.env`: a necessary file that stores the Cohere API key. See `.env_sample` for structure.
- `config.yaml`: standard configuration for datasets and model types used.
  
## Docker

### Build Docker Image

```bash
docker build -t theydo .
```

### Run Docker Container

The following command will run the container and startup the service using the `gunicorn` server:

```bash
docker run -env_file ./.env -p 8000:8000 theydo:latest
```
