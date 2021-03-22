# MLEngFlaskProject_candidate
The purpose of this project is to test the candidate's ability to build a simple Flask application.


## Problem Statement
The candidate should be able to provide a working Flask application that exposes the provided model. We do not expect a full stack deployment, e.g. uWSGI-Nginx-Flask, but simple Flask application will suffice. 

The Flask application should provide at least two routes :
 - A simple `/health` route to make sure the application is indeed alive.
 - A `/predict` route that the end user can access to query the model.

The `/predict` route should accept requests and return response in the following JSON format :
 
Sample request:
```
{
    "request_id": 1234,
    "text": "blah blah",
    "language": "en-US",
    "encoding_type": "UTF-8"
}
```

Sample response:
```
{
    "request_id": 1234,
    "sentiment": [
        {
            "sentiment_type": "toxic",
            "sentiment_score": 0.5
        },
        {
            "sentiment_type": "severe_toxic",
            "sentiment_score": 0.1
        },
        {
            "sentiment_type": "obscene",
            "sentiment_score": 0.8
        },
        {
            "sentiment_type": "threat",
            "sentiment_score": 0.9
        },
        {
            "sentiment_type": "insult",
            "sentiment_score": 0.5
        },
        {
            "sentiment_type": "identity_hate",
            "sentiment_score": 0.01
        }
    ],
    "language": "en-US"
}
```

In addition, the Flask application should :
 - Be configurable, for example by providing the ability to specify development, testing and production configurations.
 - Provide logging capabilities.
 - Contain tests, for example some form of unit tests, integration tests and/or others.

## Setup
This template provides the skeleton for the application which should be completed by the candidate. It contains:
 - `app`: The directory that should contain most if not all the Flask application code.
 - `model`: Contains a trained scikit-learn sentiment model which should be use in this application. This is a very basic model without much thought behind it so it should mostly return random results, its purpose is mostly to provide a lightweight model for the service.
 - `tests`: The directory for the tests. It may contains various types of tests, e.g. unit, integration, component, contract, etc.
 - `Dockerfile`: The definition of the Docker image
 - `requirements.txt`: Contains the base environment for the Flask application. 

The candidate should create a virtual environment and install the libraries provided in the requirement file. Other libraries may be installed if required.
```
    pip install -r requirements.txt
```
Note: I would recommend not changing the version of scikit-learn as it may cause issues while loading the provided test model.

We recommend using the `python:3.8` Docker image as a base image for the Flask service as we made sure it is possible to build the application using the provided requirements.

The image can be pull using the following command : 
```
    docker pull python:3.8
```

## Additional Notes
We expect the candidate to document his/her code and respect his/her preferred Python coding standard (e.g. PEP, Google, etc.).
