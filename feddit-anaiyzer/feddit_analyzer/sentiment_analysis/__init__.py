"""Sentiment analysis functionality.

The model used is tweeter-roberta-base-sentiment-latest.

The decision to use this model was based on the following reasons:

    - The model is pre-trained on a large corpus of tweets, which may have a similar content and
        format to Reddit comments which Feedit aims to emulate.
    - There is a serverless API available for prototyping that allows to call the model

In a real world scenario, the model may be fine-tuned on an internal dataset of labeled Feddit
comments and deployed properly using a cloud service like AWS SageMaker Endpoints. Adding all the
model observability and monitoring tools to ensure the model is working as expected.
"""

from ._analyzer import SentimentAnalyzer

__all__ = ["SentimentAnalyzer"]
