"""Core sentiment analysis functionality wrapped into ``SentimentAnalyzer`` for more adequate
outputs."""

import os

import httpx
from httpx import Response
from loguru import logger
from pydantic import ValidationError

from .errors import (
    BadRequestError,
    InternalServerError,
    NotFoundError,
    ResponseValidationError,
    UnexpectedError,
)
from .schemas import ModelResponse, SentimentAnalysis, SentimentScore
from .sentiment import Sentiment


class SentimentAnalyzer:
    """Model wrapper for sentiment analysis. Normalizes multilabeled outputs into a single polarity
    value and provides a sentiment category.

    The model internally used is tweeter-roberta-base-sentiment-latest.

    You will need to provide a Hugging Face API key to use the model trough the environment variable
    ``HUGGINGFACE_API_KEY``.

    The model API URL is defined as a constant as wrapper for the model is intrinsically tied to the
    specific model. It does not make sense to change the model without changing the implementation
    though a configuration file or similar.

    :param timeout: Default timeout for the HTTP request to the model in seconds. Default is 10
        seconds.
    """

    _MODEL_API_URL: str = (
        "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    def __init__(self, timeout: float = 10) -> None:
        self._timeout = timeout

    async def analyze_sentiment(
        self, statements: str | list[str], timeout: float | None = None
    ) -> ModelResponse:
        """Analyze the sentiment of the given statements.

        :param statements: The statements to analyze the sentiment of.
        :param timeout: Timeout for the HTTP request in seconds. If not provided, the client default
            timeout will be used. Default is ``None``.
        :return: The sentiment scores of the statements in the order they are given.
        """
        n_statements = len(statements) if isinstance(statements, list) else 1

        logger.info("Generating sentiment analysis for {} statements.", n_statements)

        model_response = await self._request_sentiment(statements, timeout)

        outputs = _process_outputs(statements, model_response)
        logger.info("Sentiment analysis generated for {} statements.", n_statements)

        return outputs

    async def _request_sentiment(
        self, statements: str | list[str], timeout: float | None = None
    ) -> list[SentimentScore]:
        """Request inference the sentiment of the given statements.

        :param statements: The statements to infer the sentiment of.
        :param timeout: Timeout for the HTTP request in seconds. If not provided, the
            client default timeout will be used. Default is ``None``.
        :return: The sentiment scores of the statements in the order they are given.
        """
        async with httpx.AsyncClient(timeout=self._choose_timeout(timeout)) as client:
            response = await client.post(
                self._MODEL_API_URL,
                json={"inputs": statements},
                headers={"Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}"},
            )

        return _handle_response(response).outputs

    def _choose_timeout(self, provided: float | None) -> float:
        """Choose the timeout value to use for the HTTP request.

        :param provided: The timeout value provided by the user.
        :return: The timeout value to use.
        """
        return provided if provided is not None else self._timeout


def _handle_response(response: Response) -> ModelResponse:
    """Handle the HTTP response from the model API.

    :param response: The HTTP response object.
    :raises ResponseValidationError: If the response does not match the expected schema.
    :raises BadRequestError: If the status code is 400.
    :raises NotFoundError: If the status code is 404.
    :raises InternalServerError: If the status code is 500.
    :raises UnexpectedError: If the status code is not 200, 400, 404, or 500.
    :return: The JSON content of the response if the status code is 200.
    """
    logger.debug("Response status code: {}", response.status_code)

    match response.status_code:
        case 200:
            content = response.json()
            logger.debug("Response content: {}", content)

            try:
                return ModelResponse.model_validate({"outputs": content})

            except (ValidationError, ValueError) as exc:
                raise ResponseValidationError(
                    "Response does not match the expected schema or values."
                ) from exc

        case 400:
            raise BadRequestError(f"Bad Request: {response.text}.")
        case 404:
            raise NotFoundError(f"Not Found: {response.text}.")
        case 500:
            raise InternalServerError(f"Internal Server Error: {response.text}.")
        case _:
            raise UnexpectedError(f"Unexpected Error: {response.status_code} - {response.text}.")


def _process_outputs(
    statements: str | list[str], outputs: list[SentimentScore]
) -> list[SentimentAnalysis]:
    """Parse the model outputs into a list of sentiment analysis.

    :param statements: The statements that were analyzed.
    :param outputs: The model outputs to parse.
    :return: The sentiment analysis for each output provided by the models.
    """
    if isinstance(statements, str):
        statements = [statements]

    return [
        _analyze_output(statement, output)
        for statement, output in zip(statements, outputs, strict=False)
    ]


def _analyze_output(statement: str, output: SentimentScore) -> SentimentAnalysis:
    """Analyze the output of the model and return a sentiment analysis.

    :param statement: The statement that was analyzed.
    :param output: The output of the model.
    :return: The sentiment analysis of the output.
    """
    scores = {score.label: score.score for score in output}
    polarity = _compute_polarity(scores["positive"], scores["neutral"], scores["negative"])
    return SentimentAnalysis(
        statement=statement, polarity=polarity, sentiment=Sentiment.from_polarity(polarity)
    )


def _compute_polarity(positive: float, neutral: float, negative: float) -> float:
    """Compute the polarity value from the scores of the three labels.

    Polarity will be provided in a range of -1 to 1, where -1 is the most negative and 1 is the most
    positive.

    Given that the model returns not only positive and negative scores, but also a neutral score,
    the neutral score will serve as a normalization factor.

    Polarity is computed with the following formula:
    (positive - negative) / (positive + neutral + negative)

    :param positive: The score of the positive label.
    :param neutral: The score of the neutral label.
    :param negative: The score of the negative label.
    :return: The computed polarity value.
    """
    return (positive - negative) / (positive + neutral + negative)
