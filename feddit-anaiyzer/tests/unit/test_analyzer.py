"""Unit tests for the ``SentimentAnalyzer``."""

import pytest
from pytest_httpx import HTTPXMock

from feddit_analyzer.sentiment_analysis import SentimentAnalyzer
from feddit_analyzer.sentiment_analysis.errors import (
    BadRequestError,
    InternalServerError,
    NotFoundError,
    ResponseValidationError,
    UnexpectedError,
)


@pytest.mark.asyncio()
async def test_analyze_sentiment_success(
    sentiment_analyzer: SentimentAnalyzer, httpx_mock: HTTPXMock, mock_model_response: callable
) -> None:
    """Test analyzing sentiment successfully."""
    statements = ["I love this!", "This is bad."]
    response_data = mock_model_response(statements)
    httpx_mock.add_response(url=SentimentAnalyzer._MODEL_API_URL, json=response_data)

    result = await sentiment_analyzer.analyze_sentiment(statements)

    assert len(result) == len(statements)
    assert all(isinstance(res.polarity, float) for res in result)
    assert all(res.sentiment in ["positive", "negative"] for res in result)


@pytest.mark.asyncio()
async def test_analyze_sentiment_bad_request(
    sentiment_analyzer: SentimentAnalyzer, httpx_mock: HTTPXMock
) -> None:
    """Test handling a 400 Bad Request error."""
    statements = ["Invalid input"]
    httpx_mock.add_response(
        url=SentimentAnalyzer._MODEL_API_URL, status_code=400, text="Bad request"
    )

    with pytest.raises(BadRequestError):
        await sentiment_analyzer.analyze_sentiment(statements)


@pytest.mark.asyncio()
async def test_analyze_sentiment_not_found(
    sentiment_analyzer: SentimentAnalyzer, httpx_mock: HTTPXMock
) -> None:
    """Test handling a 404 Not Found error."""
    statements = ["Nonexistent endpoint"]
    httpx_mock.add_response(url=SentimentAnalyzer._MODEL_API_URL, status_code=404, text="Not found")

    with pytest.raises(NotFoundError):
        await sentiment_analyzer.analyze_sentiment(statements)


@pytest.mark.asyncio()
async def test_analyze_sentiment_internal_server_error(
    sentiment_analyzer: SentimentAnalyzer, httpx_mock: HTTPXMock
) -> None:
    """Test handling a 500 Internal Server Error."""
    statements = ["Server error"]
    httpx_mock.add_response(
        url=SentimentAnalyzer._MODEL_API_URL, status_code=500, text="Internal server error"
    )

    with pytest.raises(InternalServerError):
        await sentiment_analyzer.analyze_sentiment(statements)


@pytest.mark.asyncio()
async def test_analyze_sentiment_unexpected_error(
    sentiment_analyzer: SentimentAnalyzer, httpx_mock: HTTPXMock
) -> None:
    """Test handling an unexpected status code."""
    statements = ["Unexpected error"]
    httpx_mock.add_response(
        url=SentimentAnalyzer._MODEL_API_URL, status_code=418, text="I'm a teapot"
    )

    with pytest.raises(UnexpectedError):
        await sentiment_analyzer.analyze_sentiment(statements)


@pytest.mark.asyncio()
async def test_analyze_sentiment_response_validation_error(
    sentiment_analyzer: SentimentAnalyzer, httpx_mock: HTTPXMock
) -> None:
    """Test handling a response validation error."""
    statements = ["Validation error"]
    invalid_response = {
        "outputs": [
            [
                {"label": "positive", "score": 1.5},
                {"label": "neutral", "score": -0.1},
                {"label": "negative", "score": 0.6},
            ]
        ]
    }
    httpx_mock.add_response(url=SentimentAnalyzer._MODEL_API_URL, json=invalid_response)

    with pytest.raises(ResponseValidationError):
        await sentiment_analyzer.analyze_sentiment(statements)
