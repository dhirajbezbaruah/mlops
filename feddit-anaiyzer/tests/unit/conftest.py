"""Fixtures for unit tests."""

import random
from unittest.mock import AsyncMock

import pytest_asyncio

from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.feddit_client.schemas import CommentInfo
from feddit_analyzer.sentiment_analysis import SentimentAnalyzer
from feddit_analyzer.sentiment_analysis.schemas import SentimentAnalysis
from feddit_analyzer.sentiment_analysis.sentiment import Sentiment


@pytest_asyncio.fixture
async def suffedit_title() -> str:
    """Fixture for the title of a subfeddit."""
    return "existing_subfeddit"


@pytest_asyncio.fixture
async def subfeddit_id() -> int:
    """Fixture for the ID of a subfeddit."""
    return 123


@pytest_asyncio.fixture
async def mock_feddit_client() -> FedditAPIClient:
    """Fixture for the Feddit API client."""
    return AsyncMock(FedditAPIClient)


@pytest_asyncio.fixture
async def mock_sentiment_analyzer(monkeypatch: callable) -> SentimentAnalyzer:
    """Fixture for the SentimentAnalyzer."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake_api_key")
    return AsyncMock(SentimentAnalyzer)


@pytest_asyncio.fixture
async def single_comment() -> list[CommentInfo]:
    """Fixture for a list of comments."""
    return [CommentInfo(id=1, text="This is a comment.", created_at=1625247600, username="user1")]


@pytest_asyncio.fixture
async def single_sentiment() -> list[SentimentAnalysis]:
    """Fixture for a list of sentiments."""
    return [
        SentimentAnalysis(
            statement="This is a comment.", polarity=0.5, sentiment=Sentiment.POSITIVE
        )
    ]


@pytest_asyncio.fixture
async def comments() -> list[CommentInfo]:
    """Fixture for a list of comments with different creation dates."""
    return [
        CommentInfo(id=1, text="Comment 1", created_at=1625247600, username="user1"),
        CommentInfo(id=2, text="Comment 2", created_at=1625248600, username="user2"),
    ]


@pytest_asyncio.fixture
async def sentiments() -> list[SentimentAnalysis]:
    """Fixture for a list of sentiments with different polarities."""
    return [
        SentimentAnalysis(statement="Comment 1", polarity=0.2, sentiment=Sentiment.POSITIVE),
        SentimentAnalysis(statement="Comment 2", polarity=0.8, sentiment=Sentiment.NEGATIVE),
    ]


@pytest_asyncio.fixture
def feddit_base_url() -> str:
    """Base fake URL for unit tests."""
    return "http://fake.url"


@pytest_asyncio.fixture
def feddit_client(feddit_base_url: str) -> FedditAPIClient:
    """Fixture for the Feddit API client."""
    return FedditAPIClient(feddit_base_url)


@pytest_asyncio.fixture
def sentiment_analyzer(monkeypatch: callable) -> SentimentAnalyzer:
    """Fixture for the SentimentAnalyzer."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake_api_key")
    return SentimentAnalyzer()


@pytest_asyncio.fixture
def mock_model_response() -> callable:
    """Fixture to return a mock model response with random values."""

    def _mock_response(statements: str | list[str]) -> list[list[dict[str, str | float]]]:
        if isinstance(statements, str):
            statements = [statements]
        return [
            [
                {"label": "positive", "score": random.uniform(0, 1)},
                {"label": "neutral", "score": random.uniform(0, 1)},
                {"label": "negative", "score": random.uniform(0, 1)},
            ]
            for _ in statements
        ]

    return _mock_response
