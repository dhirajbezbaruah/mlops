"""Fixtures for integration tests."""

import os

import pytest_asyncio

from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.sentiment_analysis import SentimentAnalyzer


@pytest_asyncio.fixture
def sentiment_analyzer() -> SentimentAnalyzer:
    """Fixture for the SentimentAnalyzer."""
    return SentimentAnalyzer(120)


@pytest_asyncio.fixture
async def feddit_client() -> FedditAPIClient:
    """Fixture for the Feddit API client."""
    return FedditAPIClient(os.environ["FEDDIT_API_BASE_URL"], 120)
