"""Unit tests for API core functionality."""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from feddit_analyzer.api._core import (
    analyze_comments_sentiment,
    get_subfeddit_id,
    subfeddit_cache,
)
from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.feddit_client.schemas import CommentInfo
from feddit_analyzer.sentiment_analysis import SentimentAnalyzer
from feddit_analyzer.sentiment_analysis.schemas import SentimentAnalysis


@pytest.mark.asyncio()
async def test_get_subfeddit_id_cached(
    mock_feddit_client: FedditAPIClient, suffedit_title: str, subfeddit_id: int
) -> None:
    """Test getting subfeddit ID from cache."""
    subfeddit_cache[suffedit_title] = subfeddit_id
    mock_feddit_client.get_subfeddit_info = AsyncMock()

    result = await get_subfeddit_id(suffedit_title, mock_feddit_client)
    assert result == subfeddit_id
    mock_feddit_client.get_subfeddit_info.assert_called_once_with(subfeddit_id)


@pytest.mark.asyncio()
async def test_get_subfeddit_id_not_cached(
    mock_feddit_client: FedditAPIClient, suffedit_title: str, subfeddit_id: int
) -> None:
    """Test getting subfeddit ID when not cached."""
    mock_feddit_client.get_subfeddits = AsyncMock(
        return_value=[AsyncMock(title=suffedit_title, id=subfeddit_id)]
    )

    result = await get_subfeddit_id(suffedit_title, mock_feddit_client)
    assert result == subfeddit_id


@pytest.mark.asyncio()
async def test_get_subfeddit_id_not_found(mock_feddit_client: FedditAPIClient) -> None:
    """Test handling subfeddit not found."""
    subfeddit_title = "non_existing_subfeddit"
    mock_feddit_client.get_subfeddits = AsyncMock(return_value=[])

    with pytest.raises(HTTPException):
        await get_subfeddit_id(subfeddit_title, mock_feddit_client)


@pytest.mark.asyncio()
async def test_analyze_comments_sentiment(
    mock_feddit_client: FedditAPIClient,
    mock_sentiment_analyzer: SentimentAnalyzer,
    subfeddit_id: int,
    single_comment: list[CommentInfo],
    single_sentiment: list[SentimentAnalysis],
) -> None:
    """Test analyzing sentiment of comments."""
    mock_feddit_client.get_subfeddit_comments = AsyncMock(return_value=single_comment)
    mock_sentiment_analyzer.analyze_sentiment = AsyncMock(return_value=single_sentiment)

    result = await analyze_comments_sentiment(
        subfeddit_id, None, None, False, mock_feddit_client, mock_sentiment_analyzer
    )
    assert len(result) == 1
    assert result[0].comment_id == single_comment[0].id
    assert result[0].comment == single_comment[0].text
    assert result[0].polarity == single_sentiment[0].polarity
    assert result[0].classification == single_sentiment[0].sentiment


@pytest.mark.asyncio()
async def test_analyze_comments_sentiment_no_comments(
    mock_feddit_client: FedditAPIClient,
    mock_sentiment_analyzer: SentimentAnalyzer,
    subfeddit_id: int,
) -> None:
    """Test analyzing sentiment with no comments found."""
    mock_feddit_client.get_subfeddit_comments = AsyncMock(return_value=[])

    result = await analyze_comments_sentiment(
        subfeddit_id, None, None, False, mock_feddit_client, mock_sentiment_analyzer
    )
    assert result == []


@pytest.mark.asyncio()
async def test_analyze_comments_sentiment_filter_by_date(
    mock_feddit_client: FedditAPIClient,
    mock_sentiment_analyzer: SentimentAnalyzer,
    subfeddit_id: int,
    comments: list[CommentInfo],
    sentiments: list[SentimentAnalysis],
) -> None:
    """Test analyzing sentiment with comments filtered by creation date."""
    mock_feddit_client.get_subfeddit_comments = AsyncMock(return_value=comments)
    mock_sentiment_analyzer.analyze_sentiment = AsyncMock(return_value=sentiments)

    min_datetime = 1625248000
    max_datetime = 1625249000

    result = await analyze_comments_sentiment(
        subfeddit_id,
        min_datetime,
        max_datetime,
        False,
        mock_feddit_client,
        mock_sentiment_analyzer,
    )
    assert len(result) == 1
    assert result[0].comment_id == 2
    assert result[0].comment == "Comment 2"


@pytest.mark.asyncio()
async def test_analyze_comments_sentiment_sort_by_polarity(
    mock_feddit_client: FedditAPIClient,
    mock_sentiment_analyzer: SentimentAnalyzer,
    subfeddit_id: int,
    comments: list[CommentInfo],
    sentiments: list[SentimentAnalysis],
) -> None:
    """Test analyzing sentiment with comments sorted by polarity."""
    mock_feddit_client.get_subfeddit_comments = AsyncMock(return_value=comments)
    mock_sentiment_analyzer.analyze_sentiment = AsyncMock(return_value=sentiments)

    result = await analyze_comments_sentiment(
        subfeddit_id, None, None, True, mock_feddit_client, mock_sentiment_analyzer
    )
    assert len(result) == 2
    assert result[0].comment_id == 1
    assert result[0].polarity == 0.8
    assert result[1].comment_id == 2
    assert result[1].polarity == 0.2
