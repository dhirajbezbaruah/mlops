"""Integration tests for the Feddit API client."""

import pytest

from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.feddit_client.schemas import (
    CommentInfo,
    SubfedditInfo,
    SubfedditResponse,
)


@pytest.mark.asyncio()
async def test_get_version_success(feddit_client: FedditAPIClient) -> None:
    """Test getting the API version."""
    version = await feddit_client.get_version()
    assert version == "0.1.0"


@pytest.mark.asyncio()
async def check_version(feddit_client: FedditAPIClient) -> None:
    """Test API version is supported."""
    await feddit_client.check_version()


@pytest.mark.asyncio()
async def test_get_subfeddits_success(feddit_client: FedditAPIClient) -> None:
    """Test getting a list of subfeddits."""
    subfeddits = await feddit_client.get_subfeddits()
    assert isinstance(subfeddits, list)
    assert len(subfeddits) > 0
    assert all(isinstance(subfeddit, SubfedditInfo) for subfeddit in subfeddits)


@pytest.mark.asyncio()
@pytest.mark.parametrize("subfeddit_id", [1, 2, 3])
async def test_get_subfeddit_info_success(
    feddit_client: FedditAPIClient, subfeddit_id: int
) -> None:
    """Test getting detailed information of a specific subfeddit."""
    subfeddit = await feddit_client.get_subfeddit_info(subfeddit_id)
    assert isinstance(subfeddit, SubfedditResponse)
    assert subfeddit.id == subfeddit_id
    assert len(subfeddit.comments) > 0
    assert all(isinstance(comment, CommentInfo) for comment in subfeddit.comments)


@pytest.mark.asyncio()
@pytest.mark.parametrize("subfeddit_id", [1, 2, 3])
async def test_get_subfeddit_comments_success(
    feddit_client: FedditAPIClient, subfeddit_id: int
) -> None:
    """Test getting comments for a specific subfeddit."""
    comments = await feddit_client.get_subfeddit_comments(subfeddit_id)
    assert isinstance(comments, list)
    assert len(comments) > 0
    assert all(isinstance(comment, CommentInfo) for comment in comments)
