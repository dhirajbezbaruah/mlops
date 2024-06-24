"""Unit tests for the ``FedditAPIClient``."""

import pytest
from pytest_httpx import HTTPXMock

from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.feddit_client.errors import (
    APIVersionError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    ResponseValidationError,
    UnexpectedError,
)


@pytest.mark.asyncio()
async def test_get_version_success(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test getting the API version."""
    httpx_mock.add_response(url=f"{feddit_base_url}/api/v1/version", json={"version": "0.1.0"})
    version = await feddit_client.get_version()
    assert version == "0.1.0"


@pytest.mark.asyncio()
async def test_get_version_unsupported(feddit_base_url: str, httpx_mock: HTTPXMock) -> None:
    """Test API version is unsupported."""
    httpx_mock.add_response(url=f"{feddit_base_url}/api/v1/version", json={"version": "0.2.0"})
    with pytest.raises(APIVersionError):
        await FedditAPIClient(feddit_base_url).check_version()


@pytest.mark.asyncio()
async def test_get_subfeddits_success(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test getting a list of subfeddits."""
    response_data = {
        "limit": 10,
        "skip": 0,
        "subfeddits": [
            {"id": 1, "username": "user1", "title": "subfeddit1", "description": "desc1"},
            {"id": 2, "username": "user2", "title": "subfeddit2", "description": "desc2"},
        ],
    }

    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/subfeddits/?skip=0&limit=10", json=response_data
    )

    subfeddits = await feddit_client.get_subfeddits()
    assert len(subfeddits) == 2
    assert subfeddits[0].id == 1
    assert subfeddits[1].username == "user2"


@pytest.mark.asyncio()
@pytest.mark.parametrize(("skip", "limit"), [(0, 5), (5, 10)])
async def test_get_subfeddits_params(
    feddit_client: FedditAPIClient,
    feddit_base_url: str,
    httpx_mock: HTTPXMock,
    skip: int,
    limit: int,
) -> None:
    """Test getting a list of subfeddits with parameters."""
    response_data = {
        "limit": limit,
        "skip": skip,
        "subfeddits": [
            {"id": 1, "username": "user1", "title": "subfeddit1", "description": "desc1"},
        ],
    }

    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/subfeddits/?skip={skip}&limit={limit}", json=response_data
    )

    subfeddits = await feddit_client.get_subfeddits(skip=skip, limit=limit)
    assert subfeddits[0].id == 1
    assert len(subfeddits) == 1


@pytest.mark.asyncio()
async def test_get_subfeddit_info_success(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test getting detailed information of a specific subfeddit."""
    response_data = {
        "id": 1,
        "username": "user1",
        "title": "subfeddit1",
        "description": "desc1",
        "limit": 10,
        "skip": 0,
        "comments": [],
    }

    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/subfeddit/?subfeddit_id=1", json=response_data
    )

    subfeddit = await feddit_client.get_subfeddit_info(1)
    assert subfeddit.id == 1
    assert subfeddit.title == "subfeddit1"


@pytest.mark.asyncio()
async def test_get_subfeddit_comments_success(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test getting comments for a specific subfeddit."""
    response_data = {
        "subfeddit_id": 1,
        "limit": 10,
        "skip": 0,
        "comments": [
            {"id": 1, "username": "user1", "text": "comment1", "created_at": 1625247600},
            {"id": 2, "username": "user2", "text": "comment2", "created_at": 1625248600},
        ],
    }

    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/comments/?subfeddit_id=1&skip=0&limit=10", json=response_data
    )

    comments = await feddit_client.get_subfeddit_comments(1)
    assert len(comments) == 2
    assert comments[0].id == 1
    assert comments[1].username == "user2"


@pytest.mark.asyncio()
async def test_get_version_bad_request(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test handling of a bad request when getting the API version."""
    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/version", status_code=400, text="Bad Request"
    )
    with pytest.raises(BadRequestError):
        await feddit_client.get_version()


@pytest.mark.asyncio()
async def test_get_subfeddits_not_found(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test handling of a not found error when getting subfeddits."""
    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/subfeddits/?skip=0&limit=10",
        status_code=404,
        text="Not Found",
    )
    with pytest.raises(NotFoundError):
        await feddit_client.get_subfeddits()


@pytest.mark.asyncio()
async def test_get_subfeddit_info_internal_error(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test handling of an internal server error when getting subfeddit info."""
    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/subfeddit/?subfeddit_id=1",
        status_code=500,
        text="Internal Server Error",
    )
    with pytest.raises(InternalServerError):
        await feddit_client.get_subfeddit_info(1)


@pytest.mark.asyncio()
async def test_get_subfeddit_comments_unexpected_error(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test handling of an unexpected error when getting subfeddit comments."""
    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/comments/?subfeddit_id=1&skip=0&limit=10",
        status_code=418,
        text="Unexpected",
    )
    with pytest.raises(UnexpectedError):
        await feddit_client.get_subfeddit_comments(1)


@pytest.mark.asyncio()
async def test_get_version_validation_error(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test validation error when the version response schema is incorrect."""
    httpx_mock.add_response(url=f"{feddit_base_url}/api/v1/version", json={"ver": "0.1.0"})
    with pytest.raises(ResponseValidationError):
        await feddit_client.get_version()


@pytest.mark.asyncio()
async def test_get_subfeddits_validation_error(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test validation error when the subfeddits response schema is incorrect."""
    response_data = {
        "limit": 10,
        "skip": 0,
        "subfeddits": [
            {"id": 1, "username": "user1", "title": "subfeddit1"},
            {"id": 2, "username": "user2", "title": "subfeddit2", "description": "desc2"},
        ],
    }
    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/subfeddits/?skip=0&limit=10", json=response_data
    )
    with pytest.raises(ResponseValidationError):
        await feddit_client.get_subfeddits()


@pytest.mark.asyncio()
async def test_get_subfeddit_info_validation_error(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test validation error when the subfeddit info response schema is incorrect."""
    response_data = {
        "id": 1,
        "username": "user1",
        "title": "subfeddit1",
        "description": "desc1",
        "limit": 10,
        "skip": 0,
        "comments": [{"id": 1, "username": "user1", "text": "comment1"}],
    }
    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/subfeddit/?subfeddit_id=1", json=response_data
    )
    with pytest.raises(ResponseValidationError):
        await feddit_client.get_subfeddit_info(1)


@pytest.mark.asyncio()
async def test_get_subfeddit_comments_validation_error(
    feddit_client: FedditAPIClient, feddit_base_url: str, httpx_mock: HTTPXMock
) -> None:
    """Test validation error when the subfeddit comments response schema is incorrect."""
    response_data = {
        "subfeddit_id": 1,
        "limit": 10,
        "skip": 0,
        "comments": [{"id": 1, "username": "user1", "created_at": 1625247600}],
    }
    httpx_mock.add_response(
        url=f"{feddit_base_url}/api/v1/comments/?subfeddit_id=1&skip=0&limit=10", json=response_data
    )
    with pytest.raises(ResponseValidationError):
        await feddit_client.get_subfeddit_comments(1)
