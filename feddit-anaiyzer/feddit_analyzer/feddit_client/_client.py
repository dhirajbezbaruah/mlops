"""Client to interact with the Feddit API."""

from typing import Any, ClassVar

import httpx
from httpx import Response
from loguru import logger
from pydantic import BaseModel, ValidationError

from .errors import (
    APIVersionError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    ResponseValidationError,
    UnexpectedError,
)
from .schemas import (
    CommentInfo,
    CommentsResponse,
    SubfedditResponse,
    SubfedditsResponse,
    VersionResponse,
)

_API_PREFIX = "/api/v1"


class FedditAPIClient:
    """Client for interacting with the Feddit API.

    :param base_url: The base URL of the Feddit API.
    :param timeout: Default timeout for the HTTP requests in seconds. Default is 30
        seconds.
    """

    VALID_VERSIONS: ClassVar[list[str]] = ["0.1.0"]

    def __init__(self, base_url: str, timeout: float = 30) -> None:
        self._base_url = base_url
        self._timeout = timeout

    async def check_version(self) -> None:
        """Check if the API version is supported."""
        if await self.get_version() not in self.VALID_VERSIONS:
            raise APIVersionError("API version is not supported.")

    async def get_version(self, timeout: float | None = None) -> dict[str, Any]:
        """Get Feddit API version.

        :param timeout: Timeout for the HTTP request in seconds. If not provided, the
            client default timeout will be used. Default is ``None``.
        :return: Feddit API version.
        """
        logger.debug("Getting API version")
        async with httpx.AsyncClient(timeout=self._choose_timeout(timeout)) as client:
            response = await client.get(f"{self._base_url}{_API_PREFIX}/version")
        return _handle_response(response, VersionResponse).version

    async def get_subfeddits(
        self, skip: int = 0, limit: int = 10, timeout: float | None = None
    ) -> list[SubfedditResponse]:
        """Get a list of subfeddits.

        :param skip: The number of subfeddits to skip. Default is 0.
        :param limit: The maximum number of subfeddits to return. Default is 10.
        :param timeout: Timeout for the HTTP request in seconds. If not provided, the
            client default timeout will be used.
        :raises ValueError: If the skip value is negative.
        :raises APIClientError: If an error occurs while fetching the subfeddits.
        :return: A list of subfeddits.
        """
        logger.info("Getting subfeddits skipping {} with limit {}", skip, limit)
        if skip < 0:
            raise ValueError("Skip value cannot be negative.")

        params = {"skip": skip, "limit": limit}

        async with httpx.AsyncClient(timeout=self._choose_timeout(timeout)) as client:
            response = await client.get(f"{self._base_url}{_API_PREFIX}/subfeddits/", params=params)
        return _handle_response(response, SubfedditsResponse).subfeddits

    async def get_subfeddit_info(
        self, subfeddit_id: int, timeout: float | None = None
    ) -> SubfedditResponse:
        """Get detailed information of a specific subfeddit.

        :param subfeddit_id: The ID of the subfeddit.
        :param timeout: Timeout for the HTTP request in seconds. If not provided, the
            client default timeout will be used. Default is ``None``.
        :raises APIClientError: If an error occurs while fetching the subfeddit
            information.
        :return: Detailed information of the subfeddit.
        """
        logger.info("Getting subfeddit info for subfeddit {}", subfeddit_id)
        params = {"subfeddit_id": subfeddit_id}

        async with httpx.AsyncClient(timeout=self._choose_timeout(timeout)) as client:
            response = await client.get(f"{self._base_url}{_API_PREFIX}/subfeddit/", params=params)
        return _handle_response(response, SubfedditResponse)

    async def get_subfeddit_comments(
        self, subfeddit_id: int, skip: int = 0, limit: int = 10, timeout: float | None = None
    ) -> list[CommentInfo]:
        """Get comments for a specific subfeddit.

        :param subfeddit_id: The ID of the subfeddit.
        :param skip: The number of subfeddits to skip. Default is 0.
        :param limit: The maximum number of subfeddits to return. Default is 10.
        :param timeout: Timeout for the HTTP request in seconds. If not provided, the
            client default timeout will be used. Default is ``None``.
        :raises APIClientError: If an error occurs while fetching the comments.
        :return: A list of comments.
        """
        logger.info(
            "Getting comments for subfeddit {}, skipping {} with limit {}",
            subfeddit_id,
            skip,
            limit,
        )

        params = {"subfeddit_id": subfeddit_id, "skip": skip, "limit": limit}

        async with httpx.AsyncClient(timeout=self._choose_timeout(timeout)) as client:
            response = await client.get(f"{self._base_url}{_API_PREFIX}/comments/", params=params)
        return _handle_response(response, CommentsResponse).comments

    def _choose_timeout(self, provided: float | None) -> float:
        """Choose the timeout value to use for the HTTP request.

        :param provided: The timeout value provided by the user.
        :return: The timeout value to use.
        """
        return provided if provided is not None else self._timeout


def _handle_response(response: Response, model: type[BaseModel]) -> BaseModel:
    """Handle the HTTP response from the API.

    :param response: The HTTP response object.
    :param model: The Pydantic model to validate the response against.
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
                return model.model_validate(content)

            except ValidationError as exc:
                raise ResponseValidationError(
                    "Response does not match the expected schema."
                ) from exc

        case 400:
            raise BadRequestError(f"Bad Request: {response.text}.")
        case 404:
            raise NotFoundError(f"Not Found: {response.text}.")
        case 500:
            raise InternalServerError(f"Internal Server Error: {response.text}.")
        case _:
            raise UnexpectedError(f"Unexpected Error: {response.status_code} - {response.text}.")
