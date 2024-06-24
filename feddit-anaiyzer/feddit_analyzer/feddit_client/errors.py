"""Errors for the Feddit API client."""


class APIClientError(Exception):
    """Base class for API client errors."""


class APIVersionError(APIClientError):
    """Raised when the API version is not supported."""


class BadRequestError(APIClientError):
    """Raised when the API returns a 400 status code."""


class NotFoundError(APIClientError):
    """Raised when the API returns a 404 status code."""


class InternalServerError(APIClientError):
    """Raised when the API returns a 500 status code."""


class UnexpectedError(APIClientError):
    """Raised when the API returns an unexpected status code."""


class ResponseValidationError(APIClientError):
    """Raised when the response does not match the expected schema."""


class SubfedditNotFoundError(APIClientError):
    """Raised when the subfeddit is not found."""
