"""Errors for the sentiment analysis module."""


class InvalidPolarityError(Exception):
    """Raised when the polarity value is not within the expected range."""

    def __init__(self, polarity: float) -> None:
        super().__init__(f"Invalid polarity value: {polarity}")
        self.polarity = polarity


class ModelAPIError(Exception):
    """Base class for model API errors."""


class BadRequestError(ModelAPIError):
    """Raised when the API returns a 400 status code."""


class NotFoundError(ModelAPIError):
    """Raised when the API returns a 404 status code."""


class InternalServerError(ModelAPIError):
    """Raised when the API returns a 500 status code."""


class UnexpectedError(ModelAPIError):
    """Raised when the API returns an unexpected status code."""


class ResponseValidationError(ModelAPIError):
    """Raised when the response does not match the expected schema."""
