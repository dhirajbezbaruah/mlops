"""API endpoint specification for the sentiment analysis application.

Requires the definition of the following environment variables:
- ``FEDDIT_API_BASE_URL``: The base URL for the Feddit API.
- ``HUGGINGFACE_API_KEY``: The API key for the Hugging Face API.
"""

import os

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from feddit_analyzer import __version__
from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.feddit_client.errors import (
    APIClientError,
    APIVersionError,
    SubfedditNotFoundError,
)
from feddit_analyzer.feddit_client.errors import (
    BadRequestError as FedditBadRequestError,
)
from feddit_analyzer.feddit_client.errors import (
    InternalServerError as FedditInternalServerError,
)
from feddit_analyzer.feddit_client.errors import (
    NotFoundError as FedditNotFoundError,
)
from feddit_analyzer.feddit_client.errors import (
    ResponseValidationError as FedditResponseValidationError,
)
from feddit_analyzer.feddit_client.errors import (
    UnexpectedError as FedditUnexpectedError,
)
from feddit_analyzer.sentiment_analysis import SentimentAnalyzer
from feddit_analyzer.sentiment_analysis.errors import (
    BadRequestError as ModelBadRequestError,
)
from feddit_analyzer.sentiment_analysis.errors import (
    InternalServerError as ModelInternalServerError,
)
from feddit_analyzer.sentiment_analysis.errors import (
    InvalidPolarityError,
    ModelAPIError,
)
from feddit_analyzer.sentiment_analysis.errors import (
    NotFoundError as ModelNotFoundError,
)
from feddit_analyzer.sentiment_analysis.errors import (
    ResponseValidationError as ModelResponseValidationError,
)
from feddit_analyzer.sentiment_analysis.errors import (
    UnexpectedError as ModelUnexpectedError,
)

from . import _core as core
from ._schemas import (
    CommentSentimentIDRequest,
    CommentSentimentIDResponse,
    CommentSentimentRequest,
    CommentSentimentResponse,
    VersionResponse,
)

_API_DESCRIPTION = """
This API allows users to analyze the sentiment of text statements from Feddit.

You can:
- Get the version of the API.
"""

_API_PREFIX = "/api/v1"

_TAGS_METADATA = [
    {"name": "base", "description": "Base API endpoints."},
    {"name": "subfeddit", "description": "Endpoints for sentiment analysis of Feddit comments."},
]

app = FastAPI(
    title="Feddit Analyzer",
    summary="API for sentiment analysis of text statements from Feddit.",
    description=_API_DESCRIPTION,
    version=__version__,
    contact={"name": "Martín Martínez, Daniel", "email": "dantiana98@gmail.com"},
    license_info={"name": "MIT License", "url": "https://opensource.org/license/MIT"},
    openapi_tags=_TAGS_METADATA,
)


def get_feddit_api_client() -> FedditAPIClient:
    """Get the Feddit API client.

    :raises RuntimeError: If the environment variable ``FEDDIT_API_BASE_URL`` is not set.
    :return: The Feddit API client.
    """
    try:
        base_url = os.environ["FEDDIT_API_BASE_URL"]
    except KeyError as exc:
        raise RuntimeError("Environment variable FEDDIT_API_BASE_URL is not set.") from exc

    return FedditAPIClient(base_url=base_url)


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get the sentiment analyzer.

    :return: The sentiment analyzer.
    """
    return SentimentAnalyzer()


@app.get(
    f"{_API_PREFIX}/version",
    name="version",
    response_model=VersionResponse,
    summary="Get API version.",
    description="Get the version of the deployed API.",
    response_description="API version of deployed app.",
    tags=["base"],
)
async def get_version() -> dict[str, str]:
    """Get the version of the API.

    :return: The version of the API.
    """
    return {"version": __version__}


@app.post(
    f"{_API_PREFIX}/classify_comments/subfeddit_id",
    name="comments_id",
    response_model=CommentSentimentIDResponse,
    summary="Get sentiment analysis for Feddit comments from a specific subfeddit ID.",
    description=(
        "Get sentiment analysis for comments from a specific subfeddit. It can return the 25 most "
        "recent comments at most within the specified time range (if given)."
    ),
    response_description=(
        "Sentiment analysis of 25 most recent comments from a specific subfeddit in a time range."
    ),
    tags=["subfeddit"],
)
async def get_classified_comments_from_id(
    request: CommentSentimentIDRequest,
    feddit_client: FedditAPIClient = Depends(get_feddit_api_client),
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
) -> CommentSentimentIDResponse:
    """Get sentiment analysis for comments from a specific subfeddit.

    :param request: The request for sentiment analysis of comments.
    :return: The sentiment analysis of the comments.
    """
    logger.info("Processing request for sentiment analysis of comments: {}", request)
    sentiments = await core.analyze_comments_sentiment(
        request.subfeddit_id,
        request.min_datetime,
        request.max_datetime,
        request.sort_by_polarity,
        feddit_client,
        sentiment_analyzer,
    )
    logger.info("Sentiment analysis of comments completed.")
    logger.debug("Sentiment analysis of comments: {}", sentiments)

    return CommentSentimentIDResponse(subfeddit_id=request.subfeddit_id, comments=sentiments)


@app.post(
    f"{_API_PREFIX}/classify_comments/subfeddit_title",
    name="comments_title",
    response_model=CommentSentimentResponse,
    summary="Get sentiment analysis for Feddit comments from a specific subfeddit title.",
    description=(
        "Get sentiment analysis for comments from a specific subfeddit. It can return the 25 most "
        "recent comments at most within the specified time range (if given)."
    ),
    response_description=(
        "Sentiment analysis of 25 most recent comments from a specific subfeddit in a time range."
    ),
    tags=["subfeddit"],
)
async def get_classified_comment_from_title(
    request: CommentSentimentRequest,
    feddit_client: FedditAPIClient = Depends(get_feddit_api_client),
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
) -> CommentSentimentResponse:
    """Get sentiment analysis for comments from a specific subfeddit title.

    :param request: The request for sentiment analysis of comments.
    :return: The sentiment analysis of the comments.
    """
    logger.info("Searching for subfeddit ID for title: {}", request.subfeddit_title)

    subfeddit_id = await core.get_subfeddit_id(request.subfeddit_title, feddit_client)

    logger.info(
        "Processing request for sentiment analysis of comments from subfeddit ID: {}", subfeddit_id
    )
    sentiments = await core.analyze_comments_sentiment(
        subfeddit_id,
        request.min_datetime,
        request.max_datetime,
        request.sort_by_polarity,
        feddit_client,
        sentiment_analyzer,
    )
    logger.info("Sentiment analysis of comments completed.")
    logger.debug("Sentiment analysis of comments: {}", sentiments)

    return CommentSentimentResponse(
        subfeddit_id=subfeddit_id,
        subfeddit_title=request.subfeddit_title,
        comments=sentiments,
    )


@app.exception_handler(FedditBadRequestError)
async def handle_feddit_bad_request_error(
    request: Request, exc: FedditBadRequestError
) -> JSONResponse:
    """Handle Feddit API bad request errors.

    :param request: The request that caused the error.
    :param exc: The FedditBadRequestError exception.
    :return: JSON response with the error message and status code 400.
    """
    logger.error("Feddit API bad request error: {}", exc)
    return JSONResponse(
        {
            "message": "The request made to the Feddit API was invalid.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=400,
    )


@app.exception_handler(FedditNotFoundError)
async def handle_feddit_not_found_error(request: Request, exc: FedditNotFoundError) -> JSONResponse:
    """Handle Feddit API not found errors.

    :param request: The request that caused the error.
    :param exc: The FedditNotFoundError exception.
    :return: JSON response with the error message and status code 404.
    """
    logger.error("Feddit API not found error: {}", exc)
    return JSONResponse(
        {
            "message": "The requested resource could not be found in the Feddit API.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=404,
    )


@app.exception_handler(FedditInternalServerError)
async def handle_feddit_internal_server_error(
    request: Request, exc: FedditInternalServerError
) -> JSONResponse:
    """Handle Feddit API internal server errors.

    :param request: The request that caused the error.
    :param exc: The FedditInternalServerError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("Feddit API internal server error: {}", exc)
    return JSONResponse(
        {
            "message": "The Feddit API encountered an internal server error.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(FedditResponseValidationError)
async def handle_feddit_response_validation_error(
    request: Request, exc: FedditResponseValidationError
) -> JSONResponse:
    """Handle Feddit API response validation errors.

    :param request: The request that caused the error.
    :param exc: The FedditResponseValidationError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("Feddit API response validation error: {}", exc)
    return JSONResponse(
        {
            "message": "The response from the Feddit API did not match the expected format.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(FedditUnexpectedError)
async def handle_feddit_unexpected_error(
    request: Request, exc: FedditUnexpectedError
) -> JSONResponse:
    """Handle unexpected Feddit API errors.

    :param request: The request that caused the error.
    :param exc: The FedditUnexpectedError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("Feddit API unexpected error: {}", exc)
    return JSONResponse(
        {
            "message": "An unexpected error occurred while communicating with the Feddit API.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(SubfedditNotFoundError)
async def handle_subfeddit_not_found_error(
    request: Request, exc: SubfedditNotFoundError
) -> JSONResponse:
    """Handle errors when a subfeddit is not found.

    :param request: The request that caused the error.
    :param exc: The SubfedditNotFoundError exception.
    :return: JSON response with the error message and status code 404.
    """
    logger.error("Subfeddit not found error: {}", exc)
    return JSONResponse(
        {
            "message": (
                "The specified subfeddit could not be found. Please check the subfeddit and "
                "try again."
            ),
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=404,
    )


@app.exception_handler(APIClientError)
async def handle_api_client_error(request: Request, exc: APIClientError) -> JSONResponse:
    """Handle generic API client errors.

    :param request: The request that caused the error.
    :param exc: The APIClientError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("API client error: {}", exc)
    return JSONResponse(
        {
            "message": "An error occurred with the Feddit API client. Please try again later.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(APIVersionError)
async def handle_api_version_error(request: Request, exc: APIVersionError) -> JSONResponse:
    """Handle API version errors.

    :param request: The request that caused the error.
    :param exc: The APIVersionError exception.
    :return: JSON response with the error message and status code 400.
    """
    logger.error("API version error: {}", exc)
    return JSONResponse(
        {
            "message": "The API version being used is not supported.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=400,
    )


@app.exception_handler(ModelInternalServerError)
async def handle_model_internal_server_error(
    request: Request, exc: ModelInternalServerError
) -> JSONResponse:
    """Handle internal server errors from the sentiment analysis model API.

    :param request: The request that caused the error.
    :param exc: The ModelInternalServerError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("Model API internal server error: {}", exc)
    return JSONResponse(
        {
            "message": "The sentiment analysis model API encountered an internal server error.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(InvalidPolarityError)
async def handle_invalid_polarity_error(
    request: Request, exc: InvalidPolarityError
) -> JSONResponse:
    """Handle errors for invalid polarity values.

    :param request: The request that caused the error.
    :param exc: The InvalidPolarityError exception.
    :return: JSON response with the error message and status code 400.
    """
    logger.error("Invalid polarity error: {}", exc)
    return JSONResponse(
        {
            "message": f"The provided polarity value '{exc.polarity}' is invalid.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=400,
    )


@app.exception_handler(ModelAPIError)
async def handle_model_api_error(request: Request, exc: ModelAPIError) -> JSONResponse:
    """Handle generic errors from the sentiment analysis model API.

    :param request: The request that caused the error.
    :param exc: The ModelAPIError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("Model API error: {}", exc)
    return JSONResponse(
        {
            "message": "An error occurred with the sentiment analysis model API.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(ModelResponseValidationError)
async def handle_model_response_validation_error(
    request: Request, exc: ModelResponseValidationError
) -> JSONResponse:
    """Handle response validation errors from the sentiment analysis model API.

    :param request: The request that caused the error.
    :param exc: The ModelResponseValidationError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("Model API response validation error: {}", exc)
    return JSONResponse(
        {
            "message": (
                "The response from the sentiment analysis model API did not match the expected "
                "format."
            ),
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(ModelUnexpectedError)
async def handle_model_unexpected_error(
    request: Request, exc: ModelUnexpectedError
) -> JSONResponse:
    """Handle unexpected errors from the sentiment analysis model API.

    :param request: The request that caused the error.
    :param exc: The ModelUnexpectedError exception.
    :return: JSON response with the error message and status code 500.
    """
    logger.error("Model API unexpected error: {}", exc)
    return JSONResponse(
        {
            "message": "An unexpected error occurred with the sentiment analysis model API.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=500,
    )


@app.exception_handler(ModelBadRequestError)
async def handle_model_bad_request_error(
    request: Request, exc: ModelBadRequestError
) -> JSONResponse:
    """Handle bad request errors from the sentiment analysis model API.

    :param request: The request that caused the error.
    :param exc: The ModelBadRequestError exception.
    :return: JSON response with the error message and status code 400.
    """
    logger.error("Model API bad request error: {}", exc)
    return JSONResponse(
        {
            "message": "The request made to the sentiment analysis model API was invalid.",
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=400,
    )


@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found_error(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """Handle not found errors from the sentiment analysis model API.

    :param request: The request that caused the error.
    :param exc: The ModelNotFoundError exception.
    :return: JSON response with the error message and status code 404.
    """
    logger.error("Model API not found error: {}", exc)
    return JSONResponse(
        {
            "message": (
                "The requested resource could not be found in the sentiment analysis model API. "
                "Please verify the model URI."
            ),
            "details": str(exc),
            "type": type(exc).__name__,
        },
        status_code=404,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exceptions.

    :param request: The request that caused the error.
    :param exc: The Exception that was raised.
    :return: JSON response with a generic error message and status code 500.
    """
    logger.error(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred.",
            "details": str(exc),
            "type": "InternalServerError",
        },
    )
