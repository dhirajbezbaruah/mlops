"""Implementation of the core functionlaity served by the API."""

from cachetools import TTLCache
from fastapi import HTTPException
from loguru import logger

from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.feddit_client.errors import NotFoundError
from feddit_analyzer.sentiment_analysis import SentimentAnalyzer

from ._schemas import CommentSentiment

_QUERY_LIMIT = 25
_QUERY_BATCH_SIZE = 5000

_CACHE_MAX_SIZE = 1000
_CACHE_TTL = 600

subfeddit_cache = TTLCache(_CACHE_MAX_SIZE, _CACHE_TTL)


async def get_subfeddit_id(subfeddit_title: str, feddit_client: FedditAPIClient) -> int:
    """Get the ID of a subfeddit from its title.

    It is required to look exhaustively through all subfeddits to find the desired one.
    This can be improved by adding a search functionality to the Feddit API.

    This function includes cache functionality to avoid unnecessary search.

    :param subfeddit_title: The title of the subfeddit.
    :param feddit_client: The Feddit API client.
    :return: The ID of the subfeddit.
    """
    logger.info("Getting subfeddit list to find ID")

    if subfeddit_title in subfeddit_cache:
        cached_id = subfeddit_cache[subfeddit_title]

        try:
            logger.debug("Getting subfeddit info from cache")
            await feddit_client.get_subfeddit_info(cached_id)
            logger.info("Subfeddit with title '{}' found", subfeddit_title)
            logger.debug("Subfeddit ID found: {}", cached_id)
            return cached_id

        except NotFoundError:
            logger.info("Subfeddit with ID {} not found", cached_id)
            del subfeddit_cache[subfeddit_title]

            logger.info("Removed subfeddit {} from cache")

    skip = 0

    while True:
        subfeddit_batch = await feddit_client.get_subfeddits(skip=skip, limit=_QUERY_BATCH_SIZE)

        logger.debug("Received subfeddit batch: {}", subfeddit_batch)

        for subfeddit in subfeddit_batch:
            subfeddit_cache[subfeddit.title] = subfeddit.id
            if subfeddit.title == subfeddit_title:
                logger.info("Subfeddit with title '{}' found", subfeddit_title)
                logger.debug("Subfeddit ID found: {}", subfeddit.id)
                return subfeddit.id

        if len(subfeddit_batch) < _QUERY_BATCH_SIZE:
            break

        skip += _QUERY_BATCH_SIZE

    raise HTTPException(
        status_code=404, detail=f"Subfeddit with title '{subfeddit_title}' not found"
    )


async def analyze_comments_sentiment(
    subfeddit_id: int,
    min_datetime: int | None,
    max_datetime: int | None,
    sort_by_polarity: bool,
    feddit_client: FedditAPIClient,
    sentiment_analyzer: SentimentAnalyzer,
) -> list[CommentSentiment]:
    """Analyze the sentiment of comments from a specific subfeddit.

    Initial comment retrieval needs to be exhaustive, as from testing the API, it seems
    that they are not ordered by date.

    :param subfeddit_id: The ID of the subfeddit.
    :param min_datetime: If given, the minimum datetime for comments. It has to be in Unix
        epochs.
    :param max_datetime: If given, the maximum datetime for comments. It has to be in Unix
        epochs.
    :param sort_by_polarity: Whether to sort comments by polarity.
    :return: A list of comments with their sentiment analysis.
    """
    logger.info("Extracting comments for sentiment analysis")
    comments = await _get_all_comments(subfeddit_id, min_datetime, max_datetime, feddit_client)
    logger.info("Received {} comments for sentiment analysis.", len(comments))
    logger.debug("Received comments: {}", comments)

    if not comments:
        logger.warning("No comments found.")
        return []

    logger.info("Analyzing sentiment of comments")
    comments_sentiment = await sentiment_analyzer.analyze_sentiment(
        [comment.text for comment in comments], 60
    )

    logger.info("Sentiment analysis of comments completed.")
    logger.debug("Sentiment analysis of comments: {}", comments_sentiment)

    responses = [
        CommentSentiment(
            comment_id=comment.id,
            comment=comment.text,
            polarity=sentiment.polarity,
            classification=sentiment.sentiment.value,
        )
        for comment, sentiment in zip(comments, comments_sentiment, strict=False)
    ]

    if sort_by_polarity:
        logger.info("Sorting comments by polarity")
        responses.sort(key=lambda comment: comment.polarity, reverse=True)

    return responses


async def _get_all_comments(
    subfeddit_id: int,
    min_datetime: int | None,
    max_datetime: int | None,
    feddit_client: FedditAPIClient,
) -> list[CommentSentiment]:
    """Get all comments from a specific subfeddit.

    Comments will be sorted from newest to oldest.

    Comment extraction is done in batches of 25 comments.

    :param subfeddit_id: The ID of the subfeddit.
    :param min_datetime: If given, the minimum datetime for comments. It has to be in Unix
        epochs.
    :param max_datetime: If given, the maximum datetime for comments. It has to be in Unix
        epochs.
    :param feddit_client: The Feddit API client.
    :return: A list of comments from the subfeddit.
    """
    logger.info("Getting comments for subfeddit {}", subfeddit_id)

    skip = 0
    comments = []

    while True:
        comment_batch = await feddit_client.get_subfeddit_comments(
            subfeddit_id, skip=skip, limit=_QUERY_BATCH_SIZE
        )

        stop = len(comment_batch) < _QUERY_BATCH_SIZE

        comment_batch = [
            comment
            for comment in comment_batch
            if (min_datetime is None or comment.created_at >= min_datetime)
            and (max_datetime is None or comment.created_at <= max_datetime)
        ]

        comments.extend(comment_batch)

        comments.sort(key=lambda comment: comment.created_at, reverse=True)

        comments = comments[:_QUERY_LIMIT]

        if stop:
            break

        skip += _QUERY_BATCH_SIZE

    return comments
