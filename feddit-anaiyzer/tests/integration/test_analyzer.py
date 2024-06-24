"""Integration tests for the SentimentAnalyzer.

Some error cases are skipped as they are not feasible to simulate in this test
environment. Those could be mocked but that is already covered in the unit tests.

The model performance is not tested here as it is not the responsibility of this
implementation. Nonetheless, given a test dataset we could keep track of the model
performance and ensure it does performed as promised by the model provider. Tools like
Weights & Biases could be used for this.
"""

import pytest
from loguru import logger

from feddit_analyzer.sentiment_analysis import SentimentAnalyzer
from feddit_analyzer.sentiment_analysis.errors import (
    BadRequestError,
    NotFoundError,
)
from feddit_analyzer.sentiment_analysis.schemas import SentimentAnalysis


@pytest.mark.asyncio()
async def test_analyze_sentiment_success(sentiment_analyzer: SentimentAnalyzer) -> None:
    """Test analyzing sentiment successfully with real API."""
    statements = ["I love this!", "This is bad.", "I feel neutral."]
    result = await sentiment_analyzer.analyze_sentiment(statements)

    assert len(result) == len(statements)
    assert all(
        statement == res.statement for statement, res in zip(statements, result, strict=True)
    )
    assert all(isinstance(res, SentimentAnalysis) for res in result)
    assert all(res.sentiment in ["positive", "negative"] for res in result)
    assert all(-1 <= res.polarity <= 1 for res in result)


@pytest.mark.asyncio()
async def test_analyze_sentiment_single_statement(sentiment_analyzer: SentimentAnalyzer) -> None:
    """Test analyzing sentiment of a single statement."""
    statement = "This is a test statement."
    result = await sentiment_analyzer.analyze_sentiment(statement)

    assert len(result) == 1
    assert isinstance(result[0], SentimentAnalysis)
    assert result[0].sentiment in ["positive", "negative"]


@pytest.mark.asyncio()
async def test_analyze_sentiment_bad_request(sentiment_analyzer: SentimentAnalyzer) -> None:
    """Test handling a 400 Bad Request error with real API."""
    with pytest.raises(BadRequestError):
        await sentiment_analyzer.analyze_sentiment([])


@pytest.mark.asyncio()
async def test_analyze_sentiment_not_found(sentiment_analyzer: SentimentAnalyzer) -> None:
    """Test handling a 404 Not Found error with real API."""
    sentiment_analyzer._MODEL_API_URL = (
        "https://api-inference.huggingface.co/models/nonexistent-model"
    )
    with pytest.raises(NotFoundError):
        await sentiment_analyzer.analyze_sentiment("Test statement")


@pytest.mark.asyncio()
async def test_easy_to_discern_statements(sentiment_analyzer: SentimentAnalyzer) -> None:
    """Test analyzing sentiment of easy-to-discern statements.

    This test is not meant to fail but serve as a warning if the model does not return the
    expected results for very easy to classify statements.
    """
    positive_statement = "I absolutely love this wonderful day!"
    negative_statement = "I absolutely hate this terrible day!"
    statements = [positive_statement, negative_statement]
    result = await sentiment_analyzer.analyze_sentiment(statements)

    assert len(result) == len(statements)
    assert isinstance(result[0], SentimentAnalysis)
    assert isinstance(result[1], SentimentAnalysis)

    if result[0].sentiment != "positive" or result[1].sentiment != "negative":
        logger.warning(
            "Sentiment analysis did not return expected results. "
            "There might be something wrong with the model"
        )
