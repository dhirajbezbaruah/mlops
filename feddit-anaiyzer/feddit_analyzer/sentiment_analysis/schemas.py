"""Module containing the Pydantic models for the sentiment analysis model API and analyzer
outputs."""

from pydantic import BaseModel, Field, ValidationError, field_validator

from .sentiment import Sentiment

_NLABELS = 3


class SentimentScore(BaseModel):
    """Model representing the response schema for sentiment score."""

    label: str = Field(..., title="Label", description="Sentiment label.")
    score: float = Field(..., title="Score", description="Sentiment score.")

    @field_validator("score")
    @classmethod
    def _score_must_be_between_0_and_1(cls, value: float) -> float:
        """Validate that the score is between 0 and 1.

        :param value: The score value to validate.
        :raises ValueError: If the score is not between 0 and 1.
        :return: The score value if it is valid.
        """
        if not (0 <= value <= 1):
            raise ValueError("Score must be between 0 and 1")
        return value


class ModelResponse(BaseModel):
    """Model representing the response schema for model response."""

    outputs: list[list[SentimentScore]] = Field(..., title="Outputs", description="Model outputs.")

    @field_validator("outputs")
    @classmethod
    def _validate_required_labels(
        cls, value: list[list[SentimentScore]]
    ) -> list[list[SentimentScore]]:
        """Validate that the outputs contains the three labels: positive, negative, and neutral.

        :param value: The outputs to validate.
        :raises ValueError: If the outputs do not contain the three labels.
        :return: The outputs if they are valid.
        """
        for scores in value:
            if len(scores) != _NLABELS or {score.label for score in scores} != {
                "positive",
                "negative",
                "neutral",
            }:
                raise ValidationError(
                    "Outputs must contain the three labels: positive, negative, and neutral"
                )

        return value


class SentimentAnalysis(BaseModel):
    """Model representing the request schema for sentiment analysis."""

    statement: str = Field(..., title="Statement", description="The statement to analyze.")
    polarity: float = Field(..., title="Polarity", description="The polarity of the query.")
    sentiment: Sentiment = Field(..., title="Sentiment", description="The sentiment of the query.")

    @field_validator("polarity")
    @classmethod
    def _polarity_must_be_between_minus_1_and_1(cls, value: float) -> float:
        """Validate that the polarity is between -1 and 1.

        :param value: The polarity value to validate.
        :raises ValueError: If the polarity is not between -1 and 1.
        :return: The polarity value if it is valid.
        """
        if not (-1 <= value <= 1):
            raise ValueError("Polarity must be between -1 and 1")

        return value
