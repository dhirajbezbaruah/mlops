"""Schemas for API input and output validation."""

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

_MAX_COMMENTS = 25


class VersionResponse(BaseModel):
    """Model representing the response schema for the version endpoint."""

    version: str = Field(
        ...,
        title="Version",
        description="API version of deployed app.",
        examples=["0.1.0"],
    )


class CommentSentimentIDRequest(BaseModel):
    """Model representing the request schema for sentiment analysis of comments from subfeddit
    ID."""

    subfeddit_id: int = Field(
        ..., title="Subfeddit ID", description="ID of the subfeddit.", examples=[1, 2, 3]
    )
    min_datetime: int | None = Field(
        None,
        title="Min Datetime",
        description="Minimum datetime for comments. It has to be in Unix epochs.",
        examples=[0, 1213423454, 12345655],
    )
    max_datetime: int | None = Field(
        None,
        title="Max Datetime",
        description="Maximum datetime for comments. It has to be in Unix epochs.",
        examples=[0, 1213423454, 12345655],
    )
    sort_by_polarity: bool = Field(
        False, title="Sort By Polarity", description="Sort comments by polarity."
    )

    @model_validator(mode="after")
    def validate_min_max_datetime(self) -> Self:
        """Validate that min_datetime is less than max_datetime.

        :raises ValueError: If min_datetime is greater than max_datetime.
        :return: The request if it is valid.
        """
        if (
            self.min_datetime is not None
            and self.max_datetime is not None
            and self.min_datetime > self.max_datetime
        ):
            raise ValueError("min_datetime cannot be greater than max_datetime.")

        return self


class CommentSentimentRequest(BaseModel):
    """Model representing the request schema for sentiment analysis of comments from subfeddit
    title."""

    subfeddit_title: str = Field(
        ...,
        title="Subfeddit Title",
        description="Title of the subfeddit.",
        examples=["title 1", "title 2"],
    )
    min_datetime: int | None = Field(
        None,
        title="Min Datetime",
        description="Minimum datetime for comments. It has to be in Unix epochs.",
        examples=[0, 1213423454, 12345655],
    )
    max_datetime: int | None = Field(
        None,
        title="Max Datetime",
        description="Maximum datetime for comments. It has to be in Unix epochs.",
        examples=[0, 1213423454, 12345655],
    )
    sort_by_polarity: bool = Field(
        False, title="Sort By Polarity", description="Sort comments by polarity."
    )

    @model_validator(mode="after")
    def validate_min_max_datetime(self) -> Self:
        """Validate that min_datetime is less than max_datetime.

        :raises ValueError: If min_datetime is greater than max_datetime.
        :return: The request if it is valid.
        """
        if (
            self.min_datetime is not None
            and self.max_datetime is not None
            and self.min_datetime > self.max_datetime
        ):
            raise ValueError("min_datetime cannot be greater than max_datetime.")

        return self


class CommentSentiment(BaseModel):
    """Model representing the response schema for sentiment analysis of comments."""

    comment_id: int = Field(
        ..., title="Comment ID", description="ID of the comment.", examples=[1, 2, 3]
    )
    comment: str = Field(
        ...,
        title="Comment",
        description="Content of the comment.",
        examples=["I like this.", "I dislike this."],
    )
    polarity: float = (
        Field(..., title="Polarity", description="Polarity of the comment.", examples=[1, 0, 0.5]),
    )
    classification: Literal["positive", "negative"] = Field(
        ..., title="Classification", description="Classification of the comment."
    )


class CommentSentimentIDResponse(BaseModel):
    """Model representing the response schema for sentiment analysis of comments by providing
    subfeddit ID."""

    subfeddit_id: int = Field(
        ..., title="Subfeddit ID", description="ID of the subfeddit.", examples=[1, 2, 3]
    )
    comments: list[CommentSentiment] = Field(
        ..., title="Comments", description="List of comments with sentiment analysis."
    )

    @field_validator("comments")
    @classmethod
    def no_more_than_25_comments(cls, value: list[CommentSentiment]) -> list[CommentSentiment]:
        """Validate that the number of comments is no more than 25.

        :param value: The comments to validate.
        :raises ValueError: If the number of comments is more than 25.
        :return: The comments if they are valid.
        """
        if len(value) > _MAX_COMMENTS:
            raise ValueError("Number of comments cannot be more than 25.")

        return value


class CommentSentimentResponse(BaseModel):
    """Model representing the response schema for sentiment analysis of comments."""

    subfeddit_title: str = Field(
        ...,
        title="Subfeddit Title",
        description="Title of the subfeddit.",
        examples=["title 1", "title 2"],
    )
    comments: list[CommentSentiment] = Field(
        ..., title="Comments", description="List of comments with sentiment analysis."
    )

    @field_validator("comments")
    @classmethod
    def no_more_than_25_comments(cls, value: list[CommentSentiment]) -> list[CommentSentiment]:
        """Validate that the number of comments is no more than 25.

        :param value: The comments to validate.
        :raises ValueError: If the number of comments is more than 25.
        :return: The comments if they are valid.
        """
        if len(value) > _MAX_COMMENTS:
            raise ValueError("Number of comments cannot be more than 25.")

        return value
