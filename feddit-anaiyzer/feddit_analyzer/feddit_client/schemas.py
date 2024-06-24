"""Validation schemas for Feddit client responses."""

from pydantic import BaseModel, Field


class CommentInfo(BaseModel):
    """Model representing basic fields of a comment."""

    id: int = Field(..., title="Id", description="ID of the comment")
    username: str = Field(..., title="Username", description="User who commented on the subfeddit.")
    text: str = Field(..., title="Text", description="Content of the comment.")
    created_at: int = Field(
        ..., title="Created At", description="Created time of the comment in Unix epochs."
    )


class CommentsResponse(BaseModel):
    """Model representing the response schema for comments."""

    subfeddit_id: int = Field(
        ..., title="Subfeddit Id", description="ID of the subfeddit, to which the comments belong."
    )
    limit: int = Field(10, title="Limit", description="Max number of returning comments.")
    skip: int = Field(0, title="Skip", description="Number of comments to skip.")
    comments: list[CommentInfo] = Field(
        ..., title="Comments", description="Comments in this subfeddit."
    )


class SubfedditInfo(BaseModel):
    """Model representing basic fields of a subfeddit."""

    id: int = Field(..., title="Id", description="ID of the subfeddit")
    username: str = Field(..., title="Username", description="User who created the subfeddit.")
    title: str = Field(..., title="Title", description="Title of the subfeddit")
    description: str = Field(..., title="Description", description="Description of the subfeddit")


class SubfedditsResponse(BaseModel):
    """Model representing the response schema for a list of subfeddits."""

    limit: int = Field(10, title="Limit", description="Max number of returning subfeddits.")
    skip: int = Field(0, title="Skip", description="Number of subfeddits to skip.")
    subfeddits: list[SubfedditInfo] = Field(
        ..., title="Subfeddits", description="List of subfeddits with brief information."
    )


class SubfedditResponse(BaseModel):
    """Model representing the response schema for a single subfeddit."""

    id: int = Field(..., title="Id", description="ID of the subfeddit")
    username: str = Field(..., title="Username", description="User who created the subfeddit.")
    title: str = Field(..., title="Title", description="Title of the subfeddit")
    description: str = Field(..., title="Description", description="Description of the subfeddit")
    limit: int = Field(10, title="Limit", description="Max number of returning comments.")
    skip: int = Field(0, title="Skip", description="Number of comments to skip.")
    comments: list[CommentInfo] = Field(
        ..., title="Comments", description="Comments in this subfeddit."
    )


class VersionResponse(BaseModel):
    """Model representing the response schema for the version endpoint."""

    version: str = Field(..., title="Version", description="Version of the API", examples=["0.1.0"])
