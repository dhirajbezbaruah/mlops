"""Short E2E tests for the API."""

import os

import pytest
from fastapi.testclient import TestClient

from feddit_analyzer.api import app
from feddit_analyzer.feddit_client import FedditAPIClient


@pytest.mark.asyncio()
async def test_e2e_sentiment_analysis() -> None:
    """Test the sentiment analysis endpoints."""
    client = TestClient(app)

    feddit_client = FedditAPIClient(os.environ["FEDDIT_API_BASE_URL"])

    subfeddits = await feddit_client.get_subfeddits(0, 1)

    assert len(subfeddits) == 1, "This test requires at least one subfeddit."

    subfeddit = subfeddits[0]

    response = client.post(
        "/api/v1/classify_comments/subfeddit_id",
        json={
            "subfeddit_id": subfeddit.id,
            "min_datetime": None,
            "max_datetime": None,
            "sort_by_polarity": False,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert "subfeddit_id" in response_data
    assert "comments" in response_data
    assert len(response_data["comments"]) <= 25

    response = client.post(
        "/api/v1/classify_comments/subfeddit_title",
        json={"subfeddit_title": subfeddit.title, "sort_by_polarity": False},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert "subfeddit_title" in response_data
    assert "comments" in response_data
    assert len(response_data["comments"]) <= 25

    response = client.post(
        "/api/v1/classify_comments/subfeddit_title",
        json={"subfeddit_title": subfeddit.title, "sort_by_polarity": True},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert "subfeddit_title" in response_data
    assert "comments" in response_data
    assert len(response_data["comments"]) <= 25
    sorted_comments = response.json()["comments"]
    polarities = [comment["polarity"] for comment in sorted_comments]
    assert polarities == sorted(polarities, reverse=True)
    response = client.post(
        "/api/v1/classify_comments/subfeddit_id",
        json={
            "subfeddit_id": subfeddit.id,
            "min_datetime": 1214123556,
            "max_datetime": 1814123556,
            "sort_by_polarity": True,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert "subfeddit_id" in response_data
    assert "comments" in response_data
    assert len(response_data["comments"]) <= 25
    sorted_comments = response.json()["comments"]
    polarities = [comment["polarity"] for comment in sorted_comments]
    assert polarities == sorted(polarities, reverse=True)
