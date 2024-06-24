"""Script to wait for the API to be up and running."""

import sys
import time

import click
import requests
from loguru import logger
from requests.exceptions import HTTPError, RequestException

from feddit_analyzer.feddit_client import FedditAPIClient
from feddit_analyzer.feddit_client.schemas import VersionResponse


@click.command("wait-api")
@click.argument("base-url", required=True, type=str)
@click.option("--timeout", default=10, help="Timeout in seconds.", type=float)
@click.option("--wait", default=10, help="Wait time in seconds.", type=float)
@click.option("--retries", default=3, help="Number of retries.", type=int)
def wait_api(base_url: str, timeout: float, wait: float, retries: int) -> None:
    """Wait for the API to be up and running.

    \f

    :param base_url: Base URL of the API.
    :param timeout: Timeout in seconds.
    :param wait: Wait time in seconds.
    :param retries: Number of retries.
    """
    attempts = 0
    logger.info("Checking if the API is up and running on {}", base_url)
    while attempts < retries:
        try:
            logger.info("Attempt {} of {}", attempts + 1, retries)
            response = requests.get(f"{base_url}/api/v1/version", timeout=timeout)

            response.raise_for_status()
            version = VersionResponse.model_validate(response.json()).version
            if version in FedditAPIClient.VALID_VERSIONS:
                logger.info("API is has a valid version: {}", version)
            else:
                logger.warning("API version {} is not supported.", version)

            logger.info("API is up and running at {}.", base_url)
            sys.exit(0)

        except (RequestException, HTTPError) as exc:
            logger.info("API is not up and running: {}", exc)
            time.sleep(wait)
            attempts += 1

    logger.error("API is not up and running after {} attempts.", retries)
    sys.exit(1)
