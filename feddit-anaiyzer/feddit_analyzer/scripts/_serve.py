"""Script to serve the API."""

import click
import uvicorn


@click.command("serve")
@click.option("--host", default="127.0.0.1", help="Host to serve the API on.", type=str)
@click.option("--port", default=8000, help="Port to serve the API on.", type=int)
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
@click.option(
    "--log-level",
    default="info",
    help="Log level for the server.",
    type=click.Choice(
        ["critical", "error", "warning", "info", "debug", "trace"], case_sensitive=False
    ),
)
def serve(host: str, port: int, reload: bool, log_level: str) -> None:
    """Serve the API.

    Requires the definition of the following environment variables: FEDDIT_API_BASE_URL and
    HUGGINGFACE_API_KEY.

    \f

    :param host: Host to serve the API on.
    :param port: Port to serve the API on.
    :param reload: Enable auto-reload for development.
    :param log_level: Log level for the server.
    """
    uvicorn.run(
        "feddit_analyzer.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
