"""Create or update the Foundry Toolbox that hosted agents consume at runtime."""

import os

from azure.ai.projects.models import WebSearchTool

from deploy_helpers import get_client, get_env

TOOLBOX_NAME = "tripmate-tools"


def deploy() -> None:
    client = get_client()

    bing_conn_name = os.environ.get("BING_CUSTOM_GROUNDING_CONNECTION_NAME", "")
    if not bing_conn_name:
        print("BING_CUSTOM_GROUNDING_CONNECTION_NAME not set — skipping toolbox deploy.")
        return

    instance_name = os.environ.get("BING_CUSTOM_GROUNDING_CONFIG_INSTANCE_NAME", "default")

    bing_conn_id = client.connections.get(bing_conn_name).id

    # Web Search backed by the project's Bing Custom Grounding connection.
    # The Python model classes for `custom_search_configuration` aren't fully
    # documented; we try the typed kwarg first and fall back to a raw dict.
    try:
        web_search = WebSearchTool(
            custom_search_configuration={
                "project_connection_id": bing_conn_id,
                "instance_name": instance_name,
            },
        )
    except TypeError:
        web_search = {  # type: ignore[assignment]
            "type": "web_search",
            "web_search": {
                "custom_search_configuration": {
                    "project_connection_id": bing_conn_id,
                    "instance_name": instance_name,
                },
            },
        }

    project_endpoint = get_env("AZURE_AI_PROJECT_ENDPOINT")

    version = client.beta.toolboxes.create_version(
        name=TOOLBOX_NAME,
        description="TripMate AI shared toolbox — Bing Custom Web Search",
        tools=[web_search],
    )

    consumer_endpoint = f"{project_endpoint}/toolboxes/{TOOLBOX_NAME}/mcp?api-version=v1"
    print(f"Toolbox '{TOOLBOX_NAME}' version '{version.version}' created.")
    print(f"  Consumer endpoint: {consumer_endpoint}")


if __name__ == "__main__":
    deploy()

