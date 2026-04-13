"""Build container images and deploy hosted agents to Azure AI Foundry."""

import os

from azure.ai.projects.models import (
    HostedAgentDefinition,
    ProtocolVersionRecord,
    AgentProtocol,
    BingCustomSearchPreviewTool,
    BingCustomSearchToolParameters,
    BingCustomSearchConfiguration,
)

from agents import discover_hosted_agents
from deploy_helpers import build_image, get_client, get_env


def deploy() -> None:
    client = get_client()

    project_endpoint = get_env("AZURE_AI_PROJECT_ENDPOINT")
    model_deployment_name = get_env("AZURE_AI_MODEL_DEPLOYMENT_NAME", default="o4-mini")
    aoai_endpoint = get_env("AZURE_OPENAI_ENDPOINT")
    openai_api_version = get_env("OPENAI_API_VERSION", default="2024-05-01-preview")
    registry = get_env("AZURE_CONTAINER_REGISTRY_ENDPOINT")

    protocols = [ProtocolVersionRecord(protocol=AgentProtocol.RESPONSES, version="v2")]

    # Bing Custom Search tool (optional, shared across hosted agents)
    tools = []
    bing_conn_name = os.environ.get("BING_CUSTOM_GROUNDING_CONNECTION_NAME", "")
    if bing_conn_name:
        bing_conn_id = client.connections.get(bing_conn_name).id
        tools.append(BingCustomSearchPreviewTool(
            bing_custom_search_preview=BingCustomSearchToolParameters(
                search_configurations=[BingCustomSearchConfiguration(
                    project_connection_id=bing_conn_id)]
            )
        ))

    hosted_env = {
        "AZURE_AI_PROJECT_ENDPOINT": project_endpoint,
        "AZURE_AI_MODEL_DEPLOYMENT_NAME": model_deployment_name,
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": model_deployment_name,
        "AZURE_OPENAI_ENDPOINT": aoai_endpoint,
        "OPENAI_API_VERSION": openai_api_version,
    }

    for config in discover_hosted_agents():
        if not (config.path / "Dockerfile").exists():
            print(f"Skipping '{config.name}': no Dockerfile found")
            continue

        image_tag = build_image(registry, config.name, config.path)
        env_vars = {**hosted_env, **config.env_vars}

        agent = client.agents.create_version(
            agent_name=config.name,
            description=config.description,
            definition=HostedAgentDefinition(
                container_protocol_versions=protocols,
                cpu=config.cpu,
                memory=config.memory,
                image=image_tag,
                environment_variables=env_vars,
                tools=tools if tools else None,
            ),
            headers={"Foundry-Features": "HostedAgents=V1Preview"},
        )
        print(f"Hosted agent '{config.name}' created: {agent.id}")


if __name__ == "__main__":
    deploy()
