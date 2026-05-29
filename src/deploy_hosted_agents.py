"""Build container images and deploy hosted agents to Azure AI Foundry."""

from azure.ai.projects.models import (
    HostedAgentDefinition,
    ProtocolVersionRecord,
    AgentProtocol,
)

from agents import discover_hosted_agents
from deploy_helpers import (
    assign_azure_ai_user_role,
    build_image,
    get_client,
    get_env,
)
from deploy_toolbox import TOOLBOX_NAME


def deploy() -> None:
    client = get_client()

    project_endpoint = get_env("AZURE_AI_PROJECT_ENDPOINT")
    model_deployment_name = get_env("AZURE_AI_MODEL_DEPLOYMENT_NAME", default="o4-mini")
    aoai_endpoint = get_env("AZURE_OPENAI_ENDPOINT")
    openai_api_version = get_env("OPENAI_API_VERSION", default="2024-05-01-preview")
    registry = get_env("AZURE_CONTAINER_REGISTRY_ENDPOINT")
    project_arm_id = get_env("AZURE_AI_PROJECT_ID", required=False, default="") or ""

    protocols = [ProtocolVersionRecord(protocol=AgentProtocol.RESPONSES, version="1.0.0")]

    toolbox_endpoint = f"{project_endpoint}/toolboxes/{TOOLBOX_NAME}/mcp?api-version=v1"

    # Inside containers, use the new SDK env var names. We also keep the legacy
    # AZURE_AI_* names for any code paths still reading them.
    # NOTE: FOUNDRY_* and AGENT_* are reserved env var prefixes injected by the
    # hosted agent platform — do not set them here.
    hosted_env = {
        "MODEL_DEPLOYMENT_NAME": model_deployment_name,
        "AZURE_AI_PROJECT_ENDPOINT": project_endpoint,
        "AZURE_AI_MODEL_DEPLOYMENT_NAME": model_deployment_name,
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": model_deployment_name,
        "AZURE_OPENAI_ENDPOINT": aoai_endpoint,
        "OPENAI_API_VERSION": openai_api_version,
        # Avoid the platform-reserved FOUNDRY_ prefix for our own values.
        "TOOLBOX_NAME": TOOLBOX_NAME,
        "TOOLBOX_MCP_ENDPOINT": toolbox_endpoint,
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
            ),
            metadata={"enableVnextExperience": "true"},
            headers={"Foundry-Features": "HostedAgents=V1Preview"},
        )
        print(f"Hosted agent '{config.name}' created: {agent.id}")

        # Grant the agent's dedicated Entra identity Azure AI User at project scope
        # so it can call models and reach the toolbox MCP endpoint.
        principal_id = _extract_principal_id(agent)
        if principal_id and project_arm_id:
            assign_azure_ai_user_role(principal_id, project_arm_id)
        elif not principal_id:
            print(f"  WARNING: could not find agent identity principal for '{config.name}'.")
        elif not project_arm_id:
            print("  WARNING: AZURE_AI_PROJECT_ID not set — skipping RBAC assignment.")


def _extract_principal_id(agent_version) -> str | None:
    """Best-effort extraction of the agent's Entra identity principal ID."""
    for attr in ("blueprint", "instance_identity", "identity", "agent_identity", "system_assigned_identity"):
        identity = getattr(agent_version, attr, None)
        if identity is None:
            continue
        for sub in ("principal_id", "principalId", "object_id", "objectId"):
            value = getattr(identity, sub, None)
            if not value and isinstance(identity, dict):
                value = identity.get(sub)
            if value:
                return value
    as_dict = getattr(agent_version, "as_dict", None)
    if callable(as_dict):
        data = as_dict()
        for key in ("blueprint", "instance_identity", "identity", "agentIdentity"):
            identity = data.get(key)
            if identity:
                value = identity.get("principal_id") or identity.get("principalId")
                if value:
                    return value
    return None


if __name__ == "__main__":
    deploy()

