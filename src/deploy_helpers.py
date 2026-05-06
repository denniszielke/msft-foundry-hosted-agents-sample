"""Shared helpers for agent deployment scripts."""

import os
import subprocess
from datetime import datetime
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv(override=True)

# Azure AI User role definition ID — required at project scope for hosted agent
# identities to call models and reach the toolbox MCP endpoint.
_AZURE_AI_USER_ROLE_ID = "53ca6127-db72-4b80-b1b0-d745d6d5456d"


def get_env(name: str, required: bool = True, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_client() -> AIProjectClient:
    return AIProjectClient(
        endpoint=get_env("AZURE_AI_PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )


def build_image(registry: str, agent_name: str, context_path: Path) -> str:
    """Build a container image on ACR and return the full image tag."""
    registry_name = registry.split(".")[0]
    build_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    image_tag = f"{registry}/{agent_name}:{build_tag}"

    print(f"Queuing ACR build for {image_tag} from {context_path}...")
    subprocess.run(
        [
            "az", "acr", "build",
            "--registry", registry_name,
            "--image", image_tag,
            "--platform", "linux/amd64",
            str(context_path),
        ],
        check=True,
    )
    return image_tag


def assign_azure_ai_user_role(principal_id: str, scope: str) -> None:
    """Assign the Azure AI User role to a principal at the given scope.

    Idempotent: ignores 'already exists' errors from az CLI.
    """
    print(f"Assigning Azure AI User role to principal {principal_id} at {scope}...")
    result = subprocess.run(
        [
            "az", "role", "assignment", "create",
            "--assignee-object-id", principal_id,
            "--assignee-principal-type", "ServicePrincipal",
            "--role", _AZURE_AI_USER_ROLE_ID,
            "--scope", scope,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr or ""
        if "RoleAssignmentExists" in stderr or "already exists" in stderr.lower():
            print("  Role assignment already exists — skipping.")
            return
        raise RuntimeError(f"Role assignment failed: {stderr}")

