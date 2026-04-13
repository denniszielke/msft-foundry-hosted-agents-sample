"""Shared helpers for agent deployment scripts."""

import os
import subprocess
from datetime import datetime
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv(override=True)


def get_env(name: str, required: bool = True, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_client() -> AIProjectClient:
    return AIProjectClient(
        endpoint=get_env("AZURE_AI_PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(),
    )


def build_image(registry: str, agent_name: str, context_path: Path) -> str:
    """Build a container image on ACR and return the full image tag."""
    registry_name = registry.split(".")[0]
    build_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    image_tag = f"{registry}/{agent_name}:{build_tag}"

    print(f"Queuing ACR build for {image_tag} from {context_path}...")
    subprocess.run(
        ["az", "acr", "build", "--registry", registry_name, "--image", image_tag, str(context_path)],
        check=True,
    )
    return image_tag
