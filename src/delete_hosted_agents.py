"""Delete hosted agents from Azure AI Foundry with force=true."""

import sys

from agents import discover_hosted_agents
from deploy_helpers import get_client


def delete(agent_names: list[str] | None = None) -> None:
    client = get_client()

    if agent_names:
        names = agent_names
    else:
        # Default: discover all hosted agents in this repo
        names = [cfg.name for cfg in discover_hosted_agents()]

    if not names:
        print("No agents found to delete.")
        return

    for name in names:
        try:
            result = client.agents.delete(name, params={"force": "true"})
            print(f"Deleted agent '{name}': {result}")
        except Exception as e:
            print(f"Failed to delete agent '{name}': {e}")


if __name__ == "__main__":
    # Pass agent names as CLI args, or delete all discovered hosted agents
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    delete(args)
