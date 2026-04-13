"""Deploy workflow agents from YAML definitions to Azure AI Foundry."""

from pathlib import Path

from azure.ai.projects.models import WorkflowAgentDefinition

from deploy_helpers import get_client


def deploy() -> None:
    client = get_client()

    workflows_dir = Path(__file__).parent / "workflows"
    for wf_file in sorted(workflows_dir.glob("*.yaml")):
        with open(wf_file) as f:
            wf_definition = f.read()
        workflow = client.agents.create_version(
            agent_name=wf_file.stem,
            definition=WorkflowAgentDefinition(
                workflow=wf_definition,
            ),
            headers={"Foundry-Features": "WorkflowAgents=V1Preview"},
        )
        print(f"Workflow '{wf_file.stem}' created: {workflow.id}")


if __name__ == "__main__":
    deploy()
