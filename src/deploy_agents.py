import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    HostedAgentDefinition,
    WorkflowAgentDefinition,
    PromptAgentDefinition,
    PromptAgentDefinitionTextOptions,
    ProtocolVersionRecord,
    AgentProtocol,
    BingCustomSearchPreviewTool,
    BingCustomSearchToolParameters,
    BingCustomSearchConfiguration,
    FoundryFeaturesOptInKeys,
    TextResponseFormatJsonSchema,
)
from azure.identity import DefaultAzureCredential

from dotenv import load_dotenv

load_dotenv(override=True)

def get_env(name: str, required: bool = True, default: str | None = None) -> str:
  value = os.getenv(name, default)
  if required and not value:
      raise RuntimeError(f"Missing required environment variable: {name}")
  return value


# ---------------------------------------------------------------------------
# Travel Concierge – declarative prompt agent (no container)
# ---------------------------------------------------------------------------

CONCIERGE_SYSTEM_PROMPT = """\
You are the Travel Concierge for TripMate AI, a friendly travel planning assistant.
Classify the user's intent and respond with valid JSON only.

Respond ONLY with valid JSON (no markdown, no extra text):
{"next_agent": "<agent>", "reason": "<short reason>", "response": "<message for the user>"}

<agent> must be one of: trip-scout | booking-manager | none

Routing rules:
• trip-scout — searching destinations, flights, hotels, activities, comparing travel options, planning trips
• booking-manager — booking, modifying, cancelling reservations, checking booking status
• none — greetings, general travel tips, small talk, or questions you can answer directly

The response field:
• When routing to trip-scout or booking-manager, write a brief acknowledgement like
  "Let me search for flights to Barcelona for you!" or "I'll check on that booking right away."
• When next_agent is "none", write a full friendly answer to the user's question.
"""

CONCIERGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "next_agent": {
            "type": "string",
            "enum": ["trip-scout", "booking-manager", "none"],
            "description": "The agent to route to next, or none if done."
        },
        "reason": {
            "type": "string",
            "description": "Short reason for the routing decision."
        },
        "response": {
            "type": "string",
            "description": "Human-friendly message to show the user."
        }
    },
    "required": ["next_agent", "reason", "response"],
    "additionalProperties": False
}


def main() -> None:
  project_endpoint = get_env("AZURE_AI_PROJECT_ENDPOINT", required=True)
  model_deployment_name = get_env("AZURE_AI_MODEL_DEPLOYMENT_NAME", required=True, default="o4-mini")
  aoai_endpoint = get_env("AZURE_OPENAI_ENDPOINT", required=True)
  openai_api_version = get_env("OPENAI_API_VERSION", required=True, default="2024-05-01-preview")

  credential = DefaultAzureCredential()

  client = AIProjectClient(
      endpoint=project_endpoint,
      credential=credential,
  )

  # -----------------------------------------------------------------------
  # 1. Travel Concierge (declarative prompt agent – no container)
  # -----------------------------------------------------------------------
  concierge = client.agents.create_version(
      agent_name="travel-concierge",
      description="TripMate AI Travel Concierge — classifies intent and routes to specialist agents",
      definition=PromptAgentDefinition(
          model=model_deployment_name,
          instructions=CONCIERGE_SYSTEM_PROMPT,
          temperature=0.1,
          text=PromptAgentDefinitionTextOptions(
              format=TextResponseFormatJsonSchema(
                  name="concierge_routing",
                  schema=CONCIERGE_OUTPUT_SCHEMA,
                  strict=True,
              ),
          ),
      ),
  )
  print(f"Prompt agent 'travel-concierge' created: {concierge.id}")

  # -----------------------------------------------------------------------
  # 2. Hosted agents from container images (*_IMAGE env vars)
  # -----------------------------------------------------------------------
  protocols = [ProtocolVersionRecord(protocol=AgentProtocol.RESPONSES, version="v2")]

  for key, value in os.environ.items():
      if not key.endswith("_IMAGE"):
          continue

      image_tag = value
      if not image_tag:
          continue

      base_name = key[:-len("_IMAGE")]
      agent_name = base_name.lower().replace("_", "-")

      # Bing Custom Search tool (optional)
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

      agent = client.agents.create_version(
          agent_name=agent_name,
          description=f"Hosted agent for {agent_name}",
          foundry_features=FoundryFeaturesOptInKeys.HOSTED_AGENTS_V1_PREVIEW,
          definition=HostedAgentDefinition(
              container_protocol_versions=protocols,
              cpu="1",
              memory="2Gi",
              image=image_tag,
              environment_variables={
                  "AZURE_AI_PROJECT_ENDPOINT": project_endpoint,
                  "AZURE_AI_MODEL_DEPLOYMENT_NAME": model_deployment_name,
                  "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": model_deployment_name,
                  "AZURE_OPENAI_ENDPOINT": aoai_endpoint,
                  "OPENAI_API_VERSION": openai_api_version,
              },
              tools=tools if tools else None,
          ),
      )
      print(f"Hosted agent '{agent_name}' created: {agent.id}")

  # -----------------------------------------------------------------------
  # 3. Workflow agents from YAML files
  # -----------------------------------------------------------------------
  workflows_dir = os.path.join(os.path.dirname(__file__), "workflows")
  for wf_file in os.listdir(workflows_dir):
      if not wf_file.endswith(".yaml"):
          continue
      wf_path = os.path.join(workflows_dir, wf_file)
      with open(wf_path, "r") as f:
          wf_definition = f.read()
      wf_name = wf_file[:-len(".yaml")]
      workflow = client.agents.create_version(
          agent_name=wf_name,
          foundry_features=FoundryFeaturesOptInKeys.WORKFLOW_AGENTS_V1_PREVIEW,
          definition=WorkflowAgentDefinition(
              workflow=wf_definition,
          ),
      )
      print(f"Workflow '{wf_name}' created: {workflow.id}")

if __name__ == "__main__":
  main()