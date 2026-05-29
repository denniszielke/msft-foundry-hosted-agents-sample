"""Deploy prompt-based agents to Azure AI Foundry."""

from azure.ai.projects.models import (
    PromptAgentDefinition,
    PromptAgentDefinitionTextOptions,
    TextResponseFormatJsonSchema,
    AgentEndpoint,
    AgentEndpointProtocol,
    AgentCard,
    AgentCardSkill,
)

from deploy_helpers import get_client, get_env


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


def deploy() -> None:
    client = get_client()
    model = get_env("AZURE_AI_MODEL_DEPLOYMENT_NAME", default="o4-mini")

    concierge = client.agents.create_version(
        agent_name="travel-concierge",
        description="TripMate AI Travel Concierge — classifies intent and routes to specialist agents",
        definition=PromptAgentDefinition(
            model=model,
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

    endpoint_config = AgentEndpoint(
        protocols=[
            AgentEndpointProtocol.RESPONSES,
            AgentEndpointProtocol.A2A,
        ],
    )

    agent_card = AgentCard(
        description="TripMate AI Travel Concierge — classifies intent and routes to specialist agents",
        version="1.0",
        skills=[
            AgentCardSkill(
                id="intent-routing",
                name="Intent Routing",
                description="Classifies user intent and routes to trip-scout or booking-manager",
            ),
        ],
    )

    patched_agent = client.beta.agents.patch_agent_details(
        agent_name="travel-concierge",
        agent_endpoint=endpoint_config,
        agent_card=agent_card,
    )

    endpoint = get_env("AZURE_AI_PROJECT_ENDPOINT").rstrip("/")
    agent_name = "travel-concierge"
    a2a_base = f"{endpoint}/agents/{agent_name}/endpoint/protocols/a2a"
    card_url = f"{a2a_base}/agentCard/v0.3"

    print(f"\nPrompt agent '{agent_name}' created: {concierge.id}")
    print(f"A2A base path: {a2a_base}")
    print(f"Agent card URL: {card_url}")


if __name__ == "__main__":
    deploy()
