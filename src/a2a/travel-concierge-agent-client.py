"""A2A test client for the travel-concierge agent on Azure AI Foundry."""

import asyncio
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types.a2a_pb2 import Role, SendMessageRequest

# Load .env from workspace root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

AGENT_NAME = os.getenv("A2A_AGENT_NAME", "travel-concierge")
ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT", "").rstrip("/")
if not ENDPOINT:
    sys.exit("ERROR: AZURE_AI_PROJECT_ENDPOINT is not set.")

A2A_BASE_URL = f"{ENDPOINT}/agents/{AGENT_NAME}/endpoint/protocols/a2a"
AGENT_CARD_PATH = "agentCard/v0.3"

TEST_MESSAGES = [
    "Hello, what can you do?",
    "I want to find flights to Barcelona next week",
    "Can you cancel my hotel booking?",
]


async def main() -> None:
    credential = DefaultAzureCredential()
    token = credential.get_token("https://ai.azure.com/.default").token

    print(f"Agent:          {AGENT_NAME}")
    print(f"A2A base URL:   {A2A_BASE_URL}")
    print(f"Agent card:     {A2A_BASE_URL}/{AGENT_CARD_PATH}")
    print()

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(300.0),
    ) as httpx_client:
        # Resolve the agent card
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=A2A_BASE_URL,
            agent_card_path=AGENT_CARD_PATH,
        )
        agent_card = await resolver.get_agent_card()
        print(f"Agent card resolved: {agent_card.name}")
        print(f"  description: {agent_card.description}")
        print(f"  skills: {[s.name for s in agent_card.skills]}")
        print()

        # Create a non-streaming A2A client
        config = ClientConfig(
            streaming=False,
            httpx_client=httpx_client,
        )
        client = await create_client(agent=agent_card, client_config=config)

        # Send test messages
        for text in TEST_MESSAGES:
            print(f">>> USER: {text}")
            message = new_text_message(text, role=Role.ROLE_USER)
            request = SendMessageRequest(message=message)

            try:
                async for response in client.send_message(request):
                    if response.HasField("message"):
                        for part in response.message.parts:
                            if part.text:
                                print(f"<<< AGENT: {part.text}")
                    elif response.HasField("task"):
                        task = response.task
                        # Extract text from task artifacts
                        for artifact in task.artifacts:
                            for part in artifact.parts:
                                if part.text:
                                    print(f"<<< AGENT: {part.text}")
                        # Extract text from task history
                        for msg in task.history:
                            if msg.role == Role.ROLE_AGENT:
                                for part in msg.parts:
                                    if part.text:
                                        print(f"<<< AGENT: {part.text}")
                        # Show status if no content was found
                        if not task.artifacts and not task.history:
                            print(f"    [task] id={task.id} state={task.status.state}")
                    elif response.HasField("status_update"):
                        print(f"    [status] {response.status_update}")
            except Exception as e:
                print(f"    [error] {e}")
            print()

        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
