"""Trip Scout Agent.

Travel-search agent that uses Foundry Toolbox MCP (Bing Custom Web Search) to
ground real travel options (flights, hotels, activities) for user requests.
"""

from __future__ import annotations

import os

import httpx
from agent_framework import MCPStreamableHTTPTool
from agent_framework_foundry import FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()


_SYSTEM_PROMPT = """\
You are a travel search assistant for TripMate AI.
Use the toolbox web search tool to find real, current flight, hotel, and activity
options for the user's travel request. Always:

- Suggest at least 2 flight options, 2 hotel options, and 2 activities.
- Quote prices in EUR with realistic ranges.
- Include source links / citations from your tool results.
- Finish with an estimated total budget.

Be concise and friendly.
"""

_TOOLBOX_NAME = os.environ.get("TOOLBOX_NAME", "tripmate-tools")
_PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
_MODEL_DEPLOYMENT = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")

_TOOLBOX_ENDPOINT = os.environ.get("TOOLBOX_MCP_ENDPOINT") or (
    f"{_PROJECT_ENDPOINT.rstrip('/')}/toolboxes/{_TOOLBOX_NAME}/mcp?api-version=v1"
)

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")


class _ToolboxAuth(httpx.Auth):
    """Inject a fresh Entra token on every toolbox MCP request."""

    def __init__(self, token_provider):
        self._get_token = token_provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._get_token()}"
        yield request


_http_client = httpx.AsyncClient(
    auth=_ToolboxAuth(_token_provider),
    headers={"Foundry-Features": "Toolboxes=V1Preview"},
    timeout=120.0,
)

_mcp_tool = MCPStreamableHTTPTool(
    name="toolbox",
    url=_TOOLBOX_ENDPOINT,
    http_client=_http_client,
    load_prompts=False,
)

_chat_client = FoundryChatClient(
    project_endpoint=_PROJECT_ENDPOINT,
    model=_MODEL_DEPLOYMENT,
    credential=_credential,
)

agent = _chat_client.as_agent(
    name="trip-scout",
    instructions=_SYSTEM_PROMPT,
    tools=[_mcp_tool],
)


if __name__ == "__main__":
    ResponsesHostServer(agent).run()
