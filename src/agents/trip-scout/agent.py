# Copyright (c) Microsoft. All rights reserved.
"""Trip Scout Agent.

Searches for flights, hotels, and activities based on user travel queries.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
    TextContent,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.ai.agentserver.agentframework import from_agent_framework


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """\
You are a travel search assistant for TripMate AI.
Search for flights, hotels, and activities based on the user's travel request.

Output ONLY valid JSON, no markdown, no extra text:
{
  "destination": "<city or region>",
  "flights": [
    {"airline": "<name>", "departure": "<time>", "arrival": "<time>", "price": "<X.XX€>", "class": "economy|business"}
  ],
  "hotels": [
    {"name": "<hotel name>", "rating": <1-5>, "price_per_night": "<X.XX€>", "location": "<area/neighbourhood>"}
  ],
  "activities": [
    {"name": "<activity>", "price": "<X.XX€>", "duration": "<e.g. 2 hours>"}
  ],
  "estimated_budget": "<total X.XX€>"
}

Always include at least 2 flight options, 2 hotel options, and 2 activity suggestions.
Use realistic airline names, hotel brands, and local attractions for the destination.
Prices should be in EUR.
"""


# ---------------------------------------------------------------------------
# Structured Output Models
# ---------------------------------------------------------------------------


class FlightOption(BaseModel):
    airline: str = Field(description="Airline name")
    departure: str = Field(description="Departure time")
    arrival: str = Field(description="Arrival time")
    price: str = Field(description="Price in EUR")
    flight_class: str = Field(alias="class", default="economy")


class HotelOption(BaseModel):
    name: str = Field(description="Hotel name")
    rating: int = Field(description="Star rating 1-5")
    price_per_night: str = Field(description="Price per night in EUR")
    location: str = Field(description="Area or neighbourhood")


class Activity(BaseModel):
    name: str = Field(description="Activity name")
    price: str = Field(description="Price in EUR")
    duration: str = Field(description="Estimated duration")


class TripSearchResult(BaseModel):
    destination: str
    flights: list[FlightOption] = []
    hotels: list[HotelOption] = []
    activities: list[Activity] = []
    estimated_budget: str = ""


class TripScoutOutput(BaseModel):
    human_readable: str = Field(description="User-friendly summary of the search results.")
    result: TripSearchResult = Field(description="Structured travel search results.")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TripScoutAgent(BaseAgent):
    """Searches for flights, hotels, and activities for travel planning."""

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name or "trip-scout",
            description=description or "Searches flights, hotels, and activities for travel planning.",
            **kwargs,
        )
        self._chat_client = AzureOpenAIChatClient(
            credential=DefaultAzureCredential(),
            api_version="2024-05-01-preview",
        )

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        normalized = self._normalize_messages(messages)
        user_text = normalized[-1].text if normalized else "a weekend getaway"

        llm_messages = [
            ChatMessage(role=Role.SYSTEM, text=_SYSTEM_PROMPT),
            ChatMessage(role=Role.USER, text=user_text),
        ]
        response = await self._chat_client.get_response(messages=llm_messages)

        if hasattr(response, "messages") and response.messages:
            raw = response.messages[-1].text or ""
        elif hasattr(response, "message"):
            raw = response.message.text or ""
        else:
            raw = str(response)

        parsed = json.loads(raw.strip())
        result = TripSearchResult(
            destination=parsed.get("destination", "Unknown"),
            flights=[FlightOption(**f) for f in parsed.get("flights", [])],
            hotels=[HotelOption(**h) for h in parsed.get("hotels", [])],
            activities=[Activity(**a) for a in parsed.get("activities", [])],
            estimated_budget=parsed.get("estimated_budget", "N/A"),
        )

        # Build human-readable summary
        lines = [f"🗺️ **Travel options for {result.destination}**\n"]
        lines.append("✈️ **Flights:**")
        for f in result.flights:
            lines.append(f"  • {f.airline} — {f.departure}→{f.arrival} — {f.price}")
        lines.append("\n🏨 **Hotels:**")
        for h in result.hotels:
            lines.append(f"  • {h.name} ({'⭐' * h.rating}) — {h.price_per_night}/night — {h.location}")
        lines.append("\n🎯 **Activities:**")
        for a in result.activities:
            lines.append(f"  • {a.name} — {a.price} ({a.duration})")
        lines.append(f"\n💰 **Estimated budget:** {result.estimated_budget}")

        output = TripScoutOutput(
            human_readable="\n".join(lines),
            result=result,
        )

        response_message = ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text=output.model_dump_json())],
        )

        if thread is not None:
            await self._notify_thread_of_new_messages(thread, normalized, response_message)

        return AgentRunResponse(messages=[response_message])

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        full = await self.run(messages=messages, thread=thread, **kwargs)
        if full.messages:
            msg = full.messages[0]
            yield AgentRunResponseUpdate(contents=msg.contents, role=msg.role)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    load_dotenv(dotenv_path=workspace_root / ".env", override=True)

    agent = TripScoutAgent(
        name="trip-scout",
        description="Searches flights, hotels, and activities for travel planning.",
    )
    from_agent_framework(agent).run()
