import os
import logging

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
from typing_extensions import Literal
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from azure.ai.agentserver.langgraph import from_langgraph
from azure.monitor.opentelemetry import configure_azure_monitor

logger = logging.getLogger(__name__)

load_dotenv()

if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor(enable_live_metrics=True, logger_name="__main__")

deployment_name = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

try:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    llm = init_chat_model(
        f"azure_openai:{deployment_name}",
        azure_ad_token_provider=token_provider,
    )
except Exception:
    logger.exception("Booking Manager Agent failed to start")
    raise


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def check_availability(type: str, name: str, date: str) -> dict:
    """Check availability for a flight or hotel.

    Args:
        type: Type of booking - 'flight' or 'hotel'
        name: Name of the airline/flight or hotel
        date: Date to check (e.g. '2026-06-15')
    """
    import random

    available = random.choice([True, True, True, False])  # 75% chance available
    remaining = random.randint(1, 20) if available else 0

    if type == "flight":
        price = round(random.uniform(89.0, 450.0), 2)
    else:
        price = round(random.uniform(60.0, 350.0), 2)

    return {
        "type": type,
        "name": name,
        "date": date,
        "available": available,
        "remaining_spots": remaining,
        "price_eur": price,
    }


@tool
def create_booking(flight: str, hotel: str, check_in: str, check_out: str, guests: int = 1) -> dict:
    """Create a new travel booking.

    Args:
        flight: Flight description (e.g. 'Lufthansa LH1234 morning')
        hotel: Hotel name
        check_in: Check-in date
        check_out: Check-out date
        guests: Number of guests
    """
    import uuid
    import random

    booking_id = "TM-" + str(uuid.uuid4())[:8].upper()
    flight_price = round(random.uniform(89.0, 450.0), 2)
    nights = max(1, (int(check_out.split("-")[2]) - int(check_in.split("-")[2])))
    hotel_price_per_night = round(random.uniform(80.0, 300.0), 2)
    hotel_total = round(hotel_price_per_night * nights, 2)
    total = round(flight_price + hotel_total, 2)

    return {
        "booking_id": booking_id,
        "status": "confirmed",
        "flight": {"description": flight, "price": flight_price},
        "hotel": {
            "name": hotel,
            "check_in": check_in,
            "check_out": check_out,
            "nights": nights,
            "price_per_night": hotel_price_per_night,
            "total": hotel_total,
        },
        "guests": guests,
        "total_price_eur": total,
        "currency": "EUR",
    }


@tool
def modify_booking(booking_id: str, changes: str) -> dict:
    """Modify an existing booking.

    Args:
        booking_id: The booking confirmation ID (e.g. 'TM-A1B2C3D4')
        changes: Description of changes to make (e.g. 'change hotel to W Barcelona')
    """
    import random

    price_diff = round(random.uniform(-50.0, 100.0), 2)
    new_total = round(random.uniform(200.0, 800.0), 2)

    return {
        "booking_id": booking_id,
        "status": "modified",
        "changes_applied": changes,
        "price_difference_eur": price_diff,
        "new_total_eur": new_total,
        "message": f"Booking {booking_id} has been updated. {changes}.",
    }


@tool
def get_booking(booking_id: str) -> dict:
    """Retrieve details of an existing booking.

    Args:
        booking_id: The booking confirmation ID (e.g. 'TM-A1B2C3D4')
    """
    import random

    return {
        "booking_id": booking_id,
        "status": random.choice(["confirmed", "modified"]),
        "flight": {
            "description": "Lufthansa LH1834 09:15→12:30",
            "price": round(random.uniform(89.0, 450.0), 2),
        },
        "hotel": {
            "name": "Hilton Barcelona",
            "check_in": "2026-06-19",
            "check_out": "2026-06-22",
            "nights": 3,
            "price_per_night": 185.0,
            "total": 555.0,
        },
        "guests": 1,
        "total_price_eur": round(random.uniform(300.0, 900.0), 2),
    }


@tool
def cancel_booking(booking_id: str) -> dict:
    """Cancel an existing booking.

    Args:
        booking_id: The booking confirmation ID (e.g. 'TM-A1B2C3D4')
    """
    return {
        "booking_id": booking_id,
        "status": "cancelled",
        "refund_eur": round(0.0, 2),
        "message": f"Booking {booking_id} has been cancelled. A full refund will be processed within 5-7 business days.",
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

tools = [check_availability, create_booking, modify_booking, get_booking, cancel_booking]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_MESSAGE = SystemMessage(
    content="""\
You are the Booking Manager for TripMate AI, a travel booking assistant.
Help customers book, modify, and cancel travel reservations.

Guidelines:
- Before booking, check availability for the requested flight and hotel.
- Always confirm the total price before finalizing a booking.
- After any booking change, show a clear itinerary summary.
- Be friendly, professional, and proactive about suggesting alternatives if something is unavailable.
- Use the provided tools to perform all booking operations.
"""
)


def llm_call(state: MessagesState):
    return {
        "messages": [
            llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])
        ]
    }


def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["environment", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "Action"
    return END


def build_agent() -> "StateGraph":
    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("environment", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {"Action": "environment", END: END},
    )
    agent_builder.add_edge("environment", "llm_call")

    return agent_builder.compile()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        agent = build_agent()
        adapter = from_langgraph(agent)
        adapter.run()
    except Exception:
        logger.exception("Booking Manager Agent encountered an error while running")
        raise
