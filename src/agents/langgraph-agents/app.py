import os
import sys
import logging
import random
from typing import TypedDict

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from azure.ai.agentserver.langgraph import from_langgraph
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Azure AI Foundry Tracing Setup
# https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain-traces
# ---------------------------------------------------------------------------
from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

def get_logger(module_name):
    return logging.getLogger(f"app.{module_name}")

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT", "").strip()
deployment_name = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

# Create the Azure AI Foundry tracer
if project_endpoint:
    tracer = AzureAIOpenTelemetryTracer(
        project_endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
        name="langgraph-tracing-demo",
        agent_id="three-agent-consensus",
    )
    logger.info("Azure AI Foundry tracing configured.")
else:
    tracer = None
    logger.warning(
        "AZURE_AI_PROJECT_ENDPOINT not set. "
        "Traces will NOT be exported to Azure AI Foundry."
    )

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
    logger.exception("LangGraph agents app failed to start")
    raise

class State(TypedDict):
    messages: list[BaseMessage]
    agent1_response: str
    agent2_response: str
    agent3_response: str
    agent3_agrees_with: str  # "agent1" or "agent2"
    final_answer: str


def agent1(state: State):
    """First agent: generates an initial response to the user's question."""
    response = llm.invoke(state["messages"])
    logger.info(f"Agent 1 response: {response.content}")
    return {
        "agent1_response": response.content,
    }


def agent2(state: State):
    """Second agent: always provides a different/contrarian perspective."""
    messages = state["messages"] + [
        SystemMessage(content=f"""You must provide a DIFFERENT and CONTRARIAN response to the user's question.
Another agent already responded with: "{state['agent1_response']}"

You MUST disagree or provide a completely different perspective. Do NOT agree with the previous response.
Give a substantive alternative answer that takes a different viewpoint or approach.""")
    ]
    response = llm.invoke(messages)
    logger.info(f"Agent 2 response (contrarian): {response.content}")
    return {
        "agent2_response": response.content,
    }


def agent3(state: State):
    """Third agent: randomly agrees with either agent 1 or agent 2."""
    agrees_with = random.choice(["agent1", "agent2"])
    
    if agrees_with == "agent1":
        chosen_response = state["agent1_response"]
        other_response = state["agent2_response"]
    else:
        chosen_response = state["agent2_response"]
        other_response = state["agent1_response"]
    
    messages = state["messages"] + [
        SystemMessage(content=f"""Two agents have provided different responses to the user's question.

Response A: "{chosen_response}"
Response B: "{other_response}"

You agree with Response A. Explain why you support this response and provide your endorsement of it.
Keep your response concise but make it clear you're supporting Response A's position.""")
    ]
    response = llm.invoke(messages)
    logger.info(f"Agent 3 agrees with {agrees_with}: {response.content}")
    return {
        "agent3_response": response.content,
        "agent3_agrees_with": agrees_with,
    }


def determine_majority(state: State):
    """Determines the majority consensus and returns the final answer."""
    # Count votes: Agent 1 always votes for agent1, Agent 2 always votes for agent2
    # Agent 3's vote is random (stored in agent3_agrees_with)
    votes = {"agent1": 1, "agent2": 1}  # Each agent votes for their own response
    votes[state["agent3_agrees_with"]] += 1  # Agent 3's deciding vote
    
    if votes["agent1"] > votes["agent2"]:
        winner = "agent1"
        winning_response = state["agent1_response"]
    else:
        winner = "agent2"
        winning_response = state["agent2_response"]
    
    logger.info(f"Majority vote result: {winner} wins with {votes[winner]} votes (Agent1: {votes['agent1']}, Agent2: {votes['agent2']})")
    logger.info(f"Agent 3 agreed with: {state['agent3_agrees_with']}")
    
    return {
        "final_answer": winning_response,
    }


def build_graph() -> "StateGraph":
    # Build the graph with three agents and majority consensus
    graph_builder = StateGraph(State)
    
    # Add nodes with metadata for Azure AI Foundry tracing
    graph_builder.add_node(
        "agent1",
        agent1,
        metadata={
            "agent_name": "InitialResponseAgent",
            "agent_id": "agent1-initial",
            "otel_agent_span": True,
        },
    )
    graph_builder.add_node(
        "agent2",
        agent2,
        metadata={
            "agent_name": "ContrarianAgent",
            "agent_id": "agent2-contrarian",
            "otel_agent_span": True,
        },
    )
    graph_builder.add_node(
        "agent3",
        agent3,
        metadata={
            "agent_name": "TiebreakerAgent",
            "agent_id": "agent3-tiebreaker",
            "otel_agent_span": True,
        },
    )
    graph_builder.add_node(
        "determine_majority",
        determine_majority,
        metadata={
            "agent_name": "MajorityVoteAgent",
            "agent_id": "majority-voter",
            "otel_agent_span": True,
        },
    )
    
    # Set entry point
    graph_builder.set_entry_point("agent1")
    
    # Define edges: agent1 -> agent2 -> agent3 -> determine_majority -> END
    graph_builder.add_edge("agent1", "agent2")
    graph_builder.add_edge("agent2", "agent3")
    graph_builder.add_edge("agent3", "determine_majority")
    graph_builder.add_edge("determine_majority", END)

    graph = graph_builder.compile()
    return graph

# def run_graph(human_input: str):
#     initial_state = {
#         "messages": [HumanMessage(content=human_input)],
#         "agent1_response": "",
#         "agent2_response": "",
#         "agent3_response": "",
#         "agent3_agrees_with": "",
#         "final_answer": "",
#     }

#     logger.info(f"User input: {human_input}")
#     logger.info("=" * 60)
#     logger.info("Starting three-agent majority consensus flow...")
#     logger.info("=" * 60)
    
#     try:
#         # Pass the tracer as a callback to capture all LangGraph nodes
#         config = {"callbacks": [tracer]} if tracer else {}
#         final_state = graph.invoke(initial_state, config=config)
        
#         logger.info("=" * 60)
#         logger.info("FINAL MAJORITY ANSWER:")
#         logger.info("=" * 60)
#         logger.info(final_state["final_answer"])
#     except Exception as e:
#         logger.error(f"Error running graph: {e}")
#         print(e)
    
#     logger.info("Graph run complete.")


if __name__ == "__main__":
    try:
        agent = build_graph()
        adapter = from_langgraph(agent)
        adapter.run()
    except Exception:
        logger.exception("Order Agent encountered an error while running")
        raise