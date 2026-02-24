"""
🔀 Simple Graph Implementation with LangGraph
===============================================

Educational script demonstrating graph-based workflows and language model integration.

Graphs are sequences of actions — like steps in a process.
These actions can be simple functions, language model modules, or entire subgraphs.

🎯 What You'll Learn:
- How to define graph state shared between nodes
- How nodes (functions) modify graph state
- How edges (direct and conditional) control the flow
- How to use a language model as a graph node

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain-openai, langgraph, python-dotenv
"""

import random
import textwrap
from typing import Annotated, Literal, TypedDict

import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


class SimpleGraphState(TypedDict):
    graph_state: str


def append_i_am(state: SimpleGraphState):
    """First node in the chain — appends 'I am' to the current state."""
    print("--- Node 1: append_i_am ---")
    return {"graph_state": state["graph_state"] + " I am"}


def append_happy_mood(state: SimpleGraphState):
    """Positive-mood node — appends 'happy :)' to the state."""
    print("--- Node 2: append_happy_mood ---")
    return {"graph_state": state["graph_state"] + " happy :)"}


def append_sad_mood(state: SimpleGraphState):
    """Negative-mood node — appends 'sad :(' to the state."""
    print("--- Node 3: append_sad_mood ---")
    return {"graph_state": state["graph_state"] + " sad :("}


def random_mood_router(_state: SimpleGraphState) -> Literal["happy_mood", "sad_mood"]:
    """Conditional edge that randomly routes to a happy or sad mood node (50/50 chance)."""
    random_value = random.random()
    print(f"Random value: {random_value}")

    if random_value < 0.5:
        return "happy_mood"
    return "sad_mood"


class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def demonstrate_simple_conditional_graph() -> None:
    """
    PART 1 — Graphs as Sequences of Actions

    Each node is a function that modifies the shared graph state.
    State is represented by a TypedDict storing variables and their values.
    Nodes are connected through edges — direct or conditional — that define task flow.
    """
    print_section_header("PART 1: SIMPLE GRAPH WITH CONDITIONAL ROUTING")
    print(textwrap.dedent(demonstrate_simple_conditional_graph.__doc__))

    conditional_graph_builder = StateGraph(SimpleGraphState)  # type: ignore[arg-type]
    conditional_graph_builder.add_node("introduce", append_i_am)  # type: ignore[arg-type]
    conditional_graph_builder.add_node("happy_mood", append_happy_mood)  # type: ignore[arg-type]
    conditional_graph_builder.add_node("sad_mood", append_sad_mood)  # type: ignore[arg-type]

    conditional_graph_builder.add_edge(START, "introduce")
    conditional_graph_builder.add_conditional_edges("introduce", random_mood_router)
    conditional_graph_builder.add_edge("happy_mood", END)
    conditional_graph_builder.add_edge("sad_mood", END)

    conditional_graph = conditional_graph_builder.compile()

    simple_graph_result = conditional_graph.invoke({"graph_state": "Hello, I am Tom."})  # type: ignore[arg-type]
    print(f"Final result: {simple_graph_result}")


def demonstrate_llm_as_graph_node(azure_chat_llm: AzureChatOpenAI) -> None:
    """
    PART 2 — Language Model as a Graph Component

    A language model can serve as a node inside a LangGraph workflow.
    The node receives a conversation state (list of messages), invokes the LLM,
    and returns the updated messages list with the assistant's response appended.
    """
    print_section_header("PART 2: LANGUAGE MODEL AS A GRAPH NODE")
    print(textwrap.dedent(demonstrate_llm_as_graph_node.__doc__))

    def invoke_llm_node(state: ConversationState):
        """Node that forwards the conversation to the Azure LLM and returns its reply."""
        return {"messages": [azure_chat_llm.invoke(state["messages"])]}

    llm_graph_builder = StateGraph(ConversationState)  # type: ignore[arg-type]
    llm_graph_builder.add_node("llm_node", invoke_llm_node)  # type: ignore[arg-type]
    llm_graph_builder.add_edge(START, "llm_node")
    llm_graph_builder.add_edge("llm_node", END)

    llm_graph = llm_graph_builder.compile()

    llm_graph_result = llm_graph.invoke(
        {"messages": HumanMessage(content="Tell me about Paris")})  # type: ignore[arg-type]

    for message in llm_graph_result["messages"]:
        if hasattr(message, "content"):
            print(
                f"Message type: {type(message).__name__}\n"
                f"Content: {message.content[:200]}...\n"
                f"---"
            )


if __name__ == "__main__":
    load_dotenv(override=True)
    nest_asyncio.apply()

    azure_llm = AzureChatOpenAI(model="gpt-4.1-mini")

    demonstrate_simple_conditional_graph()
    demonstrate_llm_as_graph_node(azure_chat_llm=azure_llm)
