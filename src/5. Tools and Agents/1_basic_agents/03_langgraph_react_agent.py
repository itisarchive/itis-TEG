"""
🧠 LangGraph ReAct Agent Foundation
====================================

This script demonstrates LangGraph's ReAct (Reasoning + Acting) agent pattern:

1. Creating agents with create_react_agent
2. Stateful conversations with thread-based persistence (MemorySaver)
3. Multi-tool agents that reason about which tool to select
4. How LangGraph manages the reasoning → action → observation cycle

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langgraph, langchain-openai, python-dotenv packages
"""

import textwrap
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

DEPLOYMENT_MODEL = "gpt-4.1-mini"

REACT_AGENT_SYSTEM_PROMPT = """\
You are a helpful assistant that can perform calculations and text analysis.

When working with numbers, show your reasoning step by step.
When asked to perform multiple operations, break them down clearly.
Always explain what you're doing before using tools."""


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    print(f"🔢 Computing: {a} × {b}")
    return a * b


@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    print(f"➕ Computing: {a} + {b}")
    return a + b


@tool
def power(a: int, b: int) -> int:
    """
    Calculate a raised to the power of b.

    Args:
        a: Base number
        b: Exponent
    """
    print(f"⚡ Computing: {a}^{b}")
    return a ** b


@tool
def get_word_length(word: str) -> int:
    """
    Get the length of a word in characters.

    Args:
        word: The word to measure
    """
    print(f"📏 Measuring word: '{word}'")
    return len(word)


def build_react_agent(llm: AzureChatOpenAI) -> CompiledStateGraph:
    """
    Creates a ReAct agent graph with math and text analysis tools,
    a system prompt defining agent behavior, and an in-memory
    checkpointer that enables stateful (thread-based) conversations.
    """
    available_tools = [multiply, add, power, get_word_length]

    return create_react_agent(
        llm,
        tools=available_tools,
        prompt=REACT_AGENT_SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


def invoke_agent_with_query(
        agent_graph: CompiledStateGraph,
        *,
        query: str,
        config: RunnableConfig,
) -> str:
    """Sends a user query to the agent graph and returns the final message content."""
    agent_input: dict[str, Any] = {"messages": [HumanMessage(content=query)]}
    agent_output = agent_graph.invoke(agent_input, config=config)  # type: ignore[arg-type]
    return agent_output["messages"][-1].content


def demonstrate_simple_calculation(agent_graph: CompiledStateGraph, session_config: RunnableConfig) -> None:
    """
    A single-tool invocation: the agent recognizes that the user's question
    maps to the multiply tool, calls it, and returns the result.
    """
    print_section_header("EXAMPLE 1: Simple Calculation")
    print(textwrap.dedent(demonstrate_simple_calculation.__doc__))

    simple_math_query = "What's 8 multiplied by 7?"
    print(f"Query: {simple_math_query}")

    simple_math_answer = invoke_agent_with_query(
        agent_graph, query=simple_math_query, config=session_config,
    )
    print(f"Response: {simple_math_answer}")


def demonstrate_multi_step_calculation(agent_graph: CompiledStateGraph, session_config: RunnableConfig) -> None:
    """
    The agent chains two tool calls: first it calls power(), then feeds
    the intermediate result into add() — all within a single user request.
    """
    print_section_header("EXAMPLE 2: Multi-Step Calculation")
    print(textwrap.dedent(demonstrate_multi_step_calculation.__doc__))

    multi_step_query = "Calculate 5 to the power of 3, then add 20 to the result"
    print(f"Query: {multi_step_query}")

    multi_step_answer = invoke_agent_with_query(
        agent_graph, query=multi_step_query, config=session_config,
    )
    print(f"Response: {multi_step_answer}")


def demonstrate_mixed_tool_operations(agent_graph: CompiledStateGraph, session_config: RunnableConfig) -> None:
    """
    Mixing different tool categories: the agent first measures word length
    (text tool), then multiplies the result (math tool) — demonstrating
    cross-domain tool chaining.
    """
    print_section_header("EXAMPLE 3: Mixed Operations")
    print(textwrap.dedent(demonstrate_mixed_tool_operations.__doc__))

    mixed_ops_query = "How many characters are in the word 'LangGraph'? Then multiply that by 4."
    print(f"Query: {mixed_ops_query}")

    mixed_ops_answer = invoke_agent_with_query(
        agent_graph, query=mixed_ops_query, config=session_config,
    )
    print(f"Response: {mixed_ops_answer}")


def demonstrate_conversational_memory(agent_graph: CompiledStateGraph, session_config: RunnableConfig) -> None:
    """
    The MemorySaver checkpointer persists all messages within a thread_id.
    Asking about "the previous calculation" works because the agent has
    access to the full conversation history from earlier examples.
    """
    print_section_header("EXAMPLE 4: Conversational Context")
    print(textwrap.dedent(demonstrate_conversational_memory.__doc__))

    context_recall_query = "Can you remind me what the result was from the previous calculation?"
    print(f"Query: {context_recall_query}")

    context_recall_answer = invoke_agent_with_query(
        agent_graph, query=context_recall_query, config=session_config,
    )
    print(f"Response: {context_recall_answer}")


def demonstrate_fresh_session(agent_graph: CompiledStateGraph, fresh_session_config: RunnableConfig) -> None:
    """
    A new thread_id starts a blank conversation — the agent has no memory
    of interactions from other threads, proving that context isolation works.
    """
    print_section_header("EXAMPLE 5: New Session (Fresh Context)")
    print(textwrap.dedent(demonstrate_fresh_session.__doc__))

    no_context_query = "What was the result from the previous calculation?"
    print(f"Query: {no_context_query}")

    no_context_answer = invoke_agent_with_query(
        agent_graph, query=no_context_query, config=fresh_session_config,
    )
    print(f"Response: {no_context_answer}")


def demonstrate_complex_multi_tool_reasoning(agent_graph: CompiledStateGraph, session_config: RunnableConfig) -> None:
    """
    A three-step pipeline expressed in natural language: the agent must
    plan a sequence of get_word_length → power → add, executing each
    tool in order while carrying intermediate results forward.
    """
    print_section_header("EXAMPLE 6: Complex Multi-Tool Reasoning")
    print(textwrap.dedent(demonstrate_complex_multi_tool_reasoning.__doc__))

    complex_query = textwrap.dedent("""\
        I have a word 'Python' and I want to:
        1. Find out how many characters it has
        2. Raise that number to the power of 2
        3. Add 15 to the final result

        Please work through this step by step.""")

    print(f"Complex Query: {complex_query}")

    complex_answer = invoke_agent_with_query(
        agent_graph, query=complex_query, config=session_config,
    )
    print(f"Response: {complex_answer}")


def demonstrate_agent_without_tools(agent_graph: CompiledStateGraph, session_config: RunnableConfig) -> None:
    """
    When the user's request does not require any tool, the agent simply
    responds with its own reasoning — no tool calls are generated.
    This shows that tool invocation is purely intent-driven.
    """
    print_section_header("EXAMPLE 7: Agent Decision Making")
    print(textwrap.dedent(demonstrate_agent_without_tools.__doc__))

    suggestion_query = "I need to do some math but I'm not sure what. Can you suggest something?"
    print(f"Query: {suggestion_query}")

    suggestion_answer = invoke_agent_with_query(
        agent_graph, query=suggestion_query, config=session_config,
    )
    print(f"Response: {suggestion_answer}")


def print_conversation_analysis(
        agent_graph: CompiledStateGraph,
        *,
        first_session_config: RunnableConfig,
        second_session_config: RunnableConfig,
) -> None:
    print_section_header("CONVERSATION ANALYSIS")

    first_session_state = agent_graph.get_state(first_session_config)
    second_session_state = agent_graph.get_state(second_session_config)

    print(textwrap.dedent(f"""\
        Session 1 conversation history:
        Total messages in session 1: {len(first_session_state.values['messages'])}

        Session 2 conversation history:
        Total messages in session 2: {len(second_session_state.values['messages'])}"""))


def print_key_takeaways() -> None:
    print_section_header("🎯 Key Takeaways")

    print(textwrap.dedent("""\
        1. ReAct agents combine reasoning with tool execution automatically
        2. Thread-based persistence maintains conversation context
        3. Agents make intelligent decisions about which tools to use
        4. Multi-step reasoning is handled seamlessly
        5. Different thread IDs create separate conversation contexts
        6. LangGraph manages the observe → think → act cycle
    """))


if __name__ == "__main__":
    load_dotenv(override=True)
    azure_llm = AzureChatOpenAI(model=DEPLOYMENT_MODEL, temperature=0)

    react_agent_graph = build_react_agent(azure_llm)

    session_1_config: RunnableConfig = {"configurable": {"thread_id": "math_session_1"}}
    session_2_config: RunnableConfig = {"configurable": {"thread_id": "math_session_2"}}

    demonstrate_simple_calculation(react_agent_graph, session_config=session_1_config)
    demonstrate_multi_step_calculation(react_agent_graph, session_config=session_1_config)
    demonstrate_mixed_tool_operations(react_agent_graph, session_config=session_1_config)
    demonstrate_conversational_memory(react_agent_graph, session_config=session_1_config)
    demonstrate_fresh_session(react_agent_graph, fresh_session_config=session_2_config)
    demonstrate_complex_multi_tool_reasoning(react_agent_graph, session_config=session_2_config)
    demonstrate_agent_without_tools(react_agent_graph, session_config=session_2_config)
    print_conversation_analysis(
        react_agent_graph,
        first_session_config=session_1_config,
        second_session_config=session_2_config,
    )
    print_key_takeaways()
