"""
🔗 LangChain Tools Introduction
================================

This script demonstrates LangChain's approach to tools and agents,
contrasting it with the raw OpenAI function calling from the previous example:

1. Creating tools with the @tool decorator (automatic JSON schema generation)
2. Manual tool execution vs fully automated agent-based execution
3. LangChain agents and the AgentExecutor tool-calling loop
4. Tool introspection and comparing both approaches

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, python-dotenv packages
"""

import json
import textwrap
from typing import cast

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

DEPLOYMENT_MODEL = "gpt-4.1-mini"


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


@tool
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """
    Get the current weather in a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA.
        unit: The unit of temperature, can be 'celsius' or 'fahrenheit'. Defaults to 'fahrenheit'.
    """
    weather_data = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_data)


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers together.

    Args:
        a: First number to multiply
        b: Second number to multiply
    """
    print(f"Multiplying {a} × {b} = {a * b}")
    return a * b


@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a: First number to add
        b: Second number to add
    """
    print(f"Adding {a} + {b} = {a + b}")
    return a + b


def demonstrate_manual_tool_execution(llm: AzureChatOpenAI) -> None:
    """
    bind_tools() attaches tool schemas to the model. When invoked, the model
    returns an AIMessage containing tool_calls metadata instead of executing
    anything — you must invoke the tool manually with the parsed arguments.
    """
    print_section_header("EXAMPLE 1: Manual Tool Execution")
    print(textwrap.dedent(demonstrate_manual_tool_execution.__doc__))

    llm_with_tools = llm.bind_tools([get_current_weather, multiply])

    weather_query = "What's the weather in San Francisco?"
    tool_call_response = cast(AIMessage, llm_with_tools.invoke(weather_query))

    print(textwrap.dedent(f"""\
        Model response:
        Content: {tool_call_response.content}
        Tool calls: {tool_call_response.tool_calls}"""))

    if tool_call_response.tool_calls:
        first_tool_call = tool_call_response.tool_calls[0]
        print(f"\nTool to call: {first_tool_call['name']}")
        print(f"Arguments: {first_tool_call['args']}")

        if first_tool_call["name"] == "get_current_weather":
            manual_execution_result = get_current_weather.invoke(first_tool_call["args"])
            print(f"Manual execution result: {manual_execution_result}")


def demonstrate_model_without_tools(llm: AzureChatOpenAI) -> None:
    """
    Without tools bound, the model can only respond with its internal
    knowledge — it cannot reach external data or perform actions.
    """
    print_section_header("EXAMPLE 2: Model Without Tools")
    print(textwrap.dedent(demonstrate_model_without_tools.__doc__))

    plain_response = llm.invoke("What's the weather in San Francisco?")
    print(f"Response without tools:\n{plain_response.content}")


def demonstrate_agent_with_executor(llm: AzureChatOpenAI) -> AgentExecutor:
    """
    A LangChain agent wraps a model + tools + prompt into a single unit.
    The AgentExecutor handles the full observe → think → act loop:
    it automatically invokes tools, feeds results back to the model,
    and repeats until the model produces a final answer.
    """
    print_section_header("EXAMPLE 3: LangChain Agent with Tools")
    print(textwrap.dedent(demonstrate_agent_with_executor.__doc__))

    agent_prompt = hub.pull("hwchase17/openai-tools-agent")
    available_tools = [get_current_weather, multiply, add]

    tool_calling_agent = create_tool_calling_agent(llm, available_tools, agent_prompt)
    executor = AgentExecutor(agent=tool_calling_agent, tools=available_tools, verbose=True)

    multiplication_query = "What's 5 multiplied by 8?"
    print(f"Query: {multiplication_query}")
    multiplication_result = executor.invoke({"input": multiplication_query})
    print(f"Agent result: {multiplication_result['output']}")

    return executor


def demonstrate_multi_step_query(executor: AgentExecutor) -> None:
    """
    Agents shine on multi-step tasks: the model chains tool calls
    automatically — e.g., first fetching weather data, then performing
    arithmetic on the returned temperature.
    """
    print_section_header("EXAMPLE 4: Multi-Step Query")
    print(textwrap.dedent(demonstrate_multi_step_query.__doc__))

    multi_step_query = "Get the weather in Boston, then multiply the temperature by 2"
    print(f"Complex query: {multi_step_query}")

    multi_step_result = executor.invoke({"input": multi_step_query})
    print(f"Multi-step result: {multi_step_result['output']}")


def demonstrate_tool_introspection() -> None:
    """
    Every @tool-decorated function exposes its name, description,
    and full JSON Schema via args_schema — this is exactly what
    the model sees when deciding which tool to invoke.
    """
    print_section_header("EXAMPLE 5: Tool Introspection")
    print(textwrap.dedent(demonstrate_tool_introspection.__doc__))

    available_tools = [get_current_weather, multiply, add]

    print("Available tools:")
    for registered_tool in available_tools:
        print(f"- {registered_tool.name}: {registered_tool.description}")
        print(f"  Schema: {registered_tool.args_schema.model_json_schema()}")


def print_approach_comparison() -> None:
    print_section_header("EXAMPLE 6: Comparing Approaches")

    print(textwrap.dedent("""\
        OpenAI Function Calling:
        + Direct control over function execution
        + Minimal abstraction
        - Manual conversation management
        - More boilerplate code

        LangChain Tools & Agents:
        + Automatic tool execution loop
        + Rich tool ecosystem
        + Conversation management handled
        + Easy tool composition
        - Additional abstraction layer"""))


def print_key_takeaways() -> None:
    print_section_header("🎯 Key Takeaways")

    print(textwrap.dedent("""\
        1. @tool decorator automatically generates schemas from docstrings
        2. LangChain agents handle the tool execution loop automatically
        3. AgentExecutor manages conversation flow and tool invocation
        4. Tools can be composed and reused across different agents
        5. Rich docstrings improve model understanding of tool purposes
    """))


if __name__ == "__main__":
    load_dotenv(override=True)
    azure_llm = AzureChatOpenAI(model=DEPLOYMENT_MODEL, temperature=0)

    demonstrate_manual_tool_execution(azure_llm)
    demonstrate_model_without_tools(azure_llm)
    agent_executor = demonstrate_agent_with_executor(azure_llm)
    demonstrate_multi_step_query(agent_executor)
    demonstrate_tool_introspection()
    print_approach_comparison()
    print_key_takeaways()
