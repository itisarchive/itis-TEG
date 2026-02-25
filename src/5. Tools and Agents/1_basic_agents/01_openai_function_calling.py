"""
🔧 OpenAI Function Calling Fundamentals
========================================

This script demonstrates the core concepts of OpenAI function calling
(also known as "tool use") using the Azure OpenAI API:

1. How to define tool schemas that the model understands
2. How the model autonomously decides when to invoke tools
3. How to execute tool calls and feed results back
4. A complete multi-step tool-calling workflow

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with openai and python-dotenv packages
"""

import json
import textwrap
from typing import Any

from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_named_tool_choice_param import (
    ChatCompletionNamedToolChoiceParam,
    Function as NamedToolChoiceFunction,
)

DEPLOYMENT_MODEL = "gpt-4.1-mini"


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def get_current_weather(*, location: str, unit: str = "fahrenheit") -> str:
    """Returns mock weather data for a given location (would call a real API in production)."""
    weather_data = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_data)


def calculate_power(*, base: float, exponent: float) -> str:
    """Calculates base raised to the power of exponent, returning the result as a string."""
    return str(base ** exponent)


AVAILABLE_TOOL_FUNCTIONS: dict[str, Any] = {
    "get_current_weather": get_current_weather,
    "calculate_power": calculate_power,
}

TOOL_SCHEMAS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_power",
            "description": "Calculate base raised to the power of exponent",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "number", "description": "The base number"},
                    "exponent": {"type": "number", "description": "The exponent"},
                },
                "required": ["base", "exponent"],
            },
        },
    },
]


def demonstrate_basic_tool_call(client: AzureOpenAI) -> None:
    """
    The model receives a user query along with tool definitions.
    When the query matches a tool's purpose, the model responds with
    a tool_call instead of plain text — signaling which function to
    invoke and with what arguments.
    """
    print_section_header("EXAMPLE 1: Weather Query — Model Chooses a Tool")
    print(textwrap.dedent(demonstrate_basic_tool_call.__doc__))

    weather_question = "What's the weather like in Boston?"
    weather_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(role="user", content=weather_question),
    ]

    weather_response = client.chat.completions.create(
        model=DEPLOYMENT_MODEL,
        messages=weather_messages,
        tools=TOOL_SCHEMAS,
    )

    chosen_tool_call = weather_response.choices[0].message.tool_calls
    print(f"User: {weather_question}")
    print(f"Model chose tool call: {chosen_tool_call}")


def demonstrate_no_tool_needed(client: AzureOpenAI) -> None:
    """
    When the user query does not match any available tool, the model
    simply responds with regular text — no tool_calls are produced.
    This shows that tool invocation is purely model-driven and contextual.
    """
    print_section_header("EXAMPLE 2: Irrelevant Query — No Tool Needed")
    print(textwrap.dedent(demonstrate_no_tool_needed.__doc__))

    greeting = "Hello! How are you?"
    greeting_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(role="user", content=greeting),
    ]

    greeting_response = client.chat.completions.create(
        model=DEPLOYMENT_MODEL,
        messages=greeting_messages,
        tools=TOOL_SCHEMAS,
    )

    print(f"User: {greeting}")
    print(f"Model response (plain text): {greeting_response.choices[0].message.content}")


def demonstrate_forced_tool_call(client: AzureOpenAI) -> None:
    """
    You can override the model's judgment and force it to call a specific
    tool using tool_choice={"type": "function", "function": {"name": ...}}.
    Even an irrelevant greeting will produce a tool call in this mode.
    """
    print_section_header("EXAMPLE 3: Forced Tool Call — Overriding Model Choice")
    print(textwrap.dedent(demonstrate_forced_tool_call.__doc__))

    irrelevant_input = "Hello there!"
    forced_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(role="user", content=irrelevant_input),
    ]

    forced_response = client.chat.completions.create(
        model=DEPLOYMENT_MODEL,
        messages=forced_messages,
        tools=TOOL_SCHEMAS,
        tool_choice=ChatCompletionNamedToolChoiceParam(
            type="function",
            function=NamedToolChoiceFunction(name="get_current_weather"),
        ),
    )

    forced_tool_call = forced_response.choices[0].message.tool_calls
    print(f"User: {irrelevant_input}")
    print(f"Forced tool call: {forced_tool_call}")


def demonstrate_complete_tool_workflow(client: AzureOpenAI) -> None:
    """
    Complete tool-calling workflow in five steps:

    1. User asks a question that requires external data
    2. Model responds with a tool_call (function name + arguments)
    3. We execute the function locally with parsed arguments
    4. We append both the assistant's tool_call and the tool's result to the conversation
    5. We send the enriched conversation back — the model produces a final natural-language answer
    """
    print_section_header("EXAMPLE 4: Complete Tool-Calling Workflow")
    print(textwrap.dedent(demonstrate_complete_tool_workflow.__doc__))

    user_question = "What's the weather in San Francisco and what's 2 to the power of 8?"
    conversation_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(role="user", content=user_question),
    ]

    print(f"Step 1 — User question: {user_question}")

    step2_response = client.chat.completions.create(
        model=DEPLOYMENT_MODEL,
        messages=conversation_messages,
        tools=TOOL_SCHEMAS,
    )

    assistant_message = step2_response.choices[0].message
    print("Step 2 — Model chose tool call(s)")

    if not assistant_message.tool_calls:
        print("No tool calls were produced — the model answered directly.")
        return

    conversation_messages.append(assistant_message.to_dict())

    for tool_call in assistant_message.tool_calls:
        called_function_name = tool_call.function.name
        called_function_args = json.loads(tool_call.function.arguments)

        print(f"Step 3 — Executing '{called_function_name}' with args: {called_function_args}")

        resolved_function = AVAILABLE_TOOL_FUNCTIONS[called_function_name]
        tool_result = resolved_function(**called_function_args)

        print(f"         Tool result: {tool_result}")

        conversation_messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=tool_result,
            )
        )

    print("Step 4 — Tool results appended to conversation")

    final_response = client.chat.completions.create(
        model=DEPLOYMENT_MODEL,
        messages=conversation_messages,
    )

    print(f"Step 5 — Final response: {final_response.choices[0].message.content}")


def print_key_takeaways() -> None:
    print_section_header("🎯 Key Takeaways")

    print(textwrap.dedent("""\
        1. Models can intelligently choose when to call tools based on user intent
        2. Tool calls require manual execution — the model only produces the invocation request
        3. Tool results must be appended back to the conversation for the model to use
        4. The complete workflow involves multiple API roundtrips
        5. Tools are defined using JSON Schema format wrapped in a 'tools' array
        6. You can force a specific tool call via tool_choice, or let the model decide
    """))


if __name__ == "__main__":
    load_dotenv(override=True)
    azure_openai_client = AzureOpenAI()

    demonstrate_basic_tool_call(azure_openai_client)
    demonstrate_no_tool_needed(azure_openai_client)
    demonstrate_forced_tool_call(azure_openai_client)
    demonstrate_complete_tool_workflow(azure_openai_client)
    print_key_takeaways()
