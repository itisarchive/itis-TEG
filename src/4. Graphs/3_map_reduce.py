"""
🗺️ Map-Reduce Pattern with LangGraph
======================================

Demonstrates the map-reduce pattern for generating and selecting jokes:

- MAP phase:   Parallel joke generation on different subtopics
                (each subtopic gets its own node via Send())
- REDUCE phase: Selection of the best joke from the generated set

🎯 What You'll Learn:
- How to use Send() for dynamic fan-out (map) in LangGraph
- How operator.add aggregates results from parallel nodes
- How to combine map and reduce phases in a single graph
- How to use Pydantic models for structured LLM output

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain-openai, langgraph, pydantic, python-dotenv
"""

import operator
import textwrap
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send
from pydantic import BaseModel

SUBTOPICS_GENERATION_PROMPT = (
    "Generate a list of several, max 5, subtopics that are related to the general topic: {topic}."
)
SINGLE_JOKE_PROMPT = "Create a joke about {subject}"
BEST_JOKE_SELECTION_PROMPT = (
    "Below you will find several jokes about {topic}. "
    "Choose the best one! Return the ID of the best one, "
    "starting from 0 as the ID of the first joke. Jokes: \n\n  {jokes}"
)


class SubtopicList(BaseModel):
    subjects: list[str]


class BestJokeChoice(BaseModel):
    id: int


class GeneratedJoke(BaseModel):
    joke: str


class MapReduceJokeState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


class SingleJokeGenerationState(TypedDict):
    subject: str


def generate_subtopics(
        state: MapReduceJokeState,
        *,
        azure_chat_llm: AzureChatOpenAI,
):
    """Generates a list of subtopics based on the main topic using structured LLM output."""
    subtopics_prompt = SUBTOPICS_GENERATION_PROMPT.format(topic=state["topic"])
    subtopics_response = azure_chat_llm.with_structured_output(SubtopicList).invoke(subtopics_prompt)
    return {"subjects": subtopics_response.subjects}


def fan_out_joke_generation(state: MapReduceJokeState):
    """
    Uses Send() to dynamically fan out parallel joke generation — one node per subtopic.
    Send() allows passing any state to the target node, enabling the map phase.
    """
    return [Send("generate_single_joke", {"subject": subtopic}) for subtopic in state["subjects"]]


def generate_single_joke(
        state: SingleJokeGenerationState,
        *,
        azure_chat_llm: AzureChatOpenAI,
):
    """
    MAP phase: Generates a single joke for a given subtopic.
    Results are automatically aggregated in MapReduceJokeState via operator.add.
    """
    joke_prompt = SINGLE_JOKE_PROMPT.format(subject=state["subject"])
    joke_response = azure_chat_llm.with_structured_output(GeneratedJoke).invoke(joke_prompt)
    return {"jokes": [joke_response.joke]}


def select_best_joke(
        state: MapReduceJokeState,
        *,
        azure_chat_llm: AzureChatOpenAI,
):
    """
    REDUCE phase: Selects the best joke from all generated ones.
    Aggregates results from the MAP phase and makes a final selection.
    """
    all_jokes_text = "\n\n".join(state["jokes"])
    selection_prompt = BEST_JOKE_SELECTION_PROMPT.format(
        topic=state["topic"],
        jokes=all_jokes_text,
    )
    selection_response = azure_chat_llm.with_structured_output(BestJokeChoice).invoke(selection_prompt)
    return {"best_selected_joke": state["jokes"][selection_response.id]}


def demonstrate_map_reduce_joke_generation(azure_chat_llm: AzureChatOpenAI) -> None:
    """
    Map-Reduce Joke Generator

    The graph fans out joke generation across subtopics (MAP),
    then selects the single best joke from all candidates (REDUCE).
    Higher temperature (0.7) is used for creative joke generation.
    """
    print_section_header("MAP-REDUCE JOKE GENERATION")
    print(textwrap.dedent(demonstrate_map_reduce_joke_generation.__doc__))

    def generate_subtopics_node(current_state: MapReduceJokeState):
        """Graph node wrapper injecting the LLM into subtopic generation."""
        return generate_subtopics(current_state, azure_chat_llm=azure_chat_llm)

    def generate_single_joke_node(current_state: SingleJokeGenerationState):
        """Graph node wrapper injecting the LLM into single joke generation."""
        return generate_single_joke(current_state, azure_chat_llm=azure_chat_llm)

    def select_best_joke_node(current_state: MapReduceJokeState):
        """Graph node wrapper injecting the LLM into best joke selection."""
        return select_best_joke(current_state, azure_chat_llm=azure_chat_llm)

    map_reduce_builder = StateGraph(MapReduceJokeState)  # type: ignore[arg-type]
    map_reduce_builder.add_node("generate_subtopics", generate_subtopics_node)  # type: ignore[arg-type]
    map_reduce_builder.add_node("generate_single_joke", generate_single_joke_node)  # type: ignore[arg-type]
    map_reduce_builder.add_node("select_best_joke", select_best_joke_node)  # type: ignore[arg-type]

    map_reduce_builder.add_edge(START, "generate_subtopics")
    map_reduce_builder.add_conditional_edges("generate_subtopics", fan_out_joke_generation, ["generate_single_joke"])
    map_reduce_builder.add_edge("generate_single_joke", "select_best_joke")
    map_reduce_builder.add_edge("select_best_joke", END)

    map_reduce_graph = map_reduce_builder.compile()

    print(textwrap.dedent("""\
        Topic: future of banking
        Streaming graph execution step by step...
    """))

    for graph_step in map_reduce_graph.stream({"topic": "future of banking"}):  # type: ignore[arg-type]
        print(graph_step)

    print("\n=== Process completed ===")


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


if __name__ == "__main__":
    load_dotenv(override=True)

    azure_llm = AzureChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    demonstrate_map_reduce_joke_generation(azure_chat_llm=azure_llm)
