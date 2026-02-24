"""
⚡ Parallel Node Execution in LangGraph
========================================

This script demonstrates parallel processing in LangGraph with two examples:
1. Simple asynchronous process with nodes adding names to a list
2. Parallel processing using LLM, Tavily and Wikipedia

In production environments, asynchronous processing and parallelization become
critical. This affects not only reducing user wait time for responses, but also
the ability to scale the system and containerize it.

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Tavily API key in .env file (TAVILY_API_KEY)
- Python 3.13+ with langchain-openai, langgraph, python-dotenv
"""

import operator
import textwrap
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


class ParallelNodeListState(TypedDict):
    state: Annotated[list, operator.add]


class NodeValueAppender:
    """A graph node that appends its assigned label to the shared state list."""

    def __init__(self, node_label: str):
        self._label = node_label

    def __call__(self, current_state: ParallelNodeListState) -> Any:
        print(f"Adding {self._label} to {current_state['state']}")
        return {"state": [self._label]}


class ParallelSearchState(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


def search_web_with_tavily(search_state: ParallelSearchState):
    """Searches internet resources using Tavily and returns formatted results as context."""
    print("... Searching internet resources using Tavily ... \n")

    tavily_search = TavilySearchResults(max_results=3)
    tavily_results = tavily_search.invoke(search_state["question"])

    formatted_tavily_results = "\n\n---\n\n".join(
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in tavily_results
    )

    return {"context": [formatted_tavily_results]}


def search_wikipedia(search_state: ParallelSearchState):
    """Searches Wikipedia and returns formatted article excerpts as context."""
    print("... Searching Wikipedia resources ... \n")

    wiki_documents = WikipediaLoader(
        query=search_state["question"],
        load_max_docs=2,
    ).load()

    formatted_wiki_results = "\n\n---\n\n".join(
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n'
        f"{doc.page_content}\n</Document>"
        for doc in wiki_documents
    )

    return {"context": [formatted_wiki_results]}


def generate_answer_from_context(
        search_state: ParallelSearchState,
        *,
        azure_chat_llm: AzureChatOpenAI,
):
    """Generates a final answer by combining the collected context with the original question."""
    answer_instructions = (
        f"Answer the question {search_state['question']} "
        f"using this context: {search_state['context']}"
    )

    llm_answer = azure_chat_llm.invoke([
        SystemMessage(content=answer_instructions),
        HumanMessage(content="Answer the question."),
    ])

    return {"answer": llm_answer}


def demonstrate_parallel_node_execution() -> None:
    """
    PART 1 — Simple Asynchronous Process

    Nodes introduce themselves and add their name to a single shared list.
    Nodes B and C execute in parallel after A, then D collects both results.
    """
    print_section_header("PART 1: SIMPLE ASYNCHRONOUS PROCESS")
    print(textwrap.dedent(demonstrate_parallel_node_execution.__doc__))

    parallel_builder = StateGraph(ParallelNodeListState)  # type: ignore[arg-type]
    parallel_builder.add_node("a", NodeValueAppender("I am A"))  # type: ignore[arg-type]
    parallel_builder.add_node("b", NodeValueAppender("I am B"))  # type: ignore[arg-type]
    parallel_builder.add_node("c", NodeValueAppender("I am C"))  # type: ignore[arg-type]
    parallel_builder.add_node("d", NodeValueAppender("I am D"))  # type: ignore[arg-type]

    parallel_builder.add_edge(START, "a")
    parallel_builder.add_edge("a", "b")
    parallel_builder.add_edge("a", "c")
    parallel_builder.add_edge("b", "d")
    parallel_builder.add_edge("c", "d")
    parallel_builder.add_edge("d", END)

    parallel_graph = parallel_builder.compile()

    print("Running simple asynchronous graph:")
    parallel_result = parallel_graph.invoke({"state": []})  # type: ignore[arg-type]
    print(f"Result: {parallel_result}\n")


def demonstrate_parallel_search_with_llm(azure_chat_llm: AzureChatOpenAI) -> None:
    """
    PART 2 — Parallel Processing with LLM

    The system answers questions using two sources searched in parallel:
    Internet (Tavily) and Wikipedia. Results are merged as context for the LLM.
    """
    print_section_header("PART 2: PARALLEL PROCESSING WITH LLM")
    print(textwrap.dedent(demonstrate_parallel_search_with_llm.__doc__))

    def generate_answer_node(current_search_state: ParallelSearchState):
        """Graph node wrapper that injects the LLM into the answer generation."""
        return generate_answer_from_context(
            current_search_state,
            azure_chat_llm=azure_chat_llm,
        )

    search_builder = StateGraph(ParallelSearchState)  # type: ignore[arg-type]
    search_builder.add_node("search_internet", search_web_with_tavily)  # type: ignore[arg-type]
    search_builder.add_node("search_wikipedia", search_wikipedia)  # type: ignore[arg-type]
    search_builder.add_node("generate_answer", generate_answer_node)  # type: ignore[arg-type]

    search_builder.add_edge(START, "search_wikipedia")
    search_builder.add_edge(START, "search_internet")
    search_builder.add_edge("search_wikipedia", "generate_answer")
    search_builder.add_edge("search_internet", "generate_answer")
    search_builder.add_edge("generate_answer", END)

    search_graph = search_builder.compile()

    example_question = (
        "What is a business potential of LLM-based multi-agent systems in banking? "
        "Suggest the most interesting business cases"
    )
    print(f"Asking question: {example_question}\n")

    search_result = search_graph.invoke({"question": example_question})  # type: ignore[arg-type]

    print(f"=== ANSWER ===\n{search_result['answer'].content}\n")

    print(f"=== CONTEXT INFORMATION ===\nCollected {len(search_result['context'])} information sources:")
    for source_index, source_content in enumerate(search_result["context"], start=1):
        print(f"Source {source_index}: {len(source_content)} characters")

    print(textwrap.dedent("""\

        === OTHER QUESTIONS TO TRY ===
        You can test the system with other questions, for example:
        - "What is the potential of generative technologies in banking?"
        - "What is the current state of floods in Poland?"
        - "What are the latest developments in artificial intelligence?"

        To ask a new question, use:
        new_result = search_graph.invoke({"question": "your question"})
        print(new_result["answer"].content)"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    azure_llm = AzureChatOpenAI(model="gpt-4.1-mini")

    demonstrate_parallel_node_execution()
    demonstrate_parallel_search_with_llm(azure_chat_llm=azure_llm)
