# 🔀 Graphs — LangGraph Educational Examples

Educational examples demonstrating graph-based workflows using **LangGraph** with **Azure OpenAI**.

## Project Structure

| File                      | Description                                                      |
|---------------------------|------------------------------------------------------------------|
| `1_simple graph.py`       | Simple graph with conditional routing + LLM as a graph node      |
| `2_parallel_processes.py` | Parallel node execution with Tavily & Wikipedia integration      |
| `3_map_reduce.py`         | Map-reduce pattern: parallel joke generation with best selection |
| `.env.example`            | Template for required environment variables                      |
| `pyproject.toml`          | Project dependencies and configuration                           |

## Setup

### Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) package manager

### Installation

```bash
uv sync
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

| Variable                | Description                                               |
|-------------------------|-----------------------------------------------------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint                            |
| `AZURE_OPENAI_API_KEY`  | Azure OpenAI API key                                      |
| `OPENAI_API_VERSION`    | API version (e.g. `2025-04-01-preview`)                   |
| `TAVILY_API_KEY`        | Tavily search API key (used in `2_parallel_processes.py`) |
| `LANGSMITH_API_KEY`     | *(optional)* LangSmith tracing key                        |

## Usage

```bash
uv run python "1_simple graph.py"
uv run python "2_parallel_processes.py"
uv run python "3_map_reduce.py"
```

## Learning Path

### 1. Simple Graph (`1_simple graph.py`)

- **Graph state** shared between nodes via `TypedDict`
- **Nodes** as functions modifying state
- **Direct edges** and **conditional edges** controlling flow
- **LLM as a node** inside a LangGraph workflow

### 2. Parallel Processing (`2_parallel_processes.py`)

- **Parallel node execution** — nodes B and C run concurrently after A
- **State aggregation** with `operator.add`
- **Multi-source search** — Tavily (internet) + Wikipedia in parallel
- **LLM synthesis** — combining gathered context into a final answer

### 3. Map-Reduce Pattern (`3_map_reduce.py`)

- **Dynamic fan-out** using `Send()` for variable node counts
- **MAP phase** — parallel joke generation on subtopics
- **REDUCE phase** — best joke selection from all candidates
- **Structured LLM output** via Pydantic models
