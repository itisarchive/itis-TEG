# Introduction to Generative AI — Getting Started with LLMs

Welcome to your first hands-on experience with Large Language Models!
This introductory module provides the foundation for understanding how LLMs work
and how to interact with them programmatically via **Azure OpenAI**.

## 🎯 Learning Objectives

By completing this module, you will:

- Understand the fundamental concepts of LLM interaction
- Learn how system prompts shape AI behavior
- Master parameter tuning (temperature, tokens, top_p)
- Build your first AI-powered web application
- Establish best practices for API usage and cost management

## 📚 Module Content

### 1. LLM Basics (`1_llm_basics.py`)

**🚀 Your first interactive journey with LLMs**

A comprehensive script that teaches core concepts through practical examples:

- **System Prompts** — how to give AI a personality and role
- **Temperature Control** — balancing creativity (high) vs consistency (low)
- **Token Management** — understanding cost and response length
- **Parameter Combinations** — advanced configurations for different use cases
- **Real-World Scenarios** — email assistant, learning tutor, data analyst

Key takeaways:

| Parameter             | Low values                | High values              |
|-----------------------|---------------------------|--------------------------|
| Temperature (0.0–2.0) | Deterministic, repeatable | Creative, diverse        |
| max_completion_tokens | Concise, cheaper          | Detailed, more expensive |
| top_p                 | Focused vocabulary        | Broader word choice      |

### 2. Your Own Chatbot (`2_Your_own_chatbot.py`)

**🤖 Build a complete web application**

A fully functional Streamlit chatbot featuring:

- **Streamlit Web Interface** — professional web app in minutes
- **Custom Personality** — hip-hop academic teacher (easily customizable via `SYSTEM_PROMPT`)
- **Model Selection** — switch between deployed Azure models from the sidebar
- **Session Management** — conversation memory with clear-history support
- **Streaming Responses** — real-time token-by-token output
- **Error Handling** — environment variable validation and API error messages

### 3. Notebook (`notebook.ipynb`)

The same material as `1_llm_basics.py` in an interactive Jupyter notebook format,
preceded by organizational notes for the course (tools, grading, schedule).

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Azure OpenAI resource with deployed models (
  see [Azure AI Foundry docs](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/create-resource?pivots=web-portal))

### Setup

```bash
# 1. Navigate to the module
cd "src/1. Intro"

# 2. Install dependencies
uv sync

# 3. Create your .env from the template
cp .env.example .env
# Then fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and OPENAI_API_VERSION
```

### Running the Examples

```bash
# Interactive learning script (recommended first step)
uv run python 1_llm_basics.py

# Launch the chatbot web interface
uv run streamlit run 2_Your_own_chatbot.py
# Open http://localhost:8501 in your browser
```

## 🛠️ Dependencies

Defined in `pyproject.toml`:

| Package         | Purpose                   | Min version |
|-----------------|---------------------------|-------------|
| `openai`        | Azure OpenAI SDK          | ≥ 2.21.0    |
| `streamlit`     | Web application framework | ≥ 1.54.0    |
| `python-dotenv` | `.env` file loading       | ≥ 1.2.1     |

## 🔐 Environment Variables

All scripts rely on three variables read automatically by the `AzureOpenAI` SDK
(see `.env.example`):

| Variable                | Description                            |
|-------------------------|----------------------------------------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource URL         |
| `AZURE_OPENAI_API_KEY`  | API key for authentication             |
| `OPENAI_API_VERSION`    | API version, e.g. `2025-04-01-preview` |

> ⚠️ Each model name used in code (e.g. `gpt-5-nano`, `gpt-4.1-mini`) must have
> a matching **deployment** in Azure AI Foundry.

## 🎓 Learning Path

1. **Read this README** — understand the module goals
2. **Set up your environment** — `.env` and dependencies
3. **Run `1_llm_basics.py`** — interactive learning experience
4. **Try the chatbot** — see a complete application
5. **Experiment** — modify system prompts, temperature, token limits
6. **Design your own scenario** — apply the concepts to a real problem

## 💡 Tips

- Never commit `.env` to version control — it contains your API key.
- Start with cheaper models (`gpt-4o-mini`, `gpt-4.1-nano`) during experimentation.
- Set `max_completion_tokens` to control both cost and response style.
- Be specific in system prompts — vague instructions produce vague answers.
- Run temperature demos multiple times to observe variance.

## 🚀 Next Steps

After mastering this module, continue with:

1. **Module 2 (Models)** — embeddings, response objects, multi-provider usage
2. **Module 3 (RAG)** — Retrieval Augmented Generation for knowledge-based AI
3. **Module 4 (Graphs)** — workflow automation with LangGraph
4. **Module 5 (Tools and Agents)** — autonomous AI agents
5. **Module 6 (MCP)** — Model Context Protocol for advanced integrations
