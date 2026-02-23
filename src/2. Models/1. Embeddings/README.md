# Text Embeddings — Interactive Educational Demo

Welcome to your hands-on exploration of text embeddings!
This module provides a comprehensive journey through embedding fundamentals —
from basic vector properties to advanced vector arithmetic — using **Azure OpenAI**.

## 🎯 Learning Objectives

By completing this module, you will:

- Understand what embeddings are and how they encode semantic meaning
- Compare similarity metrics: cosine similarity, Euclidean distance, dot product
- Discover counter-intuitive patterns (why _cat–dog > cat–kitten_)
- See how context dramatically changes word relationships
- Visualize semantic clusters with PCA and t-SNE
- Master the famous _king − man + woman ≈ queen_ analogy
- Connect theory to real AI applications (search, recommendations, chatbots)

## 📚 Module Content

### 1. Embeddings Basics (`1. embeddings_basics.py`)

**🚀 Comprehensive script covering seven demos**

A standalone Python script that teaches every concept through practical examples:

| Demo | Topic                    | Key Takeaway                                           |
|------|--------------------------|--------------------------------------------------------|
| 1    | Basic Embeddings         | 1536-dimensional vectors, unit magnitude, statistics   |
| 2    | Word Similarity Analysis | Three metrics compared side-by-side                    |
| 3    | The Cat-Dog Mystery      | Statistical co-occurrence ≠ taxonomic logic            |
| 4    | Context Matters          | Phrases can reverse similarity rankings                |
| 5    | Semantic Clusters        | PCA visualization + intra/inter-cluster analysis       |
| 6    | Similarity Heatmap       | Full relationship matrix with most/least similar pairs |
| 7    | Vector Arithmetic        | King–queen analogy, gender-role & conceptual analogies |

### 2. Notebook (`notebook_1.ipynb`)

The same material as `1. embeddings_basics.py` in an interactive Jupyter notebook format,
extended with a **t-SNE visualization** (section 5b) for comparison with PCA.

## 📁 File Structure

```
1. Embeddings/
├── 1. embeddings_basics.py   # Main educational script (7 demos)
├── notebook_1.ipynb          # Interactive notebook (same content + t-SNE)
├── README.md                 # This documentation
├── pyproject.toml            # Dependencies and project configuration
├── .env.example              # Template for environment variables
├── .env                      # Your API credentials (not committed)
├── semantic_clusters.png     # Generated PCA visualization
├── similarity_heatmap.png    # Generated similarity heatmap
└── uv.lock                   # Dependency lock file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Azure OpenAI resource with a deployed `text-embedding-3-small` model
  (see [Azure AI Foundry docs](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/create-resource?pivots=web-portal))

### Setup

```bash
# 1. Navigate to the module
cd "src/2. Models/1. Embeddings"

# 2. Install dependencies
uv sync

# 3. Create your .env from the template
cp .env.example .env
# Then fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and OPENAI_API_VERSION
```

### Running the Examples

```bash
# Run the comprehensive demo script
uv run python "1. embeddings_basics.py"
```

For the notebook, open `notebook_1.ipynb` in your IDE or Jupyter and run cells sequentially.

## 🛠️ Dependencies

Defined in `pyproject.toml`:

| Package         | Purpose                        | Min version |
|-----------------|--------------------------------|-------------|
| `openai`        | Azure OpenAI SDK               | ≥ 2.21.0    |
| `numpy`         | Vector operations              | ≥ 2.4.2     |
| `scikit-learn`  | Cosine similarity, PCA, t-SNE  | ≥ 1.8.0     |
| `matplotlib`    | Chart rendering                | ≥ 3.10.8    |
| `seaborn`       | Heatmap visualization          | ≥ 0.13.2    |
| `python-dotenv` | `.env` file loading            | ≥ 1.2.1     |

## 🔐 Environment Variables

All scripts rely on three variables read automatically by the `AzureOpenAI` SDK
(see `.env.example`):

| Variable                | Description                            |
|-------------------------|----------------------------------------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource URL         |
| `AZURE_OPENAI_API_KEY`  | API key for authentication             |
| `OPENAI_API_VERSION`    | API version, e.g. `2025-04-01-preview` |

> ⚠️ The model name `text-embedding-3-small` used in code must have
> a matching **deployment** in Azure AI Foundry.

## 🎓 Learning Path

1. **Read this README** — understand the module goals
2. **Set up your environment** — `.env` and dependencies
3. **Run `1. embeddings_basics.py`** — watch all seven demos execute
4. **Open `notebook_1.ipynb`** — step through cells interactively, experiment
5. **Study the generated PNGs** — `semantic_clusters.png` and `similarity_heatmap.png`
6. **Experiment** — try your own words, analogies, and context templates

## 💡 Tips

- Never commit `.env` to version control — it contains your API key.
- Both the script and notebook use an **embedding cache** — re-running is cheap.
- The script saves charts to PNG files; the notebook renders them inline.
- Try adding your own word groups to the clustering demo for deeper insight.
- Run analogy tests with domain-specific terms to see where embeddings excel (and fail).

## 🚀 Next Steps

After mastering this module, continue with:

1. **Module 2.2 (LLMs)** — response objects, multi-provider usage with LangChain
2. **Module 3 (RAG)** — Retrieval Augmented Generation for knowledge-based AI
3. **Module 4 (Graphs)** — workflow automation with LangGraph
4. **Module 5 (Tools and Agents)** — autonomous AI agents
5. **Module 6 (MCP)** — Model Context Protocol for advanced integrations
