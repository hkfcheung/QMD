# QMD Local Document Search

**Query → Model → Documents** — a local, offline document search tool.

Search your local text files using natural-language queries with BM25 lexical search, embedding-based semantic search, hybrid retrieval with reranking, and LLM-powered answer synthesis (RAG).

## Quick Start

```bash
# 1. Create and activate virtual environment
cd qmd_search
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 2. Install in development mode
pip install -e ".[dev,bm25,embed,llm]"

# 3. Generate sample documents
qmd make-sample --out data/samples

# 4. Ingest and index
qmd ingest --input data/samples --out data/index

# 5. Search
qmd query "database migration"

# 6. View index info
qmd info
```

## Features

- **Hybrid search**: BM25 lexical + semantic embedding search fused with Reciprocal Rank Fusion
- **Fuzzy matching**: Handles typos and OCR errors
- **LLM synthesis**: Optional RAG-powered answer generation (OpenAI / Anthropic)
- **Incremental indexing**: Only reprocesses files whose content has changed
- **Local-first**: All indexing and search runs locally; LLM calls are optional

## Stage Progress

- [x] **Stage 0**: Repo scaffolding
- [x] **Stage 1**: Ingestion + chunking + minimal cleaning
- [x] **Stage 2**: BM25 lexical search with fuzzy matching
- [x] **Stage 3**: Embedding retrieval (local model)
- [x] **Stage 4**: Hybrid retrieval + rerank + LLM synthesis
- [ ] **Stage 5**: Local web UI (nice-to-have)

## Project Structure

```
qmd_search/
├── qmd/
│   ├── __init__.py         # Package version
│   ├── cli.py              # Click CLI commands
│   ├── ingest.py           # File discovery, cleaning, chunking
│   ├── index_store.py      # Load/save chunk index
│   ├── sample_data.py      # Synthetic sample data generator
│   ├── search_bm25.py      # BM25 search with fuzzy matching
│   ├── search_embed.py     # Embedding search (sentence-transformers)
│   ├── hybrid.py           # Hybrid retrieval + RRF fusion
│   ├── synthesize.py       # LLM answer synthesis (RAG)
│   ├── eval.py             # Evaluation metrics
│   └── utils.py            # Text cleaning, hashing, tag inference
├── data/                   # Sample data and indexes
├── tests/                  # Unit tests
├── .env                    # API keys (not committed)
├── pyproject.toml          # Project config and dependencies
└── README.md
```

## Running Tests

```bash
pytest -v
```

## Data Model

- **Input**: `.txt` and `.md` files in a folder (nested folders supported)
- **Chunks**: Paragraph-based splitting with configurable max size and overlap
- **Metadata**: file path, name, created/modified time, chunk ID, tags (from filename)
- **Incremental indexing**: Only reprocesses files whose content has changed

## CLI Reference

| Command | Description |
|---------|-------------|
| `qmd make-sample --out <dir>` | Generate sample documents |
| `qmd ingest --input <dir> --out <index_dir>` | Ingest and chunk files |
| `qmd query "<search terms>"` | Search with hybrid ranking + LLM synthesis |
| `qmd query "<terms>" --no-synth` | Search without LLM synthesis |
| `qmd info` | Show index statistics |
| `qmd --version` | Show version |

## Configuration (.env)

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```

Defaults to OpenAI if both keys are set. LLM synthesis is skipped if no keys are configured.
