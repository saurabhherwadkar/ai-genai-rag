# AI GenAI RAG Pipeline

An educational implementation of a complete **Retrieval Augmented Generation (RAG)** pipeline in Python. This project demonstrates both the **document ingestion** side and the **query retrieval** side of RAG, using an in-memory vector database (ChromaDB) and sentence-transformers for embeddings.

## What is RAG?

RAG (Retrieval Augmented Generation) is an AI architecture that enhances language model responses by first retrieving relevant information from a knowledge base, then using that context to generate accurate, grounded answers. This project implements both halves:

1. **Ingestion Pipeline**: Load documents → Split into chunks → Generate embeddings → Store in vector database
2. **Query Pipeline**: Process query → Retrieve similar chunks → Rerank results → Generate response

## Project Structure

```
ai-genai-rag/
├── config/                          # Configuration files
│   ├── settings.yaml                # Base configuration (all parameters)
│   ├── settings.dev.yaml            # Development environment overrides
│   ├── settings.prod.yaml           # Production environment overrides
│   └── logging.yaml                 # Logging levels and handlers
├── src/rag_pipeline/                # Main source package
│   ├── main.py                      # CLI entry point
│   ├── config/                      # Settings loader and logging setup
│   │   ├── settings_loader.py       # Hierarchical YAML + env var config
│   │   └── logging_config.py        # Logging initialization from YAML
│   ├── ingestion/                   # Document ingestion components
│   │   ├── document_loader.py       # Load .txt/.md/.pdf/.json files
│   │   ├── text_chunker.py          # Split text with overlap
│   │   ├── embedding_generator.py   # Generate vector embeddings
│   │   └── ingestion_pipeline.py    # Orchestrate ingestion flow
│   ├── vectorstore/                 # Vector database management
│   │   └── vector_store_manager.py  # ChromaDB operations
│   ├── query/                       # Query processing components
│   │   ├── query_processor.py       # Clean and validate queries
│   │   ├── retriever.py             # Similarity search
│   │   ├── reranker.py              # Result reranking
│   │   └── query_pipeline.py        # Orchestrate query flow
│   ├── generation/                  # Response generation
│   │   ├── template_engine.py       # Template variable substitution
│   │   ├── prompt_builder.py        # RAG prompt construction
│   │   └── response_generator.py    # Assemble final response
│   ├── models/                      # Data classes
│   │   ├── document.py              # Document model
│   │   ├── chunk.py                 # Chunk model
│   │   ├── query_result.py          # Query result models
│   │   └── pipeline_config.py       # Configuration models
│   ├── security/                    # Security utilities
│   │   ├── input_sanitizer.py       # Input validation and sanitization
│   │   └── secrets_manager.py       # Environment-based secrets
│   └── utils/                       # Shared utilities
│       ├── exceptions.py            # Custom exception hierarchy
│       ├── text_utils.py            # Text processing helpers
│       └── metrics.py               # Performance measurement
├── tests/                           # Test suite
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/                        # Unit tests for each component
│   └── integration/                 # End-to-end pipeline tests
├── sample_data/                     # Demo documents for ingestion
├── pyproject.toml                   # Dependencies and tool configuration
├── .env.example                     # Environment variable template
└── .gitignore                       # Git ignore patterns
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| chromadb | >=0.5.23 | In-memory vector database |
| sentence-transformers | >=3.4.1 | Text embedding generation |
| pyyaml | >=6.0.2 | YAML configuration parsing |
| python-dotenv | >=1.0.1 | Environment variable loading |
| pypdf | >=5.1.0 | PDF document parsing |
| pytest | >=8.3.4 | Test framework (dev) |
| pytest-cov | >=6.0.0 | Test coverage reporting (dev) |
| pytest-mock | >=3.14.0 | Mocking utilities (dev) |
| ruff | >=0.8.6 | Linting and formatting (dev) |

## Deployment / Setup

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-genai-rag
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up environment variables (optional):**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the Application

**Ingest documents:**
```bash
rag-pipeline ingest sample_data/sample_articles/
```

**Query the knowledge base:**
```bash
rag-pipeline query "What is retrieval augmented generation?"
```

**Interactive mode:**
```bash
rag-pipeline interactive
```

**View statistics:**
```bash
rag-pipeline stats
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest -v
```

### Linting

```bash
# Check for linting issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

## Configuration

All configurable parameters are in `config/settings.yaml`. Environment-specific overrides go in `config/settings.dev.yaml` or `config/settings.prod.yaml`.

### Changing Log Level

Edit `config/logging.yaml` and change the `level` field under `handlers.console`:

```yaml
handlers:
  console:
    level: DEBUG  # Change to: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Environment Variables

Environment variables prefixed with `RAG_` override configuration:

| Variable | Description |
|----------|-------------|
| `RAG_ENVIRONMENT` | Active environment (development/production) |
| `RAG_LOG_LEVEL` | Override log level |
| `RAG_PERSIST_DIRECTORY` | Vector store persistence path |

## Architecture

### End-to-End Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                                    │
│                                                                                 │
│  ┌──────────┐    ┌────────────────┐    ┌─────────────┐    ┌────────────────┐   │
│  │   File   │    │  DocumentLoader │    │ TextChunker │    │   Embedding    │   │
│  │  System  │───▶│                │───▶│             │───▶│   Generator    │   │
│  │          │    │ .txt .md .pdf  │    │ Split text  │    │                │   │
│  │ /data/*  │    │ .json          │    │ with overlap│    │ sentence-      │   │
│  └──────────┘    └────────────────┘    └─────────────┘    │ transformers   │   │
│                                                            └───────┬────────┘   │
│                                                                    │            │
│                                                                    ▼            │
│                                                     ┌──────────────────────┐    │
│                                                     │  VectorStoreManager  │    │
│                                                     │                      │    │
│                                                     │  ChromaDB            │    │
│                                                     │  (cosine similarity) │    │
│                                                     └──────────┬───────────┘    │
│                                                                │               │
└────────────────────────────────────────────────────────────────┼────────────────┘
                                                                 │
                          ┌──────────────────────────────────────┘
                          │  Stored: chunk_id, embedding,
                          │  content, metadata
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PIPELINE                                        │
│                                                                                 │
│  ┌──────────┐    ┌────────────────┐    ┌─────────────────────────────────────┐  │
│  │  User    │    │ InputSanitizer │    │         QueryProcessor              │  │
│  │  Query   │───▶│                │───▶│                                     │  │
│  │          │    │ Security check │    │ 1. Normalize whitespace             │  │
│  │ "What is │    │ Prompt inject. │    │ 2. Remove special chars             │  │
│  │  RAG?"   │    │ detection      │    │ 3. Validate length                  │  │
│  └──────────┘    └────────────────┘    └──────────────────┬──────────────────┘  │
│                                                           │                     │
│                                                           ▼                     │
│                  ┌─────────────────────────────────────────────────────────┐     │
│                  │                    Retriever                            │     │
│                  │                                                         │     │
│                  │  1. Embed query (sentence-transformers)                 │     │
│                  │  2. Similarity search (ChromaDB) ──── Semantic Search   │     │
│                  │  3. Keyword search (ChromaDB) ─────── Syntactic Search  │     │
│                  │  4. RRF (Reciprocal Rank Fusion) to merge results       │     │
│                  │  5. Filter by similarity threshold                      │     │
│                  │  6. Deduplicate overlapping chunks                      │     │
│                  │                                                         │     │
│                  └────────────────────────────┬────────────────────────────┘     │
│                                               │                                 │
│                                               ▼                                 │
│                  ┌─────────────────────────────────────────────────────────┐     │
│                  │                    Reranker                             │     │
│                  │                                                         │     │
│                  │  Scoring signals (weighted combination):                │     │
│                  │    • Similarity score ──────── 60%                      │     │
│                  │    • Keyword overlap ──────── 30%                       │     │
│                  │    • Position bonus ───────── 10%                       │     │
│                  │                                                         │     │
│                  └────────────────────────────┬────────────────────────────┘     │
│                                               │                                 │
│                                               ▼                                 │
│                  ┌─────────────────────────────────────────────────────────┐     │
│                  │               ResponseGenerator                         │     │
│                  │                                                         │     │
│                  │  1. Build context string from ranked chunks             │     │
│                  │  2. PromptBuilder: construct RAG prompt                 │     │
│                  │  3. TemplateEngine: render response                     │     │
│                  │  4. Format source attributions                          │     │
│                  │                                                         │     │
│                  └────────────────────────────┬────────────────────────────┘     │
│                                               │                                 │
│                                               ▼                                 │
│                              ┌─────────────────────────────┐                    │
│                              │       QueryResponse         │                    │
│                              │                             │                    │
│                              │  • Generated answer         │                    │
│                              │  • Source attributions       │                    │
│                              │  • Retrieval results        │                    │
│                              │  • Similarity scores        │                    │
│                              └─────────────────────────────┘                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

- **Single Responsibility**: Each class handles one specific task
- **Dependency Injection**: Components receive dependencies via constructors
- **Immutable Data Models**: Document and Chunk use frozen dataclasses
- **Fail Fast**: Validation at boundaries with descriptive error messages
- **Defense in Depth**: Input sanitization, path traversal protection, prompt injection detection

## License

MIT
