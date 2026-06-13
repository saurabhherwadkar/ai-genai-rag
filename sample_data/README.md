# Sample Data

This directory contains sample documents for demonstrating the RAG pipeline's ingestion and query capabilities.

## Files

- `sample_articles/article_1.txt` - Introduction to Artificial Intelligence
- `sample_articles/article_2.txt` - Understanding Natural Language Processing

## Usage

Ingest these sample documents using:

```bash
rag-pipeline ingest sample_data/sample_articles/
```

Then query the knowledge base:

```bash
rag-pipeline query "What is retrieval augmented generation?"
```
