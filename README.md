# PaperLens-Agent

Structure-Aware Academic Paper Analyzer — RAG system that understands paper sections.

## Features

- PDF structure detection (Abstract, Introduction, Method, etc.)
- Structure-aware chunking (not naive token splitting)
- Query routing agent (summary / section / qa)
- Answers with source attribution (section + page)
- Interactive Streamlit UI

## Tech Stack

`streamlit` · `pymupdf` · `sentence-transformers` · `chromadb` · `openai` · `numpy` · `pydantic`

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in your OPENAI_API_KEY
```

## Run

```bash
streamlit run ui/app.py
```

## Project Structure

```
parser/     # PDF structure detection + chunking
rag/        # Embedding, vector store, RAG pipeline
agent/      # Query routing
ui/         # Streamlit app
models/     # Pydantic schemas
tests/      # Test scripts
data/       # Sample PDFs
```
