# Document Question Answering System (RAG)

A Retrieval-Augmented Generation (RAG) pipeline that answers questions grounded in custom documents. Instead of relying only on a language model's internal knowledge, the system retrieves relevant information from ingested documents and uses that context to generate accurate answers.

## What This Project Does

The system takes in documents (PDFs, text files, or data from HuggingFace), breaks them into chunks, converts those chunks into vector embeddings, stores them in a FAISS vector database, and then uses similarity search to find the most relevant chunks for any given question. Those chunks are passed as context to Google Gemini which generates a grounded answer.

This approach is useful for:
- Answering questions about private/domain-specific data
- Reducing LLM hallucinations by grounding responses in actual documents
- Building knowledge assistants for specific document sets

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────────┐
│  Document       │     │  Text        │     │  Embedding         │
│  Loader         │────▶│  Chunker     │────▶│  Engine            │
│  (PDF/TXT/HF)   │     │  (recursive) │     │  (MiniLM-L6-v2)   │
└─────────────────┘     └──────────────┘     └────────┬───────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────────┐
│  Answer         │     │  Context     │     │  FAISS Vector      │
│  Generator      │◀────│  Retrieval   │◀────│  Store             │
│  (Gemini API)   │     │  (top-K)     │     │  (Inner Product)   │
└─────────────────┘     └──────────────┘     └────────────────────┘
```

## Project Structure

```
├── config.py              # central configuration (chunk size, models, paths)
├── document_loader.py     # loads PDFs, text files, HuggingFace datasets
├── text_chunker.py        # recursive overlap-aware text splitting
├── embedding_engine.py    # sentence-transformer vector encoding
├── vector_store.py        # FAISS index management (build, search, save/load)
├── hybrid_retriever.py    # BM25 keyword search, hybrid blending, & cross-encoder re-ranking
├── answer_generator.py    # prompt crafting + Gemini API calls
├── rag_pipeline.py        # orchestrates the full RAG workflow
├── run_pipeline.py        # main entry point with CLI args
├── requirements.txt       # pip dependencies
├── .env.example           # template for API keys
├── sample_docs/           # sample documents for testing
│   └── sample_note.txt    # ML fundamentals reference text
├── faiss_store/           # saved FAISS index (auto-generated)
└── logs/                  # pipeline run logs (auto-generated)
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # on mac/linux
# venv\Scripts\activate    # on windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your Gemini API key

Get a free API key from [Google AI Studio](https://aistudio.google.com/):
1. Sign in with your Google account
2. Click "Get API key"
3. Create a new key

Then create your `.env` file:
```bash
cp .env.example .env
# edit .env and paste your API key
```

## Usage

### Basic run (local documents only)

```bash
python run_pipeline.py
```

This uses the sample document in `sample_docs/` to demonstrate the pipeline.

### With HuggingFace dataset

```bash
python run_pipeline.py --hf
```

This pulls documents from the `vectara/open_ragbench` dataset (arXiv papers) in addition to local documents. The dataset is downloaded and cached on first run.

### Custom document directory

```bash
python run_pipeline.py --docs-dir /path/to/your/pdfs
```

### Ask a single question

```bash
python run_pipeline.py --question "What is the main topic of these documents?"
```

### Enable advanced retrieval (Re-ranking & Hybrid search options)

```bash
# Enable cross-encoder re-ranking (ms-marco-MiniLM) for maximum precision
python run_pipeline.py --rerank

# Test pure FAISS vector search without BM25 hybrid keyword blending
python run_pipeline.py --no-hybrid
```

### Combine options

```bash
python run_pipeline.py --hf --rerank --question "Explain attention mechanisms"
```

## How It Works

### 1. Document Ingestion (`document_loader.py`)
- Reads PDF files using PyPDF2 (page-by-page text extraction)
- Loads plain text / markdown files
- Optionally pulls documents from HuggingFace's `vectara/open_ragbench` dataset via the `datasets` library API

### 2. Text Chunking (`text_chunker.py`)
- Uses a recursive splitting strategy that respects natural text boundaries
- Split hierarchy: paragraphs → sentences → words → hard character split
- Adds configurable overlap between chunks to preserve cross-boundary context
- Default: 500 char chunks with 80 char overlap

### 3. Embedding (`embedding_engine.py`)
- Uses `all-MiniLM-L6-v2` from sentence-transformers (384-dimensional vectors)
- L2-normalizes all vectors so inner product equals cosine similarity
- Batch encoding for efficiency

### 4. Vector & Keyword Storage (`vector_store.py` & `hybrid_retriever.py`)
- FAISS `IndexFlatIP` for exact inner-product similarity search
- Custom BM25 index built from chunk tokens (`hybrid_retriever.py`) for term frequency keyword matching
- Blends vector and keyword scores via `HYBRID_ALPHA` (default 70% vector, 30% keyword)
- Optional cross-encoder re-ranking (`rerank_chunks`) re-scores candidate chunks using `ms-marco-MiniLM-L-6-v2` for high-precision context selection

### 5. Answer Generation (`answer_generator.py`)
- Crafts a structured prompt with retrieved context passages
- Instructs the model to answer only from provided context
- Uses Google Gemini API (free tier)

## HuggingFace Dataset

The project uses `vectara/open_ragbench` which is a RAG benchmark dataset built from arXiv PDF documents. It includes:
- Full document text extracted from research papers
- Query-answer pairs for validation
- Coverage of text, tables, and image-based content

Loading happens through the HuggingFace `datasets` library API — no manual download needed.

## Configuration

All settings are centralized in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Target chunk size in characters |
| `CHUNK_OVERLAP` | 80 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL_TAG` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K_RESULTS` | 5 | Number of chunks to retrieve per query |
| `USE_HYBRID_SEARCH` | `True` | Blends FAISS vector search with BM25 keyword search |
| `HYBRID_ALPHA` | 0.7 | Weight ratio (0.7 = 70% vector, 30% keyword) |
| `USE_RERANKING` | `False` | Toggles cross-encoder re-ranking layer (`--rerank` flag) |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | LLM for answer generation |
| `HF_MAX_DOCUMENTS` | 50 | Max documents to pull from HuggingFace |

## Sample Output

```
=================================================================
  RAG Document QA System - Starting Up
=================================================================
  Embedding model:  all-MiniLM-L6-v2
  LLM model:        gemini-2.0-flash
  Chunk size:       500 chars (overlap: 80)

──────────────────────────────────────────────────
  Question 1: What is supervised learning?
──────────────────────────────────────────────────

  Answer:
  Supervised learning uses labeled data where both inputs and outputs
  are provided during training. The model learns to map inputs to
  outputs and can then predict outcomes for new unseen data. (Passage 1)

  Retrieved 5 context chunks:
    [0.724] (sample_note.txt) There are three main categories...
    [0.651] (sample_note.txt) Machine learning is a branch...
    [0.583] (sample_note.txt) Neural networks form the backbone...
```

## Experiments and Improvements

Things I tried and what worked:
- **Chunk size**: 500 chars was the sweet spot. Smaller (200) had too little context, larger (1000) diluted embeddings
- **Overlap**: 80 chars helped catch info spanning chunk boundaries
- **Vector normalization**: Normalizing embeddings and using inner product instead of L2 distance gave slightly better retrieval ranking
- Potential improvements: hybrid search (BM25 + vector), cross-encoder re-ranking, larger embedding models

## Key Learnings

- RAG dramatically reduces hallucination compared to direct LLM queries
- Chunking strategy has a huge impact on retrieval quality
- Embedding model choice matters less than good chunking (at this scale)
- FAISS is fast enough for exact search up to ~100K vectors
- Prompt engineering for the generation step is critical — the model needs clear instructions to stick to the provided context

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| PDF parsing | PyPDF2 |
| Dataset | HuggingFace `datasets` (vectara/open_ragbench) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS (faiss-cpu) |
| LLM | Google Gemini API (gemini-2.0-flash) |
| Config | python-dotenv |
