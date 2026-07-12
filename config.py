"""
config.py
---------
keeping all the tuneable knobs in one place so i dont have to hunt through
multiple files when i want to experiment with different settings.
figured this out the hard way after changing chunk sizes in 3 different places lol
"""

import os
from dotenv import load_dotenv

# pull in env vars from .env file if it exists
load_dotenv()

# ---- chunking settings ----
# played around with these values for a while - 500 chars with 80 overlap
# gave the best retrieval results in my testing. too small and you lose
# context, too big and the embeddings get diluted
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# ---- embedding model ----
# MiniLM is surprisingly good for its size (384 dims) - runs fast on cpu
# and the quality is close to the bigger models for this kind of task
EMBEDDING_MODEL_TAG = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ---- vector store ----
# using inner product because we normalize our vectors anyway,
# so it behaves like cosine similarity but FAISS handles it faster
FAISS_METRIC = "inner_product"
TOP_K_RESULTS = 5

# ---- retrieval optimization settings (Requirement 8) ----
# hybrid search blends vector similarity (FAISS) with keyword matching (BM25).
# alpha=0.7 means 70% vector weight, 30% keyword weight.
USE_HYBRID_SEARCH = True
HYBRID_ALPHA = 0.7

# cross-encoder re-ranking re-scores candidate chunks with ms-marco MiniLM.
# disabled by default because it adds ~0.5s latency on CPU, but toggle on for max accuracy.
USE_RERANKING = False


# ---- LLM / generation settings ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = "gemini-flash-latest"

# ---- huggingface dataset ----
# vectara's open ragbench is built from arxiv papers - has queries,
# answers, and full document sections. great for testing because
# it covers text, tables, and even image-based content from real pdfs
HF_DATASET_NAME = "vectara/open_ragbench"
# cap how many docs we pull from HF so it doesnt take forever on first run
HF_MAX_DOCUMENTS = 50

# ---- file paths ----
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_store", "doc_index.faiss")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline_run.log")

# make sure the directories exist
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
