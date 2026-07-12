"""
run_pipeline.py
---------------
main entry point for the RAG document question answering system.

this script demonstrates the full pipeline:
  1. ingests documents (local files + optional huggingface dataset)
  2. asks a few sample questions and prints grounded answers
  3. runs validation with test questions
  4. prints the system metrics report

run it with:
    python run_pipeline.py

by default it uses the sample docs in sample_docs/ folder.
to also pull from the huggingface dataset, use --hf flag:
    python run_pipeline.py --hf

make sure you have your GEMINI_API_KEY set in a .env file
(copy .env.example to .env and paste your key).
"""

import sys
import logging
import argparse
from rag_pipeline import RAGPipeline
import config


def setup_logging():
    """
    sets up logging to both console and file. the file log captures
    everything (DEBUG level) while console only shows INFO and above
    so its not too noisy during normal runs.
    """
    # make sure log directory exists (config.py handles this but just in case)
    import os
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # root logger config
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # console handler - show info+ level messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(levelname)s | %(message)s")
    console_handler.setFormatter(console_fmt)

    # file handler - capture everything for debugging later
    file_handler = logging.FileHandler(config.LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_fmt)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info(f"logging to console and {config.LOG_FILE}")


def parse_args():
    """
    command line arguments. kept it simple - just a flag to control
    whether we pull from huggingface or only use local docs.
    """
    parser = argparse.ArgumentParser(
        description="RAG Document Question Answering System"
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="also load documents from the HuggingFace dataset (vectara/open_ragbench)"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="custom directory containing PDF/text documents (default: sample_docs/)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="path to a single document (e.g. Resume_final.pdf) to ingest instead of a directory"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="ask a single question instead of running the full demo"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="enable cross-encoder re-ranking for higher retrieval precision (adds ~0.5s per query)"
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="disable hybrid BM25+vector search and use pure FAISS vector search instead"
    )
    return parser.parse_args()


def print_divider(title=""):
    """just a visual separator for console output"""
    if title:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    else:
        print(f"\n{'-'*65}")


def run_demo_questions(pipeline):
    """
    runs a set of demo questions to showcase the pipeline.
    these are designed to test different retrieval scenarios.
    """
    demo_questions = [
        "What is supervised learning and how does it differ from unsupervised learning?",
        "What techniques can prevent overfitting in machine learning models?",
        "Explain the bias-variance tradeoff.",
        "What is transfer learning and why is it important?",
        "How are machine learning models evaluated for classification tasks?",
    ]

    print_divider("DEMO: Sample Question Answering")

    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'─'*50}")
        print(f"  Question {i}: {question}")
        print(f"{'─'*50}")

        result = pipeline.ask(question)

        # show the answer
        print(f"\n  Answer:\n  {result['answer']}")

        # show retrieval details
        print(f"\n  Retrieved {result['context_chunks_used']} context chunks:")
        for chunk in result["retrieved_chunks"][:3]:  # show top 3
            score = chunk["relevance_score"]
            source = chunk["source"]
            preview = chunk["chunk_text"][:100] + "..." if len(chunk["chunk_text"]) > 100 else chunk["chunk_text"]
            print(f"    [{score:.3f}] ({source}) {preview}")

        print()


def main():
    setup_logging()
    args = parse_args()

    logger = logging.getLogger(__name__)

    if args.rerank:
        config.USE_RERANKING = True
    if args.no_hybrid:
        config.USE_HYBRID_SEARCH = False

    print_divider("RAG Document QA System - Starting Up")
    print(f"  Embedding model:  {config.EMBEDDING_MODEL_TAG}")
    print(f"  LLM model:        {config.GEMINI_MODEL_NAME}")
    print(f"  Chunk size:       {config.CHUNK_SIZE} chars (overlap: {config.CHUNK_OVERLAP})")
    print(f"  Hybrid search:    {'enabled (BM25 + FAISS)' if config.USE_HYBRID_SEARCH else 'disabled (pure FAISS)'}")
    print(f"  Re-ranking:       {'enabled (ms-marco-MiniLM cross-encoder)' if config.USE_RERANKING else 'disabled'}")
    print(f"  HuggingFace data: {'enabled' if args.hf else 'disabled (use --hf to enable)'}")

    # check for api key early so user isnt surprised after waiting for ingestion
    if not config.GEMINI_API_KEY:
        print("\n  ⚠️  WARNING: No GEMINI_API_KEY found!")
        print("  The pipeline will run but answers will show an error message.")
        print("  Get a free key from https://aistudio.google.com/")
        print("  Then create a .env file with: GEMINI_API_KEY=your_key_here")

    # --- initialize and run the pipeline ---
    pipeline = RAGPipeline()

    print_divider("Phase 1: Document Ingestion")

    if args.file:
        docs_source = args.file
        print(f"  Target document:  {args.file}")
    elif args.docs_dir:
        docs_source = args.docs_dir
        print(f"  Target directory: {args.docs_dir}")
    else:
        docs_source = config.SAMPLE_DOCS_DIR
        print(f"  Target directory: {config.SAMPLE_DOCS_DIR}")

    ingestion_ok = pipeline.ingest(
        source_dir=docs_source,
        use_hf=args.hf
    )

    if not ingestion_ok:
        print("\n  ❌ Ingestion failed! Check the logs for details.")
        print(f"  Log file: {config.LOG_FILE}")
        sys.exit(1)

    print(f"\n  ✅ Ingestion complete!")
    print(f"  Documents: {pipeline.metrics['total_documents']}")
    print(f"  Chunks:    {pipeline.metrics['total_chunks']}")
    print(f"  Time:      {pipeline.metrics['ingestion_time_sec']}s")

    # --- handle single question mode ---
    if args.question:
        print_divider("Single Question Mode")
        print(f"\n  Q: {args.question}\n")

        result = pipeline.ask(args.question)
        print(f"  A: {result['answer']}")

        print(f"\n  Retrieved chunks:")
        for chunk in result["retrieved_chunks"]:
            score = chunk["relevance_score"]
            source = chunk["source"]
            print(f"    [{score:.3f}] ({source})")
    else:
        # --- run demo questions ---
        run_demo_questions(pipeline)

        # --- run validation ---
        print_divider("Phase 2: Validation Run")
        validation_results = pipeline.run_validation()

        # summarize validation
        print(f"\n  Validation complete: {len(validation_results)} questions answered")
        if validation_results:
            # calculate average retrieval scores
            all_top_scores = []
            for r in validation_results:
                if r["retrieved_chunks"]:
                    all_top_scores.append(r["retrieved_chunks"][0]["relevance_score"])

            if all_top_scores:
                avg_top = sum(all_top_scores) / len(all_top_scores)
                min_top = min(all_top_scores)
                max_top = max(all_top_scores)
                print(f"  Retrieval scores (top-1): avg={avg_top:.4f}, min={min_top:.4f}, max={max_top:.4f}")

    # --- print system metrics ---
    report = pipeline.get_system_report()
    print(report)

    print(f"\n  📄 Full logs saved to: {config.LOG_FILE}")
    print(f"  💾 FAISS index saved to: {config.FAISS_INDEX_PATH}")
    print()


if __name__ == "__main__":
    main()
