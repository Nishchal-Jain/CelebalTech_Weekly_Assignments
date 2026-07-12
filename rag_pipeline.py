"""
rag_pipeline.py
---------------
this is where everything comes together. the RAGPipeline class ties
all the individual modules (loader, chunker, embedder, vector store,
generator) into a single coherent workflow.

designed it as a class because it needs to hold state (the loaded model,
the built index, the chunk registry) across multiple queries. using
separate functions for each stage made debugging way easier - when
retrieval was returning garbage i could test each step independently.

the pipeline has two main phases:
  1. ingest() - load docs, chunk them, embed them, build index
  2. ask()   - take a question, retrieve context, generate answer
"""

import time
import logging

from document_loader import load_all_documents, pull_hf_question_answers
from text_chunker import chunk_all_documents
from embedding_engine import boot_embedding_model, vectorize_chunks, vectorize_query
from vector_store import build_faiss_index, find_closest_chunks, save_index, load_index
from hybrid_retriever import BM25Searcher, hybrid_search, rerank_chunks
from answer_generator import generate_answer
import config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    end-to-end retrieval-augmented generation pipeline.

    usage is straightforward:
        pipeline = RAGPipeline()
        pipeline.ingest(use_hf=True)
        result = pipeline.ask("what is transfer learning?")
        print(result["answer"])
    """

    def __init__(self):
        # these get populated during ingest()
        self.embedding_model = None
        self.faiss_index = None
        self.bm25_searcher = None
        self.chunk_registry = []  # all chunks with metadata

        # tracking metrics for the final report
        self.metrics = {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_dimensions": 0,
            "chunk_size_setting": config.CHUNK_SIZE,
            "chunk_overlap_setting": config.CHUNK_OVERLAP,
            "embedding_model": config.EMBEDDING_MODEL_TAG,
            "llm_model": config.GEMINI_MODEL_NAME,
            "faiss_index_type": "IndexFlatIP (exact inner product)",
            "top_k": config.TOP_K_RESULTS,
            "ingestion_time_sec": 0,
            "queries_answered": 0,
        }

    def ingest(self, source_dir=None, use_hf=False):
        """
        full ingestion pipeline:
          1. load documents from files and/or huggingface
          2. chunk all the text
          3. embed chunks into vectors
          4. build the FAISS index

        this is the expensive step - embedding takes a while on cpu
        for large document sets. but it only needs to run once.
        """
        ingest_start = time.time()

        if source_dir is None:
            source_dir = config.SAMPLE_DOCS_DIR

        # --- step 1: load documents ---
        logger.info("=" * 60)
        logger.info("STEP 1: Loading documents")
        logger.info("=" * 60)

        documents = load_all_documents(
            source_dir=source_dir,
            use_hf_dataset=use_hf,
            hf_dataset_name=config.HF_DATASET_NAME,
            hf_max_docs=config.HF_MAX_DOCUMENTS
        )

        if not documents:
            logger.error("no documents loaded! check your source directory or hf dataset")
            return False

        self.metrics["total_documents"] = len(documents)

        # --- step 2: chunk the text ---
        logger.info("=" * 60)
        logger.info("STEP 2: Chunking documents")
        logger.info("=" * 60)

        self.chunk_registry = chunk_all_documents(
            documents,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )

        self.metrics["total_chunks"] = len(self.chunk_registry)

        if not self.chunk_registry:
            logger.error("chunking produced zero chunks - something is wrong with the input")
            return False

        # --- step 3: embed chunks ---
        logger.info("=" * 60)
        logger.info("STEP 3: Embedding chunks")
        logger.info("=" * 60)

        self.embedding_model = boot_embedding_model(config.EMBEDDING_MODEL_TAG)
        embeddings_matrix = vectorize_chunks(self.embedding_model, self.chunk_registry)

        self.metrics["embedding_dimensions"] = embeddings_matrix.shape[1] if embeddings_matrix.size > 0 else 0

        # --- step 4: build FAISS index ---
        logger.info("=" * 60)
        logger.info("STEP 4: Building vector index")
        logger.info("=" * 60)

        self.faiss_index = build_faiss_index(embeddings_matrix)

        # save the index for potential reuse
        save_index(self.faiss_index, config.FAISS_INDEX_PATH)

        # --- step 5: build BM25 keyword index for hybrid search ---
        logger.info("=" * 60)
        logger.info("STEP 5: Building BM25 keyword index")
        logger.info("=" * 60)
        self.bm25_searcher = BM25Searcher(self.chunk_registry)

        ingest_time = time.time() - ingest_start
        self.metrics["ingestion_time_sec"] = round(ingest_time, 2)

        logger.info(f"ingestion complete in {ingest_time:.1f}s")
        return True

    def ask(self, question):
        """
        full query pipeline:
          1. embed the question
          2. search FAISS for relevant chunks
          3. generate answer using retrieved context

        returns a dict with the answer, retrieved chunks, and metadata.
        """
        if self.faiss_index is None or self.embedding_model is None:
            return {
                "answer": "[ERROR] pipeline not initialized - run ingest() first",
                "retrieved_chunks": [],
                "question": question
            }

        # embed the question into a vector
        query_vec = vectorize_query(self.embedding_model, question)

        # find the most relevant chunks (hybrid vector + keyword or pure vector)
        if config.USE_HYBRID_SEARCH and self.bm25_searcher is not None:
            top_chunks = hybrid_search(
                faiss_index=self.faiss_index,
                bm25_searcher=self.bm25_searcher,
                query_vec=query_vec,
                embedding_model=self.embedding_model,
                query_text=question,
                all_chunks=self.chunk_registry,
                top_k=config.TOP_K_RESULTS,
                alpha=config.HYBRID_ALPHA
            )
        else:
            top_chunks = find_closest_chunks(
                self.faiss_index,
                query_vec,
                self.chunk_registry,
                top_k=config.TOP_K_RESULTS
            )

        # optional cross-encoder re-ranking
        if config.USE_RERANKING:
            top_chunks = rerank_chunks(question, top_chunks, top_k=config.TOP_K_RESULTS)

        # generate an answer using the retrieved context
        gen_result = generate_answer(
            question=question,
            context_chunks=top_chunks,
            api_key=config.GEMINI_API_KEY,
            model_name=config.GEMINI_MODEL_NAME
        )

        self.metrics["queries_answered"] += 1

        return {
            "answer": gen_result["answer"],
            "question": question,
            "retrieved_chunks": top_chunks,
            "prompt_length": gen_result["prompt_length"],
            "context_chunks_used": gen_result["context_chunks_used"],
            "model_used": gen_result["model"]
        }

    def run_validation(self, test_questions=None):
        """
        runs a batch of questions through the pipeline and logs results.
        if test_questions isnt provided, tries to pull q&a pairs from
        the huggingface dataset for validation.

        returns a list of result dicts with questions, answers, and scores.
        """
        validation_results = []

        if test_questions is None:
            # try to get questions from the hf dataset
            qa_pairs = pull_hf_question_answers(config.HF_DATASET_NAME)
            if qa_pairs:
                test_questions = [qa["question"] for qa in qa_pairs[:10]]
                logger.info(f"using {len(test_questions)} questions from hf dataset for validation")
            else:
                # fallback to generic questions about the sample doc
                test_questions = [
                    "What is supervised learning?",
                    "How does overfitting affect model performance?",
                    "What is the bias-variance tradeoff?",
                    "Explain transfer learning and why it is useful.",
                    "What are common evaluation metrics for classification?",
                ]
                logger.info("using fallback test questions for validation")

        logger.info("=" * 60)
        logger.info(f"VALIDATION: Running {len(test_questions)} test queries")
        logger.info("=" * 60)

        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- Query {i}/{len(test_questions)} ---")
            logger.info(f"Q: {question}")

            result = self.ask(question)
            validation_results.append(result)

            # log the top retrieval scores
            if result["retrieved_chunks"]:
                top_score = result["retrieved_chunks"][0]["relevance_score"]
                avg_score = sum(c["relevance_score"] for c in result["retrieved_chunks"]) / len(result["retrieved_chunks"])
                logger.info(f"retrieval: top_score={top_score:.4f}, avg_score={avg_score:.4f}")

            # truncate answer for log readability
            answer_preview = result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"]
            logger.info(f"A: {answer_preview}")

        return validation_results

    def get_system_report(self):
        """
        builds a formatted report of the system configuration and
        performance metrics. this is one of the required deliverables -
        shows what settings we used and how the system performed.
        """
        report_lines = [
            "",
            "=" * 65,
            "           SYSTEM METRICS REPORT",
            "=" * 65,
            "",
            "--- Document Ingestion ---",
            f"  Documents loaded:        {self.metrics['total_documents']}",
            f"  Total chunks created:    {self.metrics['total_chunks']}",
            f"  Ingestion time:          {self.metrics['ingestion_time_sec']}s",
            "",
            "--- Chunking Profile ---",
            f"  Chunk size (chars):      {self.metrics['chunk_size_setting']}",
            f"  Chunk overlap (chars):   {self.metrics['chunk_overlap_setting']}",
            f"  Split strategy:          Recursive (paragraph > sentence > word)",
            "",
            "--- Embedding Configuration ---",
            f"  Model:                   {self.metrics['embedding_model']}",
            f"  Vector dimensions:       {self.metrics['embedding_dimensions']}",
            f"  Normalization:           L2 (for cosine similarity via IP)",
            "",
            "--- Vector & Keyword Retrieval ---",
            f"  Engine:                  FAISS (faiss-cpu) + BM25",
            f"  Index type:              {self.metrics['faiss_index_type']}",
            f"  Vectors stored:          {self.metrics['total_chunks']}",
            f"  Hybrid search:           {'Enabled (alpha=' + str(config.HYBRID_ALPHA) + ')' if config.USE_HYBRID_SEARCH else 'Disabled (pure vector)'}",
            f"  Re-ranking layer:        {'Enabled (ms-marco-MiniLM)' if config.USE_RERANKING else 'Disabled'}",
            f"  Top-K retrieval:         {self.metrics['top_k']}",
            "",
            "--- Language Model ---",
            f"  Provider:                Google Gemini (free tier)",
            f"  Model:                   {self.metrics['llm_model']}",
            f"  Queries answered:        {self.metrics['queries_answered']}",
            "",
            "=" * 65,
        ]

        return "\n".join(report_lines)
