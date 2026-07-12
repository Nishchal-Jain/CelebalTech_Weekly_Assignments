"""
hybrid_retriever.py
-------------------
adds two advanced retrieval strategies on top of the basic FAISS vector search:

1. keyword search (BM25) - the classic text retrieval algorithm that matches
   based on term frequency. works great for exact keyword matches that
   embeddings sometimes miss (like specific names, acronyms, ids).

2. hybrid search - combines vector similarity + keyword matching. the idea
   is that neither approach is perfect alone: vectors capture meaning but
   miss exact terms, BM25 catches exact terms but misses synonyms.
   blending both gives the best of both worlds.

3. re-ranking - after the initial retrieval, we use a cross-encoder model
   to re-score the results. cross-encoders are slower than bi-encoders
   but way more accurate because they see the query and passage together
   instead of comparing pre-computed vectors.

spent some time experimenting with the alpha parameter for hybrid search.
0.7 (70% vector, 30% keyword) seemed to give the best balance for our
use case, but it depends on the type of queries you expect.
"""

import math
import logging
from collections import Counter

logger = logging.getLogger(__name__)


# ======================================================================
#  BM25 Keyword Search
# ======================================================================

class BM25Searcher:
    """
    lightweight BM25 implementation for keyword-based retrieval.
    didnt want to pull in a whole library like rank_bm25 just for this,
    so i wrote a simple version from scratch. its not as optimized as
    a production implementation but it works fine for our scale.

    BM25 is basically TF-IDF on steroids - it accounts for document
    length normalization and has saturation on term frequency so
    repeating a word 100 times doesnt give 100x the score.
    """

    def __init__(self, chunk_list, k1=1.5, b=0.75):
        """
        builds the BM25 index from a list of chunk dicts.
        k1 controls term frequency saturation (higher = less saturation)
        b controls document length normalization (0 = none, 1 = full)
        """
        self.k1 = k1
        self.b = b
        self.chunk_list = chunk_list
        self.num_docs = len(chunk_list)

        # tokenize all chunks and compute stats
        self.doc_tokens = []
        self.doc_lengths = []
        self.doc_freqs = Counter()  # how many docs contain each term

        for chunk in chunk_list:
            tokens = self._tokenize(chunk["chunk_text"])
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

            # count unique terms per doc for IDF
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / max(self.num_docs, 1)
        logger.info(f"BM25 index built: {self.num_docs} docs, {len(self.doc_freqs)} unique terms")

    def _tokenize(self, text):
        """
        dead simple tokenizer - lowercase, split on non-alphanumeric.
        not trying to be fancy here, just need something that works.
        a real system would use proper NLP tokenization.
        """
        import re
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return tokens

    def _compute_idf(self, term):
        """
        inverse document frequency - terms that appear in fewer docs
        get higher weight. the +0.5 smoothing prevents division by zero
        and gives a small boost to very rare terms.
        """
        df = self.doc_freqs.get(term, 0)
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, top_k=5):
        """
        scores all documents against the query using BM25.
        returns the top-k chunks sorted by score.
        """
        query_tokens = self._tokenize(query)

        scores = []
        for doc_idx in range(self.num_docs):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            term_counts = Counter(self.doc_tokens[doc_idx])

            for term in query_tokens:
                if term not in term_counts:
                    continue

                tf = term_counts[term]
                idf = self._compute_idf(term)

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        # sort by score descending and take top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        results = []
        for idx, score in top_results:
            if score > 0:  # only include docs that actually matched something
                chunk = self.chunk_list[idx].copy()
                chunk["bm25_score"] = score
                results.append(chunk)

        return results


# ======================================================================
#  Hybrid Search (Vector + Keyword)
# ======================================================================

def hybrid_search(faiss_index, bm25_searcher, query_vec, embedding_model,
                  query_text, all_chunks, top_k=5, alpha=0.7):
    """
    combines FAISS vector search with BM25 keyword search.

    alpha controls the blending:
      - alpha=1.0 means pure vector search
      - alpha=0.0 means pure keyword search
      - alpha=0.7 means 70% vector + 30% keyword (our default)

    the tricky part is normalizing scores since FAISS and BM25 use
    completely different score ranges. we min-max normalize both
    to [0, 1] before blending so neither dominates.
    """
    from vector_store import find_closest_chunks

    # grab more candidates than we need, then re-rank
    candidate_k = min(top_k * 3, len(all_chunks))

    # --- vector search ---
    vector_results = find_closest_chunks(faiss_index, query_vec, all_chunks, top_k=candidate_k)

    # --- keyword search ---
    keyword_results = bm25_searcher.search(query_text, top_k=candidate_k)

    # --- merge and normalize scores ---
    # use chunk_index as the key since the same chunk can appear in both
    score_map = {}  # chunk_index -> {"vector": float, "keyword": float}

    for chunk in vector_results:
        idx = chunk["chunk_index"]
        source = chunk["source"]
        key = f"{source}_{idx}"
        if key not in score_map:
            score_map[key] = {"vector": 0.0, "keyword": 0.0, "chunk": chunk}
        score_map[key]["vector"] = chunk.get("relevance_score", 0.0)

    for chunk in keyword_results:
        idx = chunk["chunk_index"]
        source = chunk["source"]
        key = f"{source}_{idx}"
        if key not in score_map:
            score_map[key] = {"vector": 0.0, "keyword": 0.0, "chunk": chunk}
        score_map[key]["keyword"] = chunk.get("bm25_score", 0.0)

    # min-max normalize each score type
    vec_scores = [v["vector"] for v in score_map.values()]
    kw_scores = [v["keyword"] for v in score_map.values()]

    vec_min, vec_max = min(vec_scores) if vec_scores else 0, max(vec_scores) if vec_scores else 1
    kw_min, kw_max = min(kw_scores) if kw_scores else 0, max(kw_scores) if kw_scores else 1

    vec_range = vec_max - vec_min if vec_max != vec_min else 1.0
    kw_range = kw_max - kw_min if kw_max != kw_min else 1.0

    # compute blended scores
    blended_results = []
    for key, data in score_map.items():
        norm_vec = (data["vector"] - vec_min) / vec_range
        norm_kw = (data["keyword"] - kw_min) / kw_range

        combined_score = alpha * norm_vec + (1 - alpha) * norm_kw

        result_chunk = data["chunk"].copy()
        result_chunk["hybrid_score"] = combined_score
        result_chunk["vector_score_norm"] = norm_vec
        result_chunk["keyword_score_norm"] = norm_kw
        result_chunk["relevance_score"] = combined_score  # override for downstream compat
        blended_results.append(result_chunk)

    # sort by combined score and return top-k
    blended_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    logger.debug(
        f"hybrid search: {len(vector_results)} vector + {len(keyword_results)} keyword "
        f"candidates -> {min(top_k, len(blended_results))} final results"
    )

    return blended_results[:top_k]


# ======================================================================
#  Cross-Encoder Re-Ranking
# ======================================================================

_reranker_model = None  # lazy-loaded global so we dont reload every query


def rerank_chunks(query, candidate_chunks, top_k=5):
    """
    re-ranks candidate chunks using a cross-encoder model.

    unlike bi-encoders (which encode query and doc separately), cross-encoders
    process the query+doc pair together through the transformer. this is
    way more accurate but also way slower, which is why we only use it
    on a small candidate set (usually 15-20 chunks) rather than the full corpus.

    using the ms-marco MiniLM cross-encoder because its small and fast
    enough to run on cpu without making people wait forever.
    """
    global _reranker_model

    if not candidate_chunks:
        return []

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("CrossEncoder not available, skipping re-ranking")
        return candidate_chunks[:top_k]

    # lazy-load the reranker model on first use
    if _reranker_model is None:
        logger.info("loading cross-encoder re-ranker model (first time only)...")
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("re-ranker model loaded")

    # build query-passage pairs for the cross-encoder
    pairs = [(query, chunk["chunk_text"]) for chunk in candidate_chunks]

    # score all pairs at once
    rerank_scores = _reranker_model.predict(pairs)

    # attach scores and sort
    for i, chunk in enumerate(candidate_chunks):
        chunk["rerank_score"] = float(rerank_scores[i])
        chunk["relevance_score"] = float(rerank_scores[i])  # override for downstream

    reranked = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)

    logger.debug(
        f"re-ranking: top score went from {candidate_chunks[0].get('hybrid_score', 0):.4f} "
        f"to rerank={reranked[0]['rerank_score']:.4f}"
    )

    return reranked[:top_k]
