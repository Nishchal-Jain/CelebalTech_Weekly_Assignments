"""
vector_store.py
---------------
handles building and querying the FAISS index. FAISS (Facebook AI
Similarity Search) is basically a specialized database for vectors -
you throw in a bunch of vectors and it can find the closest ones to
any query vector really fast.

using IndexFlatIP (inner product) instead of IndexFlatL2 (euclidean)
because we normalize all our vectors in the embedding step. when vectors
are unit-length, inner product gives the same ranking as cosine similarity
but FAISS computes it faster.

also added save/load so we dont have to re-embed everything each time
we run the pipeline. the index file ends up being pretty small since
our vectors are only 384 dimensions.
"""

import os
import numpy as np
import faiss
import logging

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings_matrix):
    """
    creates a FAISS index from the embedding matrix.

    IndexFlatIP does exact inner-product search - no approximation,
    which means perfect recall. for our document sizes (hundreds to
    low thousands of chunks) this is fast enough. if we ever needed
    to scale to millions of documents, we'd switch to an approximate
    index like IVF or HNSW.

    returns the populated FAISS index object.
    """
    if embeddings_matrix.size == 0:
        logger.error("cant build index from empty embeddings")
        return None

    if not isinstance(embeddings_matrix, np.ndarray):
        embeddings_matrix = np.array(embeddings_matrix, dtype=np.float32)
    if embeddings_matrix.dtype != np.float32 or not embeddings_matrix.flags["C_CONTIGUOUS"]:
        embeddings_matrix = np.ascontiguousarray(embeddings_matrix, dtype=np.float32)

    vec_dimension = embeddings_matrix.shape[1]
    num_vectors = embeddings_matrix.shape[0]

    logger.info(f"building FAISS index: {num_vectors} vectors x {vec_dimension} dims")

    # inner product index - works like cosine sim with normalized vectors
    index = faiss.IndexFlatIP(vec_dimension)
    index.add(embeddings_matrix)

    logger.info(f"index built successfully, total vectors stored: {index.ntotal}")
    return index


def find_closest_chunks(index, query_vec, all_chunks, top_k=5):
    """
    searches the FAISS index for the chunks most similar to the query.

    returns a list of dicts with the matched chunk info plus
    the similarity score. higher score = more relevant.

    the scores from inner product range from -1 to 1 (since vectors
    are normalized), where 1 means identical and 0 means unrelated.
    in practice anything above 0.3 is usually somewhat relevant
    and above 0.5 is a pretty solid match.
    """
    if index is None or index.ntotal == 0:
        logger.warning("index is empty, cant search")
        return []

    if not isinstance(query_vec, np.ndarray):
        query_vec = np.array(query_vec, dtype=np.float32)
    if len(query_vec.shape) == 1:
        query_vec = query_vec.reshape(1, -1)
    if query_vec.dtype != np.float32 or not query_vec.flags["C_CONTIGUOUS"]:
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

    if query_vec.shape[1] != index.d:
        logger.error(f"dimension mismatch: query has {query_vec.shape[1]} dims but index expects {index.d} dims")
        return []

    # clamp top_k so we dont ask for more results than we have vectors
    actual_k = min(top_k, index.ntotal)

    # FAISS search returns distances and indices arrays
    similarity_scores, match_indices = index.search(query_vec, actual_k)

    top_matches = []
    for rank, (score, idx) in enumerate(zip(similarity_scores[0], match_indices[0])):
        if idx == -1 or idx < 0 or idx >= len(all_chunks):
            continue

        matched_chunk = all_chunks[idx].copy()
        matched_chunk["relevance_score"] = float(score)
        matched_chunk["retrieval_rank"] = rank + 1
        top_matches.append(matched_chunk)

    logger.debug(f"retrieved {len(top_matches)} chunks (top score: {top_matches[0]['relevance_score']:.4f})" if top_matches else "no matches found")
    return top_matches


def save_index(index, filepath):
    """
    saves the FAISS index to disk so we can reuse it without
    re-embedding everything. the file is compact - a few MB
    for a thousand 384-dim vectors.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    faiss.write_index(index, filepath)
    logger.info(f"saved FAISS index to {filepath}")


def load_index(filepath):
    """
    loads a previously saved FAISS index from disk.
    returns None if the file doesnt exist.
    """
    if not os.path.exists(filepath):
        logger.warning(f"no saved index found at {filepath}")
        return None

    index = faiss.read_index(filepath)
    logger.info(f"loaded FAISS index from {filepath} ({index.ntotal} vectors)")
    return index
