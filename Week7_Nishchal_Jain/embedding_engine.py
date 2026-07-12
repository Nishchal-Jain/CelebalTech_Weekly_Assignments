"""
embedding_engine.py
-------------------
wraps the sentence-transformers library to handle all the vector stuff.

the main idea: take text strings and convert them to numerical vectors
that capture their meaning. similar texts end up with similar vectors,
which is how retrieval works - we find chunks whose vectors are close
to the query vector.

went with all-MiniLM-L6-v2 because its a good balance between speed
and quality. it produces 384-dimensional vectors which are small enough
to keep things fast but expressive enough to capture semantic nuance.
tested a couple other models but this one gave the best results for
the time it takes to encode.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def boot_embedding_model(model_tag="all-MiniLM-L6-v2"):
    """
    loads up the sentence transformer model. this takes a few seconds
    on first call because it downloads model weights, but after that
    its cached locally so subsequent runs are instant.

    returns the loaded model object ready for encoding.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed - run: pip install sentence-transformers")
        raise

    logger.info(f"loading embedding model: {model_tag}")
    model = SentenceTransformer(model_tag)

    # quick sanity check - encode a test string to verify dimensions
    test_vec = model.encode(["test"])
    vec_dimensions = test_vec.shape[1]
    logger.info(f"model loaded successfully - produces {vec_dimensions}-dim vectors")

    return model


def vectorize_chunks(model, chunk_list, batch_size=64):
    """
    takes a list of chunk dicts and encodes all their text into vectors.
    does it in batches to avoid memory issues with large document sets.

    the normalize_embeddings=True part is important - it makes all vectors
    unit length so inner product == cosine similarity. this means we can
    use the faster inner product search in FAISS instead of the slower
    cosine similarity computation.

    returns a numpy array of shape (num_chunks, embedding_dim)
    """
    if not chunk_list:
        logger.warning("got an empty chunk list, nothing to vectorize")
        return np.array([])

    # pull out just the text strings for encoding
    text_strings = [chunk["chunk_text"] for chunk in chunk_list]

    logger.info(f"vectorizing {len(text_strings)} chunks in batches of {batch_size}...")

    all_embeddings = model.encode(
        text_strings,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # crucial for cosine sim via inner product
    )

    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"embedding matrix shape: {embeddings_matrix.shape}")

    return embeddings_matrix


def vectorize_query(model, question):
    """
    encodes a single question string into a vector.
    same normalization as the chunks so the similarity scores
    are directly comparable.

    returns a numpy array of shape (1, embedding_dim)
    """
    query_vec = model.encode(
        [question],
        normalize_embeddings=True
    )

    return np.array(query_vec, dtype=np.float32)
