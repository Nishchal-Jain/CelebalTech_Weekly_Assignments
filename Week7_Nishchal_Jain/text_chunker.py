"""
text_chunker.py
---------------
breaks down big text blobs into smaller overlapping chunks for embedding.

the key insight here is that you cant just blindly split every N characters -
you end up cutting sentences in half which ruins the semantic meaning.
so i implemented a recursive approach that tries to split on natural
boundaries (paragraphs first, then sentences, then words) before falling
back to hard character splits.

the overlap between chunks is important too - without it you lose context
at chunk boundaries. 80 chars of overlap seemed to work well in my tests.
"""

import logging

logger = logging.getLogger(__name__)


def _find_best_split_point(text, separators):
    """
    tries each separator in order and returns the first one that
    actually appears in the text. this way we prefer paragraph breaks
    over sentence breaks, and sentence breaks over word breaks.
    """
    for sep in separators:
        if sep in text:
            return sep
    return None


def _recursive_split(text, max_chunk_size, separators):
    """
    the actual recursive splitting logic. keeps breaking text down
    using progressively finer separators until each piece fits
    within max_chunk_size.

    this is the core of the chunking strategy - spent a good amount
    of time getting this right because bad chunking = bad retrieval.
    """
    # base case - text already fits in one chunk
    if len(text) <= max_chunk_size:
        return [text]

    best_sep = _find_best_split_point(text, separators)

    # worst case: no separator found, just hard-split at max size
    if best_sep is None:
        chunks = []
        for start in range(0, len(text), max_chunk_size):
            chunks.append(text[start:start + max_chunk_size])
        return chunks

    # split on the best separator we found
    parts = text.split(best_sep)

    # now merge small parts together until they hit the size limit
    merged_chunks = []
    current_piece = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # check if adding this part would exceed the limit
        test_merge = current_piece + best_sep + part if current_piece else part

        if len(test_merge) <= max_chunk_size:
            current_piece = test_merge
        else:
            # save what we have so far
            if current_piece:
                merged_chunks.append(current_piece)
            # if this single part is still too big, recurse with finer separators
            if len(part) > max_chunk_size:
                remaining_seps = separators[separators.index(best_sep) + 1:] if best_sep in separators else []
                sub_chunks = _recursive_split(part, max_chunk_size, remaining_seps)
                merged_chunks.extend(sub_chunks)
                current_piece = ""
            else:
                current_piece = part

    if current_piece:
        merged_chunks.append(current_piece)

    return merged_chunks


def split_into_chunks(raw_text, chunk_size=500, overlap=80, source_tag="unknown"):
    """
    main chunking function. takes raw text and returns a list of chunk dicts.

    the overlap parameter controls how many characters from the end of one
    chunk get repeated at the start of the next. this helps maintain context
    across chunk boundaries - without it the retriever might miss relevant
    info that spans two chunks.

    returns: list of {"chunk_text": str, "chunk_index": int, "source": str}
    """
    if not raw_text or not raw_text.strip():
        return []

    # clean up the text a bit - normalize whitespace but keep paragraph breaks
    cleaned = raw_text.strip()

    # separator hierarchy: try paragraph breaks first, then sentences, then words
    separator_priority = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    raw_pieces = _recursive_split(cleaned, chunk_size, separator_priority)

    # now add overlap between consecutive chunks
    # this is where the magic happens for retrieval quality
    overlapped_chunks = []
    for i, piece in enumerate(raw_pieces):
        if i > 0 and overlap > 0:
            # grab the tail end of the previous chunk as context prefix
            prev_tail = raw_pieces[i - 1][-overlap:]
            # find a clean word boundary to start the overlap
            space_idx = prev_tail.find(" ")
            if space_idx != -1:
                prev_tail = prev_tail[space_idx + 1:]
            piece = prev_tail + " " + piece

        overlapped_chunks.append(piece.strip())

    # package everything up with metadata
    chunk_records = []
    for idx, chunk_text in enumerate(overlapped_chunks):
        if chunk_text and len(chunk_text) > 10:  # skip tiny fragments
            chunk_records.append({
                "chunk_text": chunk_text,
                "chunk_index": idx,
                "source": source_tag
            })

    logger.info(
        f"split '{source_tag}' into {len(chunk_records)} chunks "
        f"(target size={chunk_size}, overlap={overlap})"
    )
    return chunk_records


def chunk_all_documents(documents, chunk_size=500, overlap=80):
    """
    convenience wrapper that chunks a whole pile of documents at once.
    takes the output of document_loader.load_all_documents() directly.
    """
    all_chunks = []

    for doc in documents:
        doc_chunks = split_into_chunks(
            raw_text=doc["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            source_tag=doc["source"]
        )
        all_chunks.extend(doc_chunks)

    logger.info(f"chunking complete: {len(all_chunks)} total chunks from {len(documents)} documents")
    return all_chunks
