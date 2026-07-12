"""
document_loader.py
------------------
handles pulling text from different sources - pdfs, plain text files,
and the huggingface vectara/open_ragbench dataset.

i kept each loader as its own function so its easy to add new formats
later (like docx or html) without touching the existing code.

the huggingface part was tricky because open_ragbench stores its data
as structured json files (corpus/, queries.json, answers.json, qrels.json)
rather than a flat tabular format. the datasets library's load_dataset()
chokes on the mixed types in the json, so we use huggingface_hub to
download the repo and then parse the json files ourselves.
"""

import os
import json
import glob
import logging

logger = logging.getLogger(__name__)

# cached path gets set after first download - avoids re-scanning every time
_HF_SNAPSHOT_DIR = None


def grab_text_from_pdf(filepath):
    """
    reads a pdf file page by page and smashes all the text together.
    PyPDF2 isnt perfect with complex layouts but it handles most
    normal documents fine - good enough for what we need here.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.error("PyPDF2 not installed - run: pip install PyPDF2")
        return ""

    if not os.path.exists(filepath):
        logger.warning(f"pdf not found at {filepath}, skipping")
        return ""

    reader = PdfReader(filepath)
    page_texts = []

    for page_num, page in enumerate(reader.pages):
        try:
            extracted = page.extract_text()
            if extracted and extracted.strip():
                page_texts.append(extracted.strip())
                logger.debug(f"  pulled {len(extracted)} chars from page {page_num + 1}")
        except Exception as e:
            logger.warning(f"error reading page {page_num + 1} of {filepath}: {e}")
            continue

    combined = "\n\n".join(page_texts)
    logger.info(f"extracted {len(combined)} chars total from {os.path.basename(filepath)}")
    return combined


def grab_text_from_file(filepath):
    """
    straightforward text file reader. works for .txt and .md files.
    nothing fancy but gets the job done.
    """
    if not os.path.exists(filepath):
        logger.warning(f"file not found at {filepath}, skipping")
        return ""

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    logger.info(f"loaded {len(content)} chars from {os.path.basename(filepath)}")
    return content


def _locate_hf_snapshot(dataset_name):
    """
    finds the cached snapshot directory for a huggingface dataset.
    the hf_hub library caches downloads under ~/.cache/huggingface/hub/
    with a specific folder naming convention.

    downloads the dataset if it hasnt been cached yet.
    returns the path to the snapshot directory.
    """
    global _HF_SNAPSHOT_DIR
    if _HF_SNAPSHOT_DIR and os.path.isdir(_HF_SNAPSHOT_DIR):
        return _HF_SNAPSHOT_DIR

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed - run: pip install huggingface-hub")
        return None

    logger.info(f"downloading/locating dataset repo: {dataset_name}")
    logger.info("(first run downloads ~1000 files, after that its cached)")

    try:
        snapshot_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset"
        )
        _HF_SNAPSHOT_DIR = snapshot_path
        logger.info(f"dataset snapshot located at: {snapshot_path}")
        return snapshot_path
    except Exception as e:
        logger.error(f"failed to download dataset: {e}")
        return None


def pull_huggingface_corpus(dataset_name, max_documents=50):
    """
    pulls document text from the vectara/open_ragbench dataset.

    the dataset structure on huggingface is:
      pdf/arxiv/
        corpus/          -> individual json files per paper
          2401.01872v2.json  -> {title, sections: [{section_id, text}], id, ...}
        queries.json     -> {uuid: {query, type, source}}
        answers.json     -> {uuid: answer_text}
        qrels.json       -> {uuid: {doc_id, section_id}}

    we parse the corpus json files directly to extract document text.
    each paper's sections get combined into one document string.
    """
    snapshot_dir = _locate_hf_snapshot(dataset_name)
    if not snapshot_dir:
        return []

    corpus_dir = os.path.join(snapshot_dir, "pdf", "arxiv", "corpus")
    if not os.path.isdir(corpus_dir):
        logger.error(f"corpus directory not found at {corpus_dir}")
        return []

    corpus_files = sorted(glob.glob(os.path.join(corpus_dir, "*.json")))
    logger.info(f"found {len(corpus_files)} corpus files in dataset")

    # cap how many we process so ingestion doesnt take forever
    files_to_process = corpus_files[:max_documents]

    document_pile = []
    for filepath in files_to_process:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                paper_data = json.load(f)

            paper_id = paper_data.get("id", os.path.basename(filepath).replace(".json", ""))
            title = paper_data.get("title", "untitled").strip()
            sections = paper_data.get("sections", [])

            # combine all section texts into one document
            # each section has {section_id: int, text: str}
            section_texts = []
            for sec in sections:
                sec_text = sec.get("text", "").strip()
                if sec_text:
                    section_texts.append(sec_text)

            if section_texts:
                # put the title at the top followed by all sections
                full_text = f"{title}\n\n" + "\n\n".join(section_texts)
                document_pile.append({
                    "text": full_text,
                    "source": f"arxiv:{paper_id}",
                    "doc_id": paper_id
                })

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"skipping {os.path.basename(filepath)}: {e}")
            continue

    logger.info(f"collected {len(document_pile)} documents from huggingface corpus")
    return document_pile


def pull_hf_question_answers(dataset_name, max_pairs=30):
    """
    pulls question-answer pairs from the dataset for validation.

    queries.json maps uuid -> {query, type, source}
    answers.json maps uuid -> answer_text (string)
    qrels.json maps uuid -> {doc_id, section_id}

    we join these together to get complete q&a pairs.
    """
    snapshot_dir = _locate_hf_snapshot(dataset_name)
    if not snapshot_dir:
        return []

    arxiv_dir = os.path.join(snapshot_dir, "pdf", "arxiv")
    queries_path = os.path.join(arxiv_dir, "queries.json")
    answers_path = os.path.join(arxiv_dir, "answers.json")

    if not os.path.exists(queries_path) or not os.path.exists(answers_path):
        logger.warning("queries.json or answers.json not found in dataset")
        return []

    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            queries_data = json.load(f)

        with open(answers_path, "r", encoding="utf-8") as f:
            answers_data = json.load(f)

        qa_pairs = []
        for query_uuid, query_info in queries_data.items():
            if len(qa_pairs) >= max_pairs:
                break

            question = query_info.get("query", "")
            answer = answers_data.get(query_uuid, "")

            # answers can be either strings or dicts depending on the entry
            if isinstance(answer, dict):
                answer = answer.get("text", str(answer))

            if question and answer and len(question) > 10:
                qa_pairs.append({
                    "question": question.strip(),
                    "answer": str(answer).strip(),
                    "query_type": query_info.get("type", "unknown"),
                    "source_type": query_info.get("source", "unknown")
                })

        logger.info(f"loaded {len(qa_pairs)} q&a pairs for validation")
        return qa_pairs

    except Exception as e:
        logger.warning(f"couldnt load q&a pairs: {e}")
        return []


def load_all_documents(source_dir, use_hf_dataset=False, hf_dataset_name=None, hf_max_docs=50):
    """
    main entry point for document loading. scans the source directory
    for pdfs and text files, optionally pulls in huggingface data too.

    returns a list of dicts: [{"text": str, "source": str}, ...]
    keeping track of where each piece came from helps with debugging
    when a retrieval result looks weird.
    """
    document_pile = []

    if not source_dir:
        logger.warning("source_dir is empty or None, skipping local file loading")
    elif os.path.isfile(source_dir):
        logger.info(f"loading single file: {source_dir}...")
        filename = os.path.basename(source_dir)
        if filename.lower().endswith(".pdf"):
            text = grab_text_from_pdf(source_dir)
            if text:
                document_pile.append({"text": text, "source": filename})
        elif filename.lower().endswith((".txt", ".md")):
            text = grab_text_from_file(source_dir)
            if text:
                document_pile.append({"text": text, "source": filename})
        else:
            logger.warning(f"unsupported format for file: {filename}")
    elif os.path.isdir(source_dir):
        logger.info(f"scanning {source_dir} for documents...")
        for filename in sorted(os.listdir(source_dir)):
            filepath = os.path.join(source_dir, filename)

            if filename.lower().endswith(".pdf"):
                text = grab_text_from_pdf(filepath)
                if text:
                    document_pile.append({"text": text, "source": filename})

            elif filename.lower().endswith((".txt", ".md")):
                text = grab_text_from_file(filepath)
                if text:
                    document_pile.append({"text": text, "source": filename})

            else:
                logger.debug(f"skipping {filename} - unsupported format")
    else:
        logger.warning(f"source path {source_dir} doesnt exist")

    # --- optionally pull huggingface documents ---
    if use_hf_dataset and hf_dataset_name:
        hf_docs = pull_huggingface_corpus(hf_dataset_name, max_documents=hf_max_docs)
        for doc in hf_docs:
            document_pile.append({
                "text": doc["text"],
                "source": doc["source"]
            })

    logger.info(f"total documents loaded: {len(document_pile)}")
    return document_pile
