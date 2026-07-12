"""
answer_generator.py
-------------------
handles the "generation" part of RAG. takes the retrieved context chunks
and the user's question, builds a prompt, and sends it to google's
gemini api to get a grounded answer.

the prompt design is important here - i tried a few different approaches
and found that explicitly telling the model to only use the provided
context (and admit when it doesnt know) gives way better results than
just dumping everything together. without that instruction the model
tends to hallucinate stuff from its training data which defeats the
whole purpose of RAG.
"""

import logging

logger = logging.getLogger(__name__)


def craft_prompt(question, context_chunks):
    """
    builds the augmented prompt by combining retrieved context with
    the user's question. the structure matters a lot here:

    1. system instruction telling the model to stick to the context
    2. the retrieved passages clearly labeled with source info
    3. the actual question at the end

    tried putting the question first but the model seemed to anchor
    on it too much and ignore the context. question-at-the-end works
    better for getting grounded responses.
    """
    # stitch together the context passages
    context_block = ""
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("relevance_score", 0.0)
        text = chunk.get("chunk_text", "")
        context_block += f"\n--- Passage {i} (source: {source}, relevance: {score:.3f}) ---\n"
        context_block += text + "\n"

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context passages below. 

RULES:
- Answer using ONLY the information found in the context passages
- If the context doesn't contain enough information to fully answer the question, say so honestly
- Keep your answer clear and concise
- Reference which passage(s) your answer comes from when possible

CONTEXT PASSAGES:
{context_block}

QUESTION: {question}

ANSWER:"""

    return prompt


def ask_gemini(prompt, api_key, model_name="gemini-flash-latest"):
    """
    sends the crafted prompt to google's gemini api and returns
    the generated response text.

    using the google-genai sdk (newer one, not the deprecated
    google-generativeai package). the free tier has rate limits
    but for a demo/testing pipeline its more than enough.

    if the api call fails for any reason, returns an error message
    instead of crashing - the pipeline can keep running and we
    just log the failure.
    """
    api_key = api_key.strip() if api_key else ""
    if not api_key:
        error_msg = (
            "no gemini api key found! grab a free one from "
            "https://aistudio.google.com/ and put it in your .env file"
        )
        logger.error(error_msg)
        return f"[ERROR] {error_msg}"

    try:
        from google import genai
    except ImportError:
        logger.error("google-genai not installed - run: pip install google-genai")
        return "[ERROR] google-genai package not installed"

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        answer_text = response.text.strip()
        logger.info(f"gemini returned {len(answer_text)} chars")
        return answer_text

    except Exception as e:
        error_msg = f"gemini api call failed: {str(e)}"
        logger.error(error_msg)
        # return the error as the answer so the pipeline doesnt crash
        # but we still see what went wrong
        return f"[ERROR] {error_msg}"


def generate_answer(question, context_chunks, api_key, model_name="gemini-2.0-flash"):
    """
    end-to-end generation: craft prompt -> call gemini -> return answer.
    this is the function the pipeline actually calls.

    returns a dict with the answer and some metadata about the generation
    (prompt length, context chunk count, etc) for the metrics report.
    """
    prompt = craft_prompt(question, context_chunks)

    logger.info(f"sending prompt to gemini ({len(prompt)} chars, {len(context_chunks)} context chunks)")
    answer = ask_gemini(prompt, api_key, model_name)

    result = {
        "answer": answer,
        "prompt_length": len(prompt),
        "context_chunks_used": len(context_chunks),
        "model": model_name,
        "question": question
    }

    return result
