"""LLM answer synthesis from retrieved chunks (RAG)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from qmd.search_bm25 import SearchResult

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on document excerpts. "
    "Answer the user's question using ONLY the provided context. "
    "If the context doesn't contain enough information, say so. "
    "Be concise and direct. Cite which source file(s) your answer comes from."
)


def detect_provider() -> str | None:
    """Detect which LLM provider to use based on available API keys.

    Returns 'openai', 'anthropic', or None.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    for r in results:
        parts.append(
            f"[Source: {r.chunk.file_name}, chunk: {r.chunk.chunk_id}]\n"
            f"{r.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def synthesize_answer(
    query: str,
    results: list[SearchResult],
    provider: str | None = None,
) -> str:
    """Generate an LLM-synthesized answer from retrieved chunks.

    Args:
        query: The user's question.
        results: Retrieved SearchResult objects with chunk text.
        provider: 'openai', 'anthropic', or None (auto-detect).

    Returns:
        The synthesized answer string.

    Raises:
        RuntimeError: If no API key is configured.
        ImportError: If the required SDK is not installed.
    """
    if provider is None:
        provider = detect_provider()

    if provider is None:
        raise RuntimeError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"
        )

    context = build_context(results)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    if provider == "openai":
        return _call_openai(user_message)
    elif provider == "anthropic":
        return _call_anthropic(user_message)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_openai(user_message: str) -> str:
    """Call OpenAI API for answer synthesis."""
    from openai import OpenAI

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _call_anthropic(user_message: str) -> str:
    """Call Anthropic API for answer synthesis."""
    from anthropic import Anthropic

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return response.content[0].text
