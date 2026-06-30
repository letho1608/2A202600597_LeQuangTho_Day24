from __future__ import annotations

"""M5: Chunk enrichment — adds an LLM-generated ``enriched_text`` and
``auto_metadata`` (tags, summary) to each chunk.

Graceful fallback: if ``OPENAI_API_KEY`` is missing or the LLM call
fails, returns the input list unchanged (no ``enriched_text``/tags added)
so downstream code can detect "not enriched" and proceed.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import JUDGE_MODEL, OPENAI_API_KEY  # noqa: E402


# ─── Data class returned for each enriched chunk ──────────────────────────────

@dataclass
class EnrichedChunk:
    text: str
    enriched_text: str = ""
    auto_metadata: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    chunk_id: str | None = None


def _to_enriched(chunk: Any, enriched_text: str = "",
                 auto_metadata: dict | None = None) -> EnrichedChunk:
    if isinstance(chunk, dict):
        return EnrichedChunk(
            text=chunk.get("text", ""),
            enriched_text=enriched_text,
            auto_metadata=dict(auto_metadata or {}),
            metadata=dict(chunk.get("metadata", {})),
            chunk_id=chunk.get("chunk_id"),
        )
    return EnrichedChunk(
        text=getattr(chunk, "text", ""),
        enriched_text=enriched_text,
        auto_metadata=dict(auto_metadata or {}),
        metadata=dict(getattr(chunk, "metadata", {}) or {}),
        chunk_id=getattr(chunk, "chunk_id", None),
    )


# ─── LLM call (optional) ──────────────────────────────────────────────────────

def _enrich_one(text: str) -> tuple[str, dict]:
    """Ask the LLM to (1) expand the chunk and (2) emit auto-metadata.

    Returns (enriched_text, auto_metadata). Both empty on failure.
    """
    if not OPENAI_API_KEY or not text.strip():
        return "", {}

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return "", {}

    prompt = (
        "Bạn là trợ lý phân tích tài liệu HR. Đọc đoạn văn sau và:\n"
        "1. Viết bản tóm tắt mở rộng (1-3 câu) giữ nguyên ý chính.\n"
        "2. Trích xuất các thẻ (tags) và phân loại (category) theo JSON.\n\n"
        "Đoạn văn:\n---\n" + text[:2000] + "\n---\n\n"
        "Trả lời JSON (chỉ JSON): "
        '{"enriched_text": "...", "auto_metadata": {"tags": [...], '
        '"category": "...", "summary": "..."}}'
    )

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý HR. Chỉ trả lời JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
        import json
        payload = json.loads(resp.choices[0].message.content)
        return (
            str(payload.get("enriched_text", "")),
            dict(payload.get("auto_metadata", {})),
        )
    except Exception:
        return "", {}


# ─── Public API ───────────────────────────────────────────────────────────────

def enrich_chunks(chunks: list[Any]) -> list[EnrichedChunk]:
    """Enrich every chunk in ``chunks``.

    Behaviour:
        * If ``OPENAI_API_KEY`` is missing → returns input chunks unchanged
          (wrapped as ``EnrichedChunk`` with empty ``enriched_text``).
        * If LLM call fails for a chunk → that chunk's ``enriched_text``
          stays empty but the chunk is still returned.
        * Never raises — always returns a list of length ``len(chunks)``.
    """
    if not chunks:
        return []

    if not OPENAI_API_KEY:
        # Graceful fallback — no key, no work.
        return [_to_enriched(c) for c in chunks]

    enriched: list[EnrichedChunk] = []
    for chunk in chunks:
        text = chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", "")
        try:
            etxt, meta = _enrich_one(text)
        except Exception:
            etxt, meta = "", {}
        enriched.append(_to_enriched(chunk, etxt, meta))
    return enriched


if __name__ == "__main__":
    from src.m1_chunking import load_and_chunk
    _, _, children = load_and_chunk()
    out = enrich_chunks(children[:2])
    print(f"Enriched {len(out)} chunks (first 2)")
    for c in out:
        enriched_flag = "yes" if c.enriched_text else "no (fallback)"
        print(f"  [{enriched_flag}] source={c.metadata.get('source')}")
        if c.enriched_text:
            print(f"    enriched: {c.enriched_text[:80]}...")
