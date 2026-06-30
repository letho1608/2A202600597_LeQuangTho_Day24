from __future__ import annotations

"""End-to-end RAG pipeline orchestrator (m1 → m5).

``run_pipeline(question)`` returns the LLM answer and the supporting
contexts. Falls back gracefully when the LLM key is missing so callers
always get a usable response.
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATA_DIR,
    JUDGE_MODEL,
    OPENAI_API_KEY,
    RERANK_TOP_K,
)


# ─── Pipeline response ────────────────────────────────────────────────────────

@dataclass
class PipelineResponse:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ─── Index cache ──────────────────────────────────────────────────────────────

_index_cache: dict = {}


def _get_searcher():
    """Build (and cache) a HybridSearch over all child chunks."""
    if "searcher" in _index_cache:
        return _index_cache["searcher"]

    from src.m1_chunking import load_and_chunk
    from src.m2_search import HybridSearch

    _, _, children = load_and_chunk(DATA_DIR)
    hs = HybridSearch()
    hs.index(children)
    _index_cache["searcher"] = hs
    _index_cache["children"] = children
    return hs


# ─── Retrieval + rerank ───────────────────────────────────────────────────────

def _retrieve(question: str, top_k: int = 5):
    from src.m3_rerank import CrossEncoderReranker

    hs = _get_searcher()
    candidates = hs.search(question, top_k=max(top_k * 3, RERANK_TOP_K * 3))
    if not candidates:
        return []

    reranker = CrossEncoderReranker()
    top = reranker.rerank(question, candidates, top_k=top_k)
    return top


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context)"
    return (
        "Bạn là trợ lý HR. Trả lời câu hỏi dựa trên các đoạn ngữ cảnh dưới đây. "
        "Nếu không tìm thấy thông tin, hãy nói rõ là không tìm thấy.\n\n"
        f"Ngữ cảnh:\n{context_block}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời ngắn gọn, chính xác:"
    )


def _llm_answer(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return ""
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý HR nội bộ."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            timeout=30,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _extractive_answer(question: str, contexts: list[str]) -> str:
    """Cheap fallback: return the first non-empty context sentence."""
    if not contexts:
        return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."
    for ctx in contexts:
        text = (ctx or "").strip()
        if text:
            return text
    return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."


# ─── Public API ───────────────────────────────────────────────────────────────

def run_pipeline(question: str, top_k: int = 5) -> PipelineResponse:
    """Run the full RAG pipeline for a single question.

    Steps:
        1. m1 — load + hierarchical chunk all .md docs (cached).
        2. m2 — hybrid (BM25 + dense) retrieval over child chunks.
        3. m3 — cross-encoder rerank of top-k candidates.
        4. (optional) m5 — enrichment metadata appended if available.
        5. LLM answer generation; falls back to extractive snippet.
    """
    question = (question or "").strip()
    reranked = _retrieve(question, top_k=top_k)

    contexts: list[str] = []
    sources: list[str] = []
    for r in reranked:
        if isinstance(r, dict):
            contexts.append(r.get("text", ""))
            meta = r.get("metadata", {}) or {}
        else:
            contexts.append(getattr(r, "text", ""))
            meta = getattr(r, "metadata", {}) or {}
        if meta.get("source") and meta["source"] not in sources:
            sources.append(meta["source"])

    prompt = _build_prompt(question, contexts)
    answer = _llm_answer(prompt)
    if not answer:
        answer = _extractive_answer(question, contexts)

    return PipelineResponse(
        question=question,
        answer=answer,
        contexts=contexts,
        metadata={
            "sources": sources,
            "n_contexts": len(contexts),
            "model": JUDGE_MODEL if OPENAI_API_KEY else "extractive-fallback",
        },
    )


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = "Nhân viên được nghỉ bao nhiêu ngày phép năm theo chính sách v2024?"
    resp = run_pipeline(q)
    print(f"Q: {resp.question}")
    print(f"A: {resp.answer}")
    print(f"Sources: {resp.metadata.get('sources')}")
    print(f"#contexts: {resp.metadata.get('n_contexts')}")
