from __future__ import annotations

"""M3: Cross-encoder reranker (top-k refinement after hybrid retrieval).

Uses ``flashrank`` if installed; otherwise falls back to a lightweight
token-overlap score (Jaccard on normalized tokens) so the module is
always importable and runnable in CI / offline.
"""

import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K  # noqa: E402


_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _fallback_score(query: str, doc: str) -> float:
    """Jaccard + length-aware overlap, used when no reranker model exists."""
    q = _tokenize(query)
    d = _tokenize(doc)
    if not q or not d:
        return 0.0
    jaccard = len(q & d) / len(q | d)
    coverage = len(q & d) / len(q)
    return 0.6 * jaccard + 0.4 * coverage


# ─── CrossEncoderReranker ─────────────────────────────────────────────────────

@dataclass
class _RerankHit:
    text: str
    score: float
    metadata: dict
    chunk_id: str | None = None


class CrossEncoderReranker:
    """Rerank candidate docs against a query.

    Tries ``flashrank.Ranker`` first; if unavailable, uses the fallback
    overlap scorer. Either way the public interface is identical.
    """

    def __init__(self, model_name: str | None = None,
                 default_top_k: int = RERANK_TOP_K):
        self.default_top_k = default_top_k
        self._ranker = None
        self._ranker_kind: str = "fallback"

        if model_name is None:
            model_name = "ms-marco-MiniLM-L-12-v2"

        try:
            from flashrank import Ranker  # type: ignore
            try:
                self._ranker = Ranker(model_name=model_name)
                self._ranker_kind = "flashrank"
            except Exception:
                self._ranker = None
        except Exception:
            self._ranker = None

    # ── Scoring helpers ─────────────────────────────────────────────────────

    def _score_flashrank(self, query: str, docs: list[Any]) -> list[float]:
        """Score via flashrank; returns one float per input doc."""
        passages = []
        for d in docs:
            text = d.get("text") if isinstance(d, dict) else getattr(d, "text", "")
            passages.append({"id": str(len(passages)), "text": text or "",
                             "meta": {}})
        try:
            ranked = self._ranker.rerank(query, passages)
        except Exception:
            return [_fallback_score(query, p["text"]) for p in passages]
        score_by_id = {str(h.get("id")): float(h.get("score", 0.0)) for h in ranked}
        return [score_by_id.get(p["id"], 0.0) for p in passages]

    def _score_fallback(self, query: str, docs: list[Any]) -> list[float]:
        scores: list[float] = []
        for d in docs:
            text = d.get("text") if isinstance(d, dict) else getattr(d, "text", "")
            scores.append(_fallback_score(query, text))
        return scores

    # ── Public API ─────────────────────────────────────────────────────────

    def rerank(self, q: str, docs: list[Any], top_k: int | None = None) -> list[Any]:
        """Rerank ``docs`` for query ``q`` and return the top ``top_k``.

        Each returned element is a SearchResult-like object with the same
        shape as the input — i.e. it keeps the original ``text``,
        ``metadata`` and updates ``score`` to the rerank score.
        """
        if not docs:
            return []
        k = top_k if top_k is not None else self.default_top_k

        if self._ranker is not None and self._ranker_kind == "flashrank":
            scores = self._score_flashrank(q, docs)
        else:
            scores = self._score_fallback(q, docs)

        # Attach scores, sort descending, slice top_k.
        annotated = list(zip(docs, scores))
        annotated.sort(key=lambda x: x[1], reverse=True)
        annotated = annotated[:k]

        reranked: list[Any] = []
        for original, new_score in annotated:
            if isinstance(original, dict):
                new_doc = dict(original)
                new_doc["score"] = float(new_score)
                reranked.append(new_doc)
            else:
                # Dataclass / object — mutate score field if present.
                try:
                    original.score = float(new_score)  # type: ignore[attr-defined]
                    reranked.append(original)
                except Exception:
                    # Wrap as a dict.
                    reranked.append({
                        "text": getattr(original, "text", ""),
                        "score": float(new_score),
                        "metadata": getattr(original, "metadata", {}),
                        "chunk_id": getattr(original, "chunk_id", None),
                    })
        return reranked


# ─── Convenience ──────────────────────────────────────────────────────────────

def rerank_results(q: str, docs: list[Any], top_k: int = RERANK_TOP_K) -> list[Any]:
    return CrossEncoderReranker().rerank(q, docs, top_k=top_k)


if __name__ == "__main__":
    sample = [
        {"text": "Mỗi nhân viên chính thức được hưởng 15 ngày phép năm.",
         "score": 0.6, "metadata": {"source": "nghi_phep_nam_v2024.md"}},
        {"text": "Bảo hiểm sức khỏe PVI hạn mức 200 triệu.",
         "score": 0.4, "metadata": {"source": "bao_hiem_suc_khoe.md"}},
        {"text": "Phép năm phải đăng ký trước 2 ngày.",
         "score": 0.5, "metadata": {"source": "nghi_phep_nam_v2024.md"}},
    ]
    rr = CrossEncoderReranker()
    print(f"Reranker kind: {rr._ranker_kind}")
    ranked = rr.rerank("bao nhiêu ngày phép năm", sample, top_k=2)
    for r in ranked:
        print(f"  score={r.get('score') if isinstance(r, dict) else r.score:.3f} "
              f"{r.get('text') if isinstance(r, dict) else r.text}")
