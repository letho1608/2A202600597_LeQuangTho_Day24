from __future__ import annotations

"""M2: Hybrid search — BM25 (sparse) + Dense (bi-encoder) with RRF fusion.

Falls back gracefully when ``sentence_transformers`` or ``qdrant_client``
are unavailable so the module always imports and runs.
"""

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K  # noqa: E402


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    chunk_id: str | None = None

    def __getitem__(self, key: str) -> Any:  # dict-like access
        if key == "text":
            return self.text
        if key == "score":
            return self.score
        if key == "metadata":
            return self.metadata
        raise KeyError(key)


# ─── Tokenizer (Vietnamese-friendly) ─────────────────────────────────────────

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# ─── BM25 implementation (no external dep) ────────────────────────────────────

class _BM25:
    """Minimal Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.dl = [len(d) for d in corpus]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.df: Counter = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        if not doc:
            return 0.0
        tf: Counter = Counter(doc)
        norm = 1 - self.b + self.b * self.dl[doc_idx] / self.avgdl if self.avgdl else 1.0
        s = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            num = tf[term] * (self.k1 + 1)
            den = tf[term] + self.k1 * norm
            s += idf * num / den
        return s

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        q_tokens = _tokenize(query)
        scored = [(i, self.score(q_tokens, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]


# ─── Dense encoder (optional) ────────────────────────────────────────────────

def _try_load_dense_encoder():
    """Load a sentence-transformers model if available.

    Returns either a callable ``encode(list[str]) -> list[list[float]]`` or
    ``None`` if sentence-transformers / model are unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        # Lightweight default; real deployments swap EMBEDDING_MODEL.
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None

    def _encode(texts: list[str]):
        return model.encode(texts, normalize_embeddings=True).tolist()

    return _encode


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den_a = math.sqrt(sum(x * x for x in a))
    den_b = math.sqrt(sum(y * y for y in b))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


# ─── HybridSearch ─────────────────────────────────────────────────────────────

class HybridSearch:
    """Sparse + dense hybrid retriever with reciprocal-rank fusion.

    Usage:
        hs = HybridSearch()
        hs.index(chunks)           # chunks: list[Chunk] or list[dict]
        results = hs.search(q)     # list[SearchResult]
    """

    def __init__(self, bm25_top_k: int = BM25_TOP_K, dense_top_k: int = DENSE_TOP_K,
                 rrf_k: int = 60):
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.rrf_k = rrf_k

        self._chunks: list[Any] = []      # source chunks (Chunk or dict)
        self._tokens: list[list[str]] = [] # tokenized texts (BM25)
        self._bm25: _BM25 | None = None
        self._encoder = _try_load_dense_encoder()
        self._embeddings: list[list[float]] | None = None

    # ── Indexing ────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        return getattr(chunk, "text", "")

    @staticmethod
    def _chunk_meta(chunk: Any) -> dict:
        if isinstance(chunk, dict):
            return dict(chunk.get("metadata", {}))
        meta = getattr(chunk, "metadata", {}) or {}
        return dict(meta)

    @staticmethod
    def _chunk_id(chunk: Any, fallback_idx: int) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("chunk_id", fallback_idx))
        return str(getattr(chunk, "chunk_id", fallback_idx))

    def index(self, chunks: list[Any]) -> None:
        """Build BM25 (and dense if available) indexes over ``chunks``."""
        self._chunks = list(chunks)
        self._tokens = [_tokenize(self._chunk_text(c)) for c in self._chunks]
        self._bm25 = _BM25(self._tokens)
        self._embeddings = None

        if self._encoder is not None and self._chunks:
            try:
                texts = [self._chunk_text(c) for c in self._chunks]
                self._embeddings = self._encoder(texts)
            except Exception:
                self._embeddings = None

    # ── Searching ───────────────────────────────────────────────────────────

    def _bm25_search(self, q: str) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        return self._bm25.top_k(q, self.bm25_top_k)

    def _dense_search(self, q: str) -> list[tuple[int, float]]:
        if self._encoder is None or not self._embeddings:
            return []
        try:
            q_vec = self._encoder([q])[0]
        except Exception:
            return []
        scored = [
            (i, _cosine(q_vec, doc_vec))
            for i, doc_vec in enumerate(self._embeddings)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.dense_top_k]

    def search(self, q: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        """Return up to ``top_k`` SearchResult objects for query ``q``."""
        if not self._chunks:
            return []

        bm25_hits = self._bm25_search(q)
        dense_hits = self._dense_search(q)

        rrf: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(bm25_hits):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (idx, _score) in enumerate(dense_hits):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for idx, fused in ranked:
            chunk = self._chunks[idx]
            results.append(SearchResult(
                text=self._chunk_text(chunk),
                score=float(fused),
                metadata=self._chunk_meta(chunk),
                chunk_id=self._chunk_id(chunk, idx),
            ))
        return results


# ─── Convenience for the pipeline ─────────────────────────────────────────────

def index_chunks(chunks: list[Any]) -> HybridSearch:
    """Build and return a populated ``HybridSearch`` over ``chunks``."""
    hs = HybridSearch()
    hs.index(chunks)
    return hs


if __name__ == "__main__":
    from src.m1_chunking import load_and_chunk
    _, _, children = load_and_chunk()
    print(f"Indexing {len(children)} child chunks...")
    hs = HybridSearch()
    hs.index(children)
    for q in ["nghỉ phép năm 2024", "mật khẩu tối thiểu bao nhiêu ký tự",
              "phụ cấp ăn trưa"]:
        print(f"\nQ: {q}")
        for r in hs.search(q, top_k=3):
            src = r.metadata.get("source", "?")
            print(f"  [{r.score:.3f}] ({src}) {r.text[:80].replace(chr(10), ' ')}...")
