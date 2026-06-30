from __future__ import annotations

"""M1: Document loading + hierarchical (parent/child) chunking.

Reads Markdown files from ``config.DATA_DIR``, splits each document into
parent chunks (~ config.HIERARCHICAL_PARENT_SIZE chars) and child chunks
(~ config.HIERARCHICAL_CHILD_SIZE chars). Each child carries a
``metadata.parent_id`` pointer to its parent.

Vietnamese documents are split by paragraphs (since ``underthesea`` /
``langchain`` splitters are not always installed in CI). Markdown
headers are still detected so parents get a ``section`` field when
present.
"""

import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Generic chunk with text and metadata.

    ``metadata`` is a plain dict; chunks expose ``.parent_id`` for
    parent-child lookups.
    """

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def parent_id(self) -> str | None:
        return self.metadata.get("parent_id")

    @parent_id.setter
    def parent_id(self, value: str | None) -> None:
        self.metadata["parent_id"] = value


# ─── Document loading ─────────────────────────────────────────────────────────

def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Read every ``*.md`` file under ``data_dir``.

    Returns:
        list of {"text": str, "metadata": dict} where ``metadata`` carries
        ``source`` (filename), ``path`` (absolute path) and any front-matter
        fields parsed from the top-of-file blockquote.
    """
    documents: list[dict] = []
    if not os.path.isdir(data_dir):
        return documents

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".md"):
            continue
        fpath = os.path.join(data_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue

        metadata = _parse_front_matter(text)
        metadata["source"] = fname
        metadata["path"] = fpath

        documents.append({"text": text, "metadata": metadata})
    return documents


def _parse_front_matter(text: str) -> dict:
    """Extract ``> Key: Value`` lines that appear right after the H1."""
    metadata: dict = {}
    for line in text.splitlines()[:10]:
        line = line.strip()
        if not line.startswith(">"):
            continue
        body = line.lstrip(">").strip()
        if ":" in body:
            key, _, value = body.partition(":")
            metadata[key.strip().lower().replace(" ", "_")] = value.strip()
    return metadata


# ─── Hierarchical chunking ────────────────────────────────────────────────────

_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a Markdown doc into (header, body) sections by H1/H2/H3.

    The first section uses the document title (line 0) as its header.
    """
    sections: list[tuple[str, str]] = []
    matches = list(_MD_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    # Pre-title prelude (before first header) → attach to title.
    first_start = matches[0].start()
    prelude = text[:first_start].strip()
    title_line = text.splitlines()[0].lstrip("# ").strip() if text.lstrip().startswith("#") else ""

    for i, m in enumerate(matches):
        header = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if i == 0 and prelude and title_line and header == title_line:
            body = (prelude + "\n\n" + body).strip()
        sections.append((header, body))

    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Vietnamese-friendly paragraph splitter (blank-line delimited)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def _chunk_by_size(text: str, max_size: int) -> list[str]:
    """Greedy pack paragraphs into chunks no larger than ``max_size``.

    Falls back to hard character slicing for paragraphs larger than
    ``max_size`` (rare for these short policy docs).
    """
    chunks: list[str] = []
    buf = ""
    for para in _split_paragraphs(text):
        if not buf:
            buf = para
            continue
        if len(buf) + 2 + len(para) <= max_size:
            buf = buf + "\n\n" + para
        else:
            chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)

    # Hard-split any oversized chunk.
    out: list[str] = []
    for c in chunks:
        if len(c) <= max_size:
            out.append(c)
        else:
            for i in range(0, len(c), max_size):
                out.append(c[i : i + max_size])
    return out


def chunk_hierarchical(
    text: str,
    metadata: dict,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
) -> tuple[list[Chunk], list[Chunk]]:
    """Split a single document into parent and child chunks.

    Args:
        text:       the full document text.
        metadata:   document-level metadata (will be copied to every chunk).
        parent_size: target max chars per parent chunk.
        child_size:  target max chars per child chunk.

    Returns:
        (parents, children) where each item is a ``Chunk`` and every child
        has its ``parent_id`` set to the id of its parent.
    """
    parents: list[Chunk] = []
    children: list[Chunk] = []

    sections = _split_into_sections(text)
    for section_idx, (header, body) in enumerate(sections):
        # Build parent-level chunks for this section.
        section_chunks = _chunk_by_size(body, parent_size) if body.strip() else [""]
        for p_idx, p_text in enumerate(section_chunks):
            parent_meta = dict(metadata)
            parent_meta["chunk_type"] = "parent"
            parent_meta["section"] = header
            parent_meta["section_index"] = section_idx
            parent_meta["parent_index"] = p_idx
            parent = Chunk(text=p_text, metadata=parent_meta)
            parents.append(parent)

            # Children — skip empty parents.
            if not p_text.strip():
                continue
            child_chunks = _chunk_by_size(p_text, child_size)
            for c_idx, c_text in enumerate(child_chunks):
                child_meta = dict(metadata)
                child_meta["chunk_type"] = "child"
                child_meta["section"] = header
                child_meta["section_index"] = section_idx
                child_meta["parent_index"] = p_idx
                child_meta["child_index"] = c_idx
                child = Chunk(text=c_text, metadata=child_meta)
                child.parent_id = parent.chunk_id
                children.append(child)

    return parents, children


# ─── Convenience: load + chunk in one call ────────────────────────────────────

def load_and_chunk(
    data_dir: str = DATA_DIR,
) -> tuple[list[dict], list[Chunk], list[Chunk]]:
    """Load every .md file and return (documents, parents, children)."""
    documents = load_documents(data_dir)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for doc in documents:
        p, c = chunk_hierarchical(doc["text"], doc["metadata"])
        parents.extend(p)
        children.extend(c)
    return documents, parents, children


if __name__ == "__main__":
    docs, p, c = load_and_chunk()
    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(p)} parents, {len(c)} children")
    if c:
        sample = c[0]
        print(f"Sample child: parent_id={sample.parent_id}, "
              f"section={sample.metadata.get('section')}, "
              f"len={len(sample.text)}")
