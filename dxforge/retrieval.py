"""Evidence retrieval over the KB.

Default: BM25 lexical retrieval (rank_bm25) — pure Python, no model download, so
the repo runs anywhere with a plain `pip install`. Returns citable nodes.

Optional upgrade (requirements-full): swap in sentence-transformer embeddings +
Chroma by setting RETRIEVAL=dense. Kept behind a flag so the base clone-and-run
path has zero heavy dependencies.
"""
from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .kb import KnowledgeBase, load_kb

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class LexicalRetriever:
    def __init__(self, kb: KnowledgeBase):
        self.nodes = kb.all_text_nodes()
        self._corpus = [_tok(f"{n['title']} {n['text']}") for n in self.nodes]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, k: int = 4) -> list[dict]:
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(zip(self.nodes, scores), key=lambda x: x[1], reverse=True)
        out = []
        for node, score in ranked[:k]:
            if score <= 0:
                continue
            out.append({**node, "score": round(float(score), 3)})
        return out


@lru_cache(maxsize=1)
def get_retriever() -> LexicalRetriever:
    return LexicalRetriever(load_kb())
