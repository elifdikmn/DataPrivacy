"""FAISS index üzerinde arama yaparak bir soruya en alakalı dokümanları bulma."""

import json

import faiss
import numpy as np

from . import config
from .embeddings import embed_query

_index: faiss.Index | None = None
_documents: list[dict] | None = None


def _load():
    global _index, _documents
    if _index is None:
        if not config.FAISS_INDEX_PATH.exists():
            raise RuntimeError(
                "Index bulunamadı. Önce 'python -m app.indexing' çalıştırıp index'i kurun."
            )
        _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        with open(config.DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            _documents = json.load(f)
    return _index, _documents


def search(query: str, top_k: int = 5) -> list[dict]:
    """Soruya en alakalı top_k dokümanı, benzerlik skoruyla birlikte döndürür."""
    index, documents = _load()

    query_vector = embed_query(query).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = documents[idx]
        results.append({
            "text": doc["text"],
            "type": doc["type"],
            "metadata": doc["metadata"],
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "hangi parametreler şifre topluyor"
    print(f"Soru: {query}\n")
    for r in search(query, top_k=5):
        print(f"[{r['score']:.3f}] ({r['type']}) {r['text'][:120]}")
