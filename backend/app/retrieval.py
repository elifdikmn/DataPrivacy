"""FAISS index'leri üzerinde arama yaparak bir soruya en alakalı dokümanları bulma.

Kayıtlar ve knowledge (analiz bulguları) ayrı index'lerde tutulduğu için ikisi de
ayrı ayrı aranıp birleştiriliyor — böylece az sayıdaki knowledge dokümanı, çok
sayıdaki kayıt arasında hiçbir zaman kaybolmuyor.
"""

import json

import faiss
import numpy as np

from . import config
from .embeddings import embed_query

_record_index: faiss.Index | None = None
_knowledge_index: faiss.Index | None = None
_record_docs: list[dict] | None = None
_knowledge_docs: list[dict] | None = None


def _load():
    global _record_index, _knowledge_index, _record_docs, _knowledge_docs
    if _record_index is None:
        if not config.FAISS_RECORDS_PATH.exists() or not config.FAISS_KNOWLEDGE_PATH.exists():
            raise RuntimeError(
                "Index bulunamadı. Önce 'python -m app.indexing' çalıştırıp index'i kurun."
            )
        _record_index = faiss.read_index(str(config.FAISS_RECORDS_PATH))
        _knowledge_index = faiss.read_index(str(config.FAISS_KNOWLEDGE_PATH))
        with open(config.DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _record_docs = data["records"]
        _knowledge_docs = data["knowledge"]
    return _record_index, _knowledge_index, _record_docs, _knowledge_docs


def _search_one(index: faiss.Index, docs: list[dict], query_vector: np.ndarray, top_k: int) -> list[dict]:
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = docs[idx]
        results.append({
            "text": doc["text"],
            "type": doc["type"],
            "metadata": doc["metadata"],
            "score": float(score),
        })
    return results


def search(query: str, top_k: int = 5, top_k_knowledge: int = 2) -> list[dict]:
    """Soruyla en alakalı kayıtları ve analiz bulgularını ayrı ayrı arayıp birleştirir."""
    record_index, knowledge_index, record_docs, knowledge_docs = _load()

    query_vector = embed_query(query).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    record_results = _search_one(record_index, record_docs, query_vector, top_k)
    knowledge_results = _search_one(knowledge_index, knowledge_docs, query_vector, top_k_knowledge)

    # Knowledge bulgularını önce göster — bunlar genellikle daha "cevap" niteliğinde.
    return knowledge_results + record_results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "which parameters collect passwords"
    print(f"Query: {query}\n")
    for r in search(query):
        print(f"[{r['score']:.3f}] ({r['type']}) {r['text'][:120]}")
