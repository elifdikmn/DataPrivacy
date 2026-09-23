"""FAISS index'leri üzerinde arama yaparak bir soruya en alakalı dokümanları bulma.

Kayıtlar, knowledge (analiz bulguları) ve audit (politika denetimi) ayrı
index'lerde tutulduğu için üçü de ayrı ayrı aranıp birleştiriliyor — böylece
sayıca az olan knowledge/audit dokümanları, çok sayıdaki kayıt arasında
hiçbir zaman kaybolmuyor.
"""

import json

import faiss
import numpy as np

from . import config
from .embeddings import embed_query
from .index_state import index_is_current, MANIFEST_PATH

_loaded_fingerprint: str | None = None
_record_index: faiss.Index | None = None
_knowledge_index: faiss.Index | None = None
_audit_index: faiss.Index | None = None
_record_docs: list[dict] | None = None
_knowledge_docs: list[dict] | None = None
_audit_docs: list[dict] | None = None


def _load():
    global _loaded_fingerprint
    global _record_index, _knowledge_index, _audit_index
    global _record_docs, _knowledge_docs, _audit_docs
    if not index_is_current():
        raise RuntimeError("Retrieval index missing or stale. Run python -m app.indexing, then restart the backend.")
    fingerprint = json.loads(MANIFEST_PATH.read_text())["source_sha256"]
    if _record_index is None or fingerprint != _loaded_fingerprint:
        required = [config.FAISS_RECORDS_PATH, config.FAISS_KNOWLEDGE_PATH, config.FAISS_AUDIT_PATH, config.DOCUMENTS_PATH]
        if not all(p.exists() for p in required):
            raise RuntimeError(
                "Retrieval index not found. Run python -m app.indexing, then restart the backend."
            )
        records = faiss.read_index(str(config.FAISS_RECORDS_PATH))
        knowledge = faiss.read_index(str(config.FAISS_KNOWLEDGE_PATH))
        audit = faiss.read_index(str(config.FAISS_AUDIT_PATH))
        with open(config.DOCUMENTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if any(idx.ntotal != len(data[key]) for idx, key in
               [(records, "records"), (knowledge, "knowledge"), (audit, "audit")]):
            raise RuntimeError("Index/document mismatch. Rebuild the index.")
        _record_index, _knowledge_index, _audit_index = records, knowledge, audit
        _record_docs, _knowledge_docs, _audit_docs = data["records"], data["knowledge"], data["audit"]
        _loaded_fingerprint = fingerprint
    return (
        _record_index, _knowledge_index, _audit_index,
        _record_docs, _knowledge_docs, _audit_docs,
    )


def _search_one(index: faiss.Index, docs: list[dict], query_vector: np.ndarray, top_k: int) -> list[dict]:
    if index.ntotal == 0 or top_k <= 0:
        return []
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))
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


def search(query: str, top_k: int = 5, top_k_knowledge: int = 6, top_k_audit: int = 3) -> list[dict]:
    """Soruyla en alakalı kayıtları, analiz bulgularını ve politika denetimi
    sonuçlarını ayrı ayrı arayıp birleştirir."""
    record_index, knowledge_index, audit_index, record_docs, knowledge_docs, audit_docs = _load()

    query_vector = embed_query(query).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    knowledge_results = _search_one(knowledge_index, knowledge_docs, query_vector, top_k_knowledge)
    audit_results = _search_one(audit_index, audit_docs, query_vector, top_k_audit)
    record_results = _search_one(record_index, record_docs, query_vector, top_k)

    # Knowledge ve audit bulgularını önce göster — bunlar genellikle daha "cevap" niteliğinde.
    return knowledge_results + audit_results + record_results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "which parameters collect passwords"
    print(f"Query: {query}\n")
    for r in search(query):
        print(f"[{r['score']:.3f}] ({r['type']}) {r['text'][:120]}")
