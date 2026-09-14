"""Veri + bilgi tabanı metinlerini embedding'e çevirip FAISS index kurma."""

import json

import faiss
import numpy as np

from . import config
from .embeddings import embed_texts


def build_documents() -> list[dict]:
    """Ham veri kayıtlarını ve knowledge/ özet metinlerini tek bir doküman listesine çevirir.

    Her doküman: {"text": embed edilecek metin, "type": "record" | "knowledge", "metadata": {...}}
    """
    documents = []

    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    for record in records:
        name = record.get("name") or ""
        description = record.get("description") or ""
        main_type = record.get("main_data_type") or ""
        data_type = record.get("data_type") or ""
        plugin_count = len(record.get("plugin_id_filenames") or [])

        text = f"Parameter: {name}. Description: {description}. Category: {main_type} / {data_type}."
        documents.append({
            "text": text,
            "type": "record",
            "metadata": {
                "name": name,
                "description": description,
                "main_data_type": main_type,
                "data_type": data_type,
                "plugin_count": plugin_count,
            },
        })

    for path in sorted(config.KNOWLEDGE_DIR.glob("rq*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append({
            "text": text,
            "type": "knowledge",
            "metadata": {"filename": path.name},
        })

    return documents


def build_index():
    print("Dokümanlar hazırlanıyor...")
    documents = build_documents()
    print(f"Toplam doküman: {len(documents)} ({sum(1 for d in documents if d['type'] == 'record')} kayıt, "
          f"{sum(1 for d in documents if d['type'] == 'knowledge')} bilgi tabanı dosyası)")

    print("Embedding hesaplanıyor (bu biraz zaman alabilir)...")
    texts = [d["text"] for d in documents]
    embeddings = embed_texts(texts)

    # Cosine benzerliği için vektörleri normalize edip inner-product index kullanıyoruz.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)

    print(f"Index kaydedildi: {config.FAISS_INDEX_PATH}")
    print(f"Dokümanlar kaydedildi: {config.DOCUMENTS_PATH}")
    print(f"Vektör boyutu: {dimension}, toplam vektör: {index.ntotal}")


if __name__ == "__main__":
    build_index()
