"""Veri + bilgi tabanı metinlerini embedding'e çevirip FAISS index kurma.

Kayıtlar (12.811 tanesi) ve knowledge dosyaları (5 tanesi) ayrı FAISS index'lerinde
tutuluyor. Aksi halde retrieval sırasında az sayıdaki knowledge dokümanı, çok
sayıdaki kısa kayıt arasında kaybolabiliyor (aynı havuzda top-k'ya giremiyor).
"""

import json

import faiss
import numpy as np

from . import config
from .embeddings import embed_texts


def build_record_documents() -> list[dict]:
    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    documents = []
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
    return documents


def _chunk_knowledge_file(text: str) -> list[str]:
    """Bir knowledge dosyasını '## ' başlıklarına göre bölümlere ayırır.

    Her bölüm, hangi dosyaya ait olduğunu kaybetmemesi için dosyanın başlığını
    (# ...) da içinde taşır — böylece embedding tek başına o bölümü temsil eder,
    ama bağlamdan da kopmaz.
    """
    lines = text.strip().split("\n")
    title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else ""

    chunks = []
    current_header = None
    current_lines = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            header_part = f"{current_header}\n" if current_header else ""
            chunks.append(f"{title}\n\n{header_part}{body}")

    for line in lines[1:]:
        if line.startswith("## "):
            flush()
            current_header = line
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    return chunks or [text]


def build_knowledge_documents() -> list[dict]:
    documents = []
    for path in sorted(config.KNOWLEDGE_DIR.glob("rq*.md")):
        full_text = path.read_text(encoding="utf-8")
        for chunk in _chunk_knowledge_file(full_text):
            documents.append({
                "text": chunk,
                "type": "knowledge",
                "metadata": {"filename": path.name},
            })
    return documents


def _build_faiss_index(texts: list[str]) -> faiss.Index:
    embeddings = embed_texts(texts)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_index():
    print("Dokümanlar hazırlanıyor...")
    record_docs = build_record_documents()
    knowledge_docs = build_knowledge_documents()
    print(f"Kayıt dokümanı: {len(record_docs)}, knowledge parçası (bölüm bazlı): {len(knowledge_docs)}")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Kayıtlar için embedding hesaplanıyor (bu biraz zaman alabilir)...")
    record_index = _build_faiss_index([d["text"] for d in record_docs])
    faiss.write_index(record_index, str(config.FAISS_RECORDS_PATH))

    print("Knowledge parçaları için embedding hesaplanıyor...")
    knowledge_index = _build_faiss_index([d["text"] for d in knowledge_docs])
    faiss.write_index(knowledge_index, str(config.FAISS_KNOWLEDGE_PATH))

    with open(config.DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"records": record_docs, "knowledge": knowledge_docs}, f, ensure_ascii=False)

    print(f"Kayıt index'i kaydedildi: {config.FAISS_RECORDS_PATH} ({record_index.ntotal} vektör)")
    print(f"Knowledge index'i kaydedildi: {config.FAISS_KNOWLEDGE_PATH} ({knowledge_index.ntotal} vektör)")
    print(f"Dokümanlar kaydedildi: {config.DOCUMENTS_PATH}")


if __name__ == "__main__":
    build_index()
