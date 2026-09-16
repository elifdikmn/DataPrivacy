"""Veri + bilgi tabanı + politika denetimi metinlerini embedding'e çevirip FAISS index kurma.

Üç ayrı doküman türü, üç ayrı FAISS index'te tutuluyor (kayıtlar, knowledge,
audit). Aksi halde retrieval sırasında sayıca az olan knowledge/audit
dokümanları, çok sayıdaki kısa kayıt arasında kaybolabiliyor.
"""

import json

import faiss
import numpy as np

from . import config
from .index_state import MANIFEST_PATH, source_fingerprint
from .domain import disclosure_status, unique_action_ids
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
        plugin_count = len(unique_action_ids(record.get("plugin_id_filenames")))

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


def _plugin_name_from_filename(filename: str) -> str:
    # "domain.com , Plugin Title.json" -> "Plugin Title"
    stem = filename[:-5] if filename.endswith(".json") else filename
    parts = stem.split(" , ", 1)
    return parts[1] if len(parts) == 2 else stem


def _item_disclosure_status(item: dict) -> tuple[str, str | None]:
    return disclosure_status(item)


def build_audit_documents() -> list[dict]:
    """final_results/ altındaki, gizlilik politikası karşılaştırmalı etiketli
    verilerden 'bu veri toplanıyor ama politikada var mı yok mu' dokümanları üretir."""
    documents = []

    for path in sorted(config.FINAL_RESULTS_DIR.glob("*.json")):
        plugin_name = _plugin_name_from_filename(path.name)
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        for item in items:
            status, sentence = _item_disclosure_status(item)
            if status == "NO_POLICY_TEXT":
                continue  # bu parametre için hiç karşılaştırılabilir politika metni yok

            data_name = item.get("data_name") or ""
            data_type = item.get("data_type") or ""
            description = item.get("description") or ""

            if status == "UNDISCLOSED":
                disclosure_text = "is NOT disclosed anywhere in the plugin's privacy policy"
            else:
                quality = {"DISCLOSED_CLEAR": "clearly", "DISCLOSED_VAGUE": "vaguely", "DISCLOSED_AMBIGUOUS": "ambiguously", "DISCLOSED_INCORRECT": "incorrectly"}[status]
                disclosure_text = f'is disclosed ({quality}) in the privacy policy: "{sentence}"'

            text = (
                f"Plugin '{plugin_name}' collects '{data_name}' ({data_type}). "
                f"Privacy policy audit: this data collection {disclosure_text}."
            )
            documents.append({
                "text": text,
                "type": "audit",
                "metadata": {
                    "plugin": plugin_name,
                    "data_name": data_name,
                    "data_type": data_type,
                    "description": description,
                    "status": status,
                },
            })

    return documents


def _build_faiss_index(texts: list[str]) -> faiss.Index:
    embeddings = embed_texts(texts)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_index():
    MANIFEST_PATH.unlink(missing_ok=True)
    initial_fingerprint = source_fingerprint()
    print("Dokümanlar hazırlanıyor...")
    record_docs = build_record_documents()
    knowledge_docs = build_knowledge_documents()
    audit_docs = build_audit_documents()
    print(f"Kayıt dokümanı: {len(record_docs)}, knowledge parçası: {len(knowledge_docs)}, "
          f"audit (politika denetimi) dokümanı: {len(audit_docs)}")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Kayıtlar için embedding hesaplanıyor (bu biraz zaman alabilir)...")
    record_index = _build_faiss_index([d["text"] for d in record_docs])
    faiss.write_index(record_index, str(config.FAISS_RECORDS_PATH))

    print("Knowledge parçaları için embedding hesaplanıyor...")
    knowledge_index = _build_faiss_index([d["text"] for d in knowledge_docs])
    faiss.write_index(knowledge_index, str(config.FAISS_KNOWLEDGE_PATH))

    print("Audit dokümanları için embedding hesaplanıyor...")
    audit_index = _build_faiss_index([d["text"] for d in audit_docs])
    faiss.write_index(audit_index, str(config.FAISS_AUDIT_PATH))

    with open(config.DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"records": record_docs, "knowledge": knowledge_docs, "audit": audit_docs},
            f, ensure_ascii=False,
        )

    if source_fingerprint() != initial_fingerprint:
        raise RuntimeError("Sources changed during indexing; rebuild the index.")
    MANIFEST_PATH.write_text(json.dumps({"source_sha256": initial_fingerprint}))

    print(f"Kayıt index'i kaydedildi: {config.FAISS_RECORDS_PATH} ({record_index.ntotal} vektör)")
    print(f"Knowledge index'i kaydedildi: {config.FAISS_KNOWLEDGE_PATH} ({knowledge_index.ntotal} vektör)")
    print(f"Audit index'i kaydedildi: {config.FAISS_AUDIT_PATH} ({audit_index.ntotal} vektör)")
    print(f"Dokümanlar kaydedildi: {config.DOCUMENTS_PATH}")


if __name__ == "__main__":
    build_index()
