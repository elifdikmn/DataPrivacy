"""Uygulama ayarları — .env dosyasından okunur."""

import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def parse_cors_origins(value: str) -> list[str]:
    """Read exact browser origins; an origin has no path or trailing slash."""
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


CORS_ORIGINS = parse_cors_origins(
    os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
)

# /ask istek sınırları (0 = kapalı). İstemci başına dakika/gün ve tüm istemciler için
# günlük toplam; toplam sınır en kötü durumda günlük LLM maliyetini sınırlar.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "8"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "100"))
GLOBAL_DAILY_LIMIT = int(os.getenv("GLOBAL_DAILY_LIMIT", "500"))

# Embedding: yerel, ücretsiz bir sentence-transformers modeli (Hugging Face'ten
# ilk çalıştırmada indirilir, sonrasında yerel önbellekten kullanılır).
# Çok dilli model kullanıyoruz çünkü sorular Türkçe, veri kayıtları İngilizce
# teknik terimlerle (örn. "password", "api_key") — tek dilli bir İngilizce model
# (all-MiniLM-L6-v2) bu ikisini semantik olarak eşleştiremiyor.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

DATA_PATH = BACKEND_DIR / os.getenv("DATA_PATH", "data/data_entries_final.json")
KNOWLEDGE_DIR = BACKEND_DIR / os.getenv("KNOWLEDGE_DIR", "knowledge")
FINAL_RESULTS_DIR = BACKEND_DIR / os.getenv("FINAL_RESULTS_DIR", "final_results")

STATIC_DIR = BACKEND_DIR / "static"

INDEX_DIR = BACKEND_DIR / "app" / "index_store"
FAISS_RECORDS_PATH = INDEX_DIR / "records.faiss"
FAISS_KNOWLEDGE_PATH = INDEX_DIR / "knowledge.faiss"
FAISS_AUDIT_PATH = INDEX_DIR / "audit.faiss"
DOCUMENTS_PATH = INDEX_DIR / "documents.json"
