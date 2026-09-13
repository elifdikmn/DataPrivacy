"""Uygulama ayarları — .env dosyasından okunur."""

import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_EMBEDDING_MODEL = os.getenv("TOGETHER_EMBEDDING_MODEL", "")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

DATA_PATH = BACKEND_DIR / os.getenv("DATA_PATH", "data/data_entries_final.json")
KNOWLEDGE_DIR = BACKEND_DIR / os.getenv("KNOWLEDGE_DIR", "knowledge")
