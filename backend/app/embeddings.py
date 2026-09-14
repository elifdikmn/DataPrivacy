"""Together API ile metinleri embedding'e çevirme (OpenAI uyumlu endpoint üzerinden)."""

import numpy as np
from openai import OpenAI

from . import config

BATCH_SIZE = 96

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.TOGETHER_API_KEY:
            raise RuntimeError(
                "TOGETHER_API_KEY tanımlı değil. backend/.env dosyasını kontrol edin."
            )
        if not config.TOGETHER_EMBEDDING_MODEL:
            raise RuntimeError(
                "TOGETHER_EMBEDDING_MODEL tanımlı değil. Together'ın güncel embedding "
                "modelleri listesinden (api.together.ai/models) bir model adı seçip "
                "backend/.env dosyasına yazın."
            )
        _client = OpenAI(api_key=config.TOGETHER_API_KEY, base_url=config.TOGETHER_BASE_URL)
    return _client


def embed_texts(texts: list[str]) -> np.ndarray:
    """Metin listesini embedding vektörlerine çevirir. (n_texts, boyut) şeklinde döner."""
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=config.TOGETHER_EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedding: {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    return np.array(all_embeddings, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Tek bir sorguyu embedding'e çevirir. (boyut,) şeklinde döner."""
    return embed_texts([text])[0]
