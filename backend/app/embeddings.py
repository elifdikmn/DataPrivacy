"""sentence-transformers ile yerel, ücretsiz embedding hesaplama.

İlk çalıştırmada model Hugging Face'ten indirilir (küçük, ~80MB), sonraki
çalıştırmalarda yerel önbellekten (~/.cache) kullanılır — API anahtarı ya da
internet bağlantısı gerektirmez (ilk indirme hariç).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Embedding modeli yükleniyor: {config.EMBEDDING_MODEL_NAME} (ilk seferde indirilir)...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Metin listesini embedding vektörlerine çevirir. (n_texts, boyut) şeklinde döner."""
    model = get_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """Tek bir sorguyu embedding'e çevirir. (boyut,) şeklinde döner."""
    return embed_texts([text])[0]
