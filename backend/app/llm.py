"""Anthropic Claude ile RAG cevabı üretme."""

import anthropic

from . import config

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY tanımlı değil. backend/.env dosyasını "
                "backend/.env.example'dan kopyalayıp anahtarınızı girin."
            )
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """You are an assistant that analyzes what data GPT Actions (custom GPT \
plugins) collect and the privacy risks involved. Answer the user's question based on the \
CONTEXT you are given, which contains data records and analysis findings.

Rules:
- Base your answer only on the CONTEXT; do not make up anything not in it.
- If the CONTEXT is not sufficient to answer the question, say so explicitly.
- Always answer in English, even though the CONTEXT itself may contain Turkish text \
(the analysis findings were originally written in Turkish) — translate/summarize as needed.
- Keep the answer short and to the point."""


def ask(question: str, context: str) -> str:
    """Sends the CONTEXT text and the user's question to Claude and returns the answer."""
    response = get_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            }
        ],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def test_connection() -> str:
    """Basit bir çağrı ile API anahtarının ve modelin doğru çalıştığını kontrol eder."""
    response = get_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "Tek kelimeyle cevap ver: 2+2 kaçtır?"}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


if __name__ == "__main__":
    print("Claude API bağlantısı test ediliyor...")
    try:
        print("Cevap:", test_connection())
        print("Bağlantı başarılı.")
    except RuntimeError as e:
        print(f"HATA: {e}")
    except anthropic.AuthenticationError:
        print("HATA: API anahtarı geçersiz. backend/.env dosyasındaki ANTHROPIC_API_KEY'i kontrol edin.")
    except anthropic.APIConnectionError:
        print("HATA: Ağ bağlantısı kurulamadı.")
    except anthropic.APIStatusError as e:
        print(f"HATA: API {e.status_code} döndürdü: {e.message}")
