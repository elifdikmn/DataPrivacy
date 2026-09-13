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


SYSTEM_PROMPT = """Sen, GPT eklentilerinin (GPT Actions) topladığı verileri ve bunların \
gizlilik risklerini analiz eden bir asistansın. Sana verilen BAĞLAM içindeki veri \
kayıtlarına ve analiz bulgularına dayanarak kullanıcının sorusunu cevapla.

Kurallar:
- Sadece BAĞLAM'daki bilgiye dayan; BAĞLAM'da olmayan bir şeyi uydurma.
- BAĞLAM soruyu cevaplamaya yetmiyorsa bunu açıkça söyle.
- Kısa ve net cevap ver, gereksiz uzatma."""


def ask(question: str, context: str) -> str:
    """BAĞLAM metnini ve kullanıcı sorusunu Claude'a gönderip cevabı döndürür."""
    response = get_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"BAĞLAM:\n{context}\n\nSORU: {question}",
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
