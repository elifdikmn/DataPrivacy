"""Doğrulanmış sayısal gerçekler (project_facts.json) — LLM'in sayıları ham
kayıtlardan kendi kafasında hesaplamasını/tahmin etmesini önlemek için her
sorunun context'ine ekleniyor, cevap sonrası da basit bir doğrulama yapılıyor."""

import json
import logging
import re
from functools import lru_cache

from . import config

logger = logging.getLogger("chatbot.facts")

FACTS_PATH = config.APP_DIR / "project_facts.json"


@lru_cache(maxsize=1)
def load_facts() -> dict:
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_facts_block() -> str:
    """Tüm facts tablosunu (küçük olduğu için tamamını) context'e eklenecek
    düz metin bloğuna çevirir."""
    facts = load_facts()
    return "FACTS (verified numbers — copy exactly, do not recompute):\n" + json.dumps(facts, indent=2)


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _flatten_numbers(obj) -> set[str]:
    """FACTS içindeki tüm sayısal değerleri düz bir string kümesine çevirir
    (12811, 7.3, 0.85, ... gibi), karşılaştırma için."""
    numbers: set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            numbers |= _flatten_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            numbers |= _flatten_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        numbers.add(f"{obj:g}")
    return numbers


def verify_answer_numbers(answer_text: str) -> list[str]:
    """Cevap metnindeki sayıları FACTS'teki değerlerle karşılaştırır. FACTS'te
    hiç geçmeyen sayıları döner ve konsola uyarı loglar — cevabı engellemez,
    sadece görünür kılar (erken hata yakalama için).

    Tek haneli sayılar (0-9) — "4 kategori", "25 sınıf" gibi çoğunlukla sabit/
    yapısal referanslar — gürültüyü azaltmak için hariç tutuluyor.
    """
    known_numbers = _flatten_numbers(load_facts())
    found = set(_NUMBER_RE.findall(answer_text))

    def is_trivial(n: str) -> bool:
        # Only whole numbers below 10 are treated as trivial/structural (e.g. "4
        # categories", "25 classes") — decimals like F1 scores (0.61) or small
        # percentages must still be checked, since those are exactly the kind
        # of number that gets misattributed.
        return "." not in n and abs(int(n)) < 10

    unknown = sorted(n for n in found if n not in known_numbers and not is_trivial(n))

    if unknown:
        logger.warning(
            "Answer contains numbers not found in project_facts.json (possible "
            "hallucination/miscalculation): %s",
            unknown,
        )
    return unknown
