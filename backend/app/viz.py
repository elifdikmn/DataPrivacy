"""Sorguya göre bulunan kayıtların kategori dağılımını gösteren grafik verisi üretimi."""

from collections import Counter

SENSITIVE_CATEGORIES = {
    "Security credentials",
    "Personal information",
    "Health information",
    "Finance information",
}


def sources_category_chart(sources: list[dict]) -> list[dict] | None:
    """Bulunan kayıtların (record tipindeki) main_data_type dağılımını, frontend'in
    Recharts ile çizeceği düz bir veri listesine çevirir. Hiç record yoksa None döner."""
    counts = Counter(
        s["metadata"].get("main_data_type", "Unknown")
        for s in sources
        if s["type"] == "record"
    )
    if not counts:
        return None

    return [
        {"category": category, "count": count, "sensitive": category in SENSITIVE_CATEGORIES}
        for category, count in counts.most_common()
    ]
