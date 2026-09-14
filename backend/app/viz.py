"""Sorguya göre bulunan kayıtların kategori/alt-kategori dağılımını, frontend'in
Recharts Treemap ile çizeceği iç içe bir veri yapısına çevirme."""

from collections import defaultdict

SENSITIVE_CATEGORIES = {
    "Security credentials",
    "Personal information",
    "Health information",
    "Finance information",
}


def sources_category_treemap(sources: list[dict]) -> list[dict] | None:
    """Bulunan kayıtların (record tipindeki) main_data_type / data_type dağılımını,
    her ana kategorinin altında alt kategori düğümleri taşıyan bir treemap veri
    yapısına çevirir. Hiç record yoksa None döner."""
    tree: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for s in sources:
        if s["type"] != "record":
            continue
        m = s["metadata"]
        main_type = m.get("main_data_type") or "Unknown"
        sub_type = m.get("data_type") or "Unknown"
        tree[main_type][sub_type] += 1

    if not tree:
        return None

    nodes = []
    for main_type, subs in tree.items():
        children = [
            {"name": sub_type, "size": count}
            for sub_type, count in sorted(subs.items(), key=lambda kv: -kv[1])
        ]
        nodes.append({
            "name": main_type,
            "sensitive": main_type in SENSITIVE_CATEGORIES,
            "size": sum(c["size"] for c in children),
            "children": children,
        })

    nodes.sort(key=lambda n: -n["size"])
    return nodes
