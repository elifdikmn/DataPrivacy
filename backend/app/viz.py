"""Sorguya göre bulunan kayıtların kategori dağılımını gösteren grafik üretimi."""

from collections import Counter

import plotly.graph_objects as go


def sources_category_chart(sources: list[dict]) -> str | None:
    """Bulunan kayıtların (record tipindeki) main_data_type dağılımını bar chart olarak
    Plotly JSON string'ine çevirir. Hiç record yoksa None döner."""
    counts = Counter(
        s["metadata"].get("main_data_type", "Unknown")
        for s in sources
        if s["type"] == "record"
    )
    if not counts:
        return None

    categories = list(counts.keys())
    values = list(counts.values())

    fig = go.Figure(data=[go.Bar(x=categories, y=values)])
    fig.update_layout(
        title="Data categories among the retrieved results",
        xaxis_title="Category",
        yaxis_title="Number of matched records",
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig.to_json()
