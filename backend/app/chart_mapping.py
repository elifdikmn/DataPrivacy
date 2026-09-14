"""Soru → sabit görsel eşleştirme tablosu.

Chatbot artık soruya göre canlı grafik üretmiyor; bunun yerine, önceden
notebook'lardan (veya elle) hazırlanmış PNG'lerden birini, sorunun bu
sabit listedeki hangi soruyla eşleştiğine bakarak gösteriyor. Eşleşme
yoksa hiç görsel gösterilmiyor.
"""

QUESTION_CHART_MAP: dict[str, str] = {
    "what data are collected by gpt actions?": "treemap_main_categories.png",
    "what percentage of collected data is sensitive?": "rq1_category_distribution.png",
    "which sensitive data types appear most often?": "treemap_with_subtypes.png",
    "do plugins write descriptions less often for sensitive parameters?": "rq2_description_rate.png",
    "how accurately can a parameter's category be predicted from its name?": "rq3_confusion_matrix.png",
    "which words predict sensitive categories?": "rq3_feature_importance.png",
    "do natural risky vs. safe clusters emerge among plugins?": "rq4_silhouette_scores.png",
    "which plugin clusters have the highest sensitive-data share?": "rq4_cluster_sensitivity.png",
    'can mislabeled "other" records be identified automatically?': "rq5_other_confidence_distribution.png",
    'are there hidden sensitive parameters mislabeled as "other"?': "rq5_other_confidence_distribution.png",
    "which parameters collect passwords?": "rq3_password_breakdown.png",
    "do plugins disclose what they collect in their privacy policies?": "rq6_policy_disclosure.png",
}


def get_chart_filename(question: str) -> str | None:
    """Soru metnini (baş/son boşluk ve büyük/küçük harf farkı yok sayılarak) sabit
    tabloda arar. Eşleşme yoksa None döner — serbest metin sorularında görsel yok."""
    return QUESTION_CHART_MAP.get(question.strip().lower())
