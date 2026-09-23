"""Soru → sabit görsel eşleştirme tablosu.

Chatbot artık soruya göre canlı grafik üretmiyor; bunun yerine, önceden
notebook'lardan (veya elle) hazırlanmış PNG'lerden birini, sorunun bu
sabit listedeki hangi soruyla eşleştiğine bakarak gösteriyor. Eşleşme
yoksa hiç görsel gösterilmiyor.
"""

import re

QUESTION_CHART_MAP: dict[str, str] = {
    "what are the model performance confidence intervals?": "rq3_validation_confidence_intervals.png",
    "what data are collected by gpt actions?": "treemap_main_categories.html",
    "what percentage of collected data is sensitive?": "rq1_category_distribution.png",
    "which sensitive data types appear most often?": "rq1_sensitive_breakdown.png",
    "do plugins write descriptions less often for sensitive parameters?": "rq2_description_rate.png",
    "how accurately can a parameter's category be predicted from its name?": "rq3_confusion_matrix.png",
    "which words predict sensitive categories?": "rq3_feature_importance.png",
    "do natural risky vs. safe clusters emerge among plugins?": "rq4_cluster_sensitivity.png",
    "which plugin clusters have the highest sensitive-data share?": "rq4_cluster_sensitivity.png",
    'can mislabeled "other" records be identified automatically?': "rq5_other_confidence_distribution.png",
    'are there hidden sensitive parameters mislabeled as "other"?': "rq5_other_confidence_distribution.png",
    "which parameters collect passwords?": "rq3_password_breakdown.png",
    "do plugins disclose what they collect in their privacy policies?": "rq6_policy_disclosure.png",
    "are sensitive parameters also the undisclosed ones?": "rq7_sensitive_undisclosed.png",
}

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _normalise(question: str) -> str:
    """Büyük/küçük harf, fazla boşluk, tipografik tırnak ve sondaki noktalama farkını yok sayar."""
    text = re.sub(r"\s+", " ", question.translate(_QUOTES)).strip().lower()
    return text.rstrip(" ?!.")


_NORMALISED_MAP = {_normalise(q): chart for q, chart in QUESTION_CHART_MAP.items()}


def get_chart_filename(question: str) -> str | None:
    """Soru metnini sabit tabloda arar. Eşleşme yoksa None döner —
    serbest metin sorularında görsel yok."""
    return _NORMALISED_MAP.get(_normalise(question))
