"""Reviewed answers for the demo's fixed questions.

Free-form questions still use retrieval and the LLM. These exact prompts have
stable, fact-backed answers so a rerun cannot swap denominators or overstate
what a small audit or a model score establishes.
"""

from .chart_mapping import _normalise
from .facts import load_facts

_DESCRIPTION_EQUAL_QUESTION = "are sensitive requests explained as often as other requests"
_DESCRIPTION_LESS_QUESTION = "do plugins write descriptions less often for sensitive parameters"

REVIEWED_QUESTIONS = {
    "what data are collected by gpt actions",
    "what percentage of collected data is sensitive",
    "which sensitive data types appear most often",
    _DESCRIPTION_EQUAL_QUESTION,
    _DESCRIPTION_LESS_QUESTION,
    "do plugins disclose what they collect in their privacy policies",
    "are sensitive parameters also the undisclosed ones",
    "what are the model performance confidence intervals",
    "how accurately can a parameter's category be predicted from its name and description",
    "which words predict sensitive categories",
    "do risky vs. safe clusters emerge, and which groups have the highest sensitive-data share",
}


def _pct_interval(bounds: list[float]) -> str:
    return f"{bounds[0] * 100:.1f}–{bounds[1] * 100:.1f}%"


def _f1_interval(bounds: list[float]) -> str:
    return f"{bounds[0]:.3f}–{bounds[1]:.3f}"


def answer_researcher_question(question: str, audience: str) -> dict | None:
    normalized = _normalise(question)
    if audience != "researcher" or normalized not in REVIEWED_QUESTIONS:
        return None
    description_leading = "Not quite" if normalized == _DESCRIPTION_EQUAL_QUESTION else "Yes, slightly"
    if normalized == _DESCRIPTION_EQUAL_QUESTION:
        normalized = _DESCRIPTION_LESS_QUESTION

    facts = load_facts()
    dataset = facts["dataset"]
    sensitive = facts["sensitive_categories"]
    breakdown = sensitive["breakdown"]
    descriptions = facts["description_rate_test"]
    model = facts["category_prediction_model"]
    models = model["uncertainty"]["models"]
    baseline = models["baseline"]
    improved = models["word_char_balanced"]
    audit = facts["privacy_policy_audit"]
    intersection = facts["sensitive_undisclosed_intersection"]
    clusters = facts["clustering"]

    answers = {
        "what data are collected by gpt actions": (
            f"The catalog contains **{dataset['total_records']:,} declared data-request fields** "
            f"across **{dataset['total_plugins']:,} GPT Actions**, grouped into "
            f"{dataset['main_data_type_classes']} broad categories. App usage data, "
            f"Identifier, and Other are the three largest categories "
            f"({sensitive['top3_categories_pct']}% combined); Query and Time follow. "
            "These are Action schema declarations, not evidence that every field was "
            "filled in by a user or transmitted at runtime.",
            "rq1_kategori_dagilimi.md",
        ),
        "what percentage of collected data is sensitive": (
            f"**{sensitive['sensitive_data_count']:,} of {dataset['total_records']:,} fields "
            f"({sensitive['sensitive_data_pct']}%)** fall into this project's four selected "
            "categories: Security credentials, Personal information, Health information, "
            "and Finance information. This is a project-defined subset, not a legal "
            "classification or a complete privacy-risk measure; Location, Identifier, "
            "and Message can also be sensitive in context. The fields describe possible "
            "requests, not confirmed data transmission.",
            "rq1_kategori_dagilimi.md",
        ),
        "which sensitive data types appear most often": (
            f"**Personal information** is the largest of the four selected categories "
            f"({breakdown['Personal information']['count']} fields; "
            f"{breakdown['Personal information']['pct']:.1f}% of all fields), followed by "
            f"Security credentials ({breakdown['Security credentials']['count']}; "
            f"{breakdown['Security credentials']['pct']:.2f}%), Finance information "
            f"({breakdown['Finance information']['count']}; "
            f"{breakdown['Finance information']['pct']:.2f}%), and Health information "
            f"({breakdown['Health information']['count']}; "
            f"{breakdown['Health information']['pct']:.2f}%). These are counts of "
            "declared fields, not counts of users or confirmed disclosures.",
            "rq1_kategori_dagilimi.md",
        ),
        "do plugins write descriptions less often for sensitive parameters": (
            f"**{description_leading}.** {descriptions['sensitive_has_description_pct']:.1f}% "
            f"of fields in the selected sensitive categories have a schema description, "
            f"versus {descriptions['non_sensitive_has_description_pct']:.1f}% of other "
            "fields—a difference of about 3.4 percentage points. The association is "
            f"statistically detectable (p={descriptions['chi_square_p_value']:.4f}) "
            f"but very weak (Cramér's V={descriptions['cramers_v']:.3f}). "
            "Presence does not measure description quality, and a schema description "
            "is not a privacy-policy disclosure. Tests and intervals assume independent "
            "parameter records, although fields from one Action may be related.",
            "rq2_aciklama_orani.md",
        ),
        "do plugins disclose what they collect in their privacy policies": (
            f"In the separate audit, **{audit['undisclosed_count']} of "
            f"{audit['total_parameters_with_comparable_policy_text']} comparable "
            f"fields ({audit['undisclosed_pct']:.1f}%)** had no disclosure in the "
            "corresponding privacy-policy text. Of the 30 not completely omitted, "
            "16 were clearly disclosed and 14 were vague, ambiguous, or incorrect. Comparable policy "
            "text was available for 184 Actions, so this is a non-random subset, not "
            "an ecosystem-wide estimate. The audit concerns declared requests in "
            "Action schemas; it does not observe runtime transmission.",
            "rq6_policy_audit.md",
        ),
        "are sensitive parameters also the undisclosed ones": (
            f"**{intersection['undisclosed_count']} of "
            f"{intersection['sensitive_count']}** audited fields unambiguously mapped "
            "to this project's selected sensitive categories were undisclosed "
            f"({intersection['undisclosed_pct_of_sensitive']:.1f}%; 95% Wilson interval "
            f"{_pct_interval(intersection['ci95_wilson_of_sensitive'])}). The overall "
            f"audited rate was {audit['undisclosed_count']}/"
            f"{audit['total_parameters_with_comparable_policy_text']} "
            f"({audit['undisclosed_pct']:.1f}%). These rates are similar in this small, "
            "non-random audit, but the remaining fields cannot all be classified as "
            "non-sensitive, so this does not establish whether sensitivity changes "
            "disclosure. The hand mapping excludes ambiguous labels, and the audit "
            "contains no Health-information label.",
            "rq7_hassas_ve_ifsa_edilmemis.md",
        ),
        "what are the model performance confidence intervals": (
            "On the 25-category record-level test set, the word-only TF-IDF + "
            f"Logistic Regression baseline reached **{baseline['accuracy']['estimate'] * 100:.1f}% "
            f"accuracy** (95% CI {_pct_interval(baseline['accuracy']['ci95'])}) and "
            f"macro F1 {baseline['macro_f1']['estimate']:.3f} "
            f"(CI {_f1_interval(baseline['macro_f1']['ci95'])}). The validation-selected "
            f"word-plus-character model with balanced class weights reached "
            f"**{improved['accuracy']['estimate'] * 100:.1f}% accuracy** "
            f"(CI {_pct_interval(improved['accuracy']['ci95'])}) and macro F1 "
            f"{improved['macro_f1']['estimate']:.3f} "
            f"(CI {_f1_interval(improved['macro_f1']['ci95'])}). These bootstrap "
            "intervals condition on the fitted models and this particular split; "
            "they do not include retraining uncertainty or establish performance "
            "on entirely unseen Actions.",
            "rq3_model_performansi.md",
        ),
        "how accurately can a parameter's category be predicted from its name and description": (
            f"For the 25 broad categories, the validation-selected word-plus-character "
            f"TF-IDF + Logistic Regression model achieved **{improved['accuracy']['estimate'] * 100:.1f}% "
            f"accuracy** and macro F1 {improved['macro_f1']['estimate']:.3f} "
            f"on {model['test_size']:,} held-out parameter records. The word-only "
            f"baseline reached {baseline['accuracy']['estimate'] * 100:.1f}% accuracy "
            f"and macro F1 {baseline['macro_f1']['estimate']:.3f}. Performance varies "
            "by category; rare classes remain harder to recognize. Most test-set "
            "Actions also appear in training through different fields, so the "
            "result should not be presented as accuracy on entirely new Actions.",
            "rq3_model_performansi.md",
        ),
        "which words predict sensitive categories": (
            "In the **word-only baseline model**, positive Logistic Regression "
            "coefficients for Security credentials include key, api_key, token, and "
            "password; for Personal information they include email, gender, "
            "firstname, and birthday. Health and Finance rankings also contain "
            "generic phrases, so those coefficients require caution. These are "
            "associations within this fitted model, not proof of what developers "
            "intended, a causal effect, or the word ranking of the improved "
            "word-plus-character model. Individual category predictions still "
            "need evaluation against reviewed labels.",
            "rq3_model_performansi.md",
        ),
        "do risky vs. safe clusters emerge, and which groups have the highest sensitive-data share": (
            f"**No clear risky-versus-safe split emerged.** K-Means grouped "
            f"{clusters['total_eligible_plugins']:,} Actions with at least "
            f"{clusters['min_params_threshold']} distinct fields by their "
            f"25-category request profiles. K={clusters['best_k']} gave the "
            f"highest silhouette score ({clusters['silhouette_score']:.3f}) "
            "**within the tested range of K=2–10**; it is not a proven global "
            "optimum. The personal-information/messaging group "
            f"({clusters['cluster_summary']['2']['n_plugins']} Actions; "
            f"{clusters['cluster_summary']['2']['sensitive_pct']:.1f}% selected-sensitive "
            "share) and security-credentials/app-usage group "
            f"({clusters['cluster_summary']['1']['n_plugins']} Actions; "
            f"{clusters['cluster_summary']['1']['sensitive_pct']:.1f}%) rank highest "
            "on that measure. The share covers only four project-selected "
            "categories, so it is **not a validated risk score** and a low share "
            "does not establish safety.",
            "rq4_kumeleme.md",
        ),
    }

    response, filename = answers[normalized]
    return {
        "answer": response,
        "sources": [{
            "text": "Reviewed project result and its scope and limitations.",
            "type": "finding",
            "metadata": {"filename": filename},
            "score": 1.0,
        }],
    }


def answer_general_question(question: str, audience: str) -> dict | None:
    """Give the shared demo questions the same findings in shorter language."""
    normalized = _normalise(question)
    if audience != "general":
        return None
    description_leading = "Not quite" if normalized == _DESCRIPTION_EQUAL_QUESTION else "Yes, slightly"
    if normalized == _DESCRIPTION_EQUAL_QUESTION:
        normalized = _DESCRIPTION_LESS_QUESTION

    facts = load_facts()
    dataset = facts["dataset"]
    sensitive = facts["sensitive_categories"]
    breakdown = sensitive["breakdown"]
    descriptions = facts["description_rate_test"]
    audit = facts["privacy_policy_audit"]
    intersection = facts["sensitive_undisclosed_intersection"]
    answers = {
        "what data are collected by gpt actions": (
            f"The dataset lists **{dataset['total_records']:,} fields** that "
            f"{dataset['total_plugins']:,} GPT Actions can request. They cover "
            f"{dataset['main_data_type_classes']} categories; App usage data, "
            "Identifier, and Other are the largest. These are possible requests, "
            "not proof that users sent the data.",
            "rq1_kategori_dagilimi.md",
        ),
        "what percentage of collected data is sensitive": (
            f"**{sensitive['sensitive_data_count']:,} of {dataset['total_records']:,} "
            f"fields ({sensitive['sensitive_data_pct']}%)** fall into four categories "
            "we selected as sensitive: Security credentials, Personal information, "
            "Health information, and Finance information. This is our project's "
            "definition, not a complete measure of privacy risk.",
            "rq1_kategori_dagilimi.md",
        ),
        "which sensitive data types appear most often": (
            "Of the four categories we selected, **Personal information** "
            f"appears most often ({breakdown['Personal information']['count']} fields), "
            f"followed by Security credentials ({breakdown['Security credentials']['count']}), "
            f"Finance information ({breakdown['Finance information']['count']}), and "
            f"Health information ({breakdown['Health information']['count']}). "
            "These are fields an Action can request, not confirmed user submissions.",
            "rq1_kategori_dagilimi.md",
        ),
        "do plugins write descriptions less often for sensitive parameters": (
            f"**{description_leading}.** Sensitive fields have descriptions slightly less often "
            f"than other fields ({descriptions['sensitive_has_description_pct']:.1f}% "
            f"versus {descriptions['non_sensitive_has_description_pct']:.1f}%). "
            "The difference is small. This checks whether a field has a "
            "description, not whether its privacy policy explains the request.",
            "rq2_aciklama_orani.md",
        ),
        "do plugins disclose what they collect in their privacy policies": (
            f"In the smaller policy audit, **{audit['undisclosed_count']} of "
            f"{audit['total_parameters_with_comparable_policy_text']} fields "
            f"({audit['undisclosed_pct']:.1f}%)** were not disclosed in the "
            "corresponding privacy policies. This finding is about the audited "
            "group, not every GPT Action.",
            "rq6_policy_audit.md",
        ),
        "are sensitive parameters also the undisclosed ones": (
            f"**Yes, in this small audit:** {intersection['undisclosed_count']} of "
            f"{intersection['sensitive_count']} fields we could clearly map "
            "to our selected sensitive categories were not disclosed in their "
            "privacy policies. The sample is too small to say whether sensitive "
            "fields are treated differently from other fields.",
            "rq7_hassas_ve_ifsa_edilmemis.md",
        ),
    }
    if normalized not in answers:
        return None

    response, filename = answers[normalized]
    return {
        "answer": response,
        "sources": [{
            "text": "Reviewed project result and its scope and limitations.",
            "type": "finding",
            "metadata": {"filename": filename},
            "score": 1.0,
        }],
    }
