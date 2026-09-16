"""Shared definitions for notebook analysis and retrieval documents."""
SENSITIVE_CATEGORIES = ['Security credentials', 'Personal information', 'Health information', 'Finance information']
DISCLOSURE_PRIORITY = [('CLEAR', 'DISCLOSED_CLEAR'), ('VAGUE', 'DISCLOSED_VAGUE'),
                       ('AMBIGUOUS', 'DISCLOSED_AMBIGUOUS'), ('INCORRECT', 'DISCLOSED_INCORRECT')]

def unique_action_ids(values):
    """Preserve first-occurrence order; repeated IDs are not distinct Actions."""
    return list(dict.fromkeys(values or []))

def disclosure_status(item):
    collection = item.get('collection') or []
    by_label = {}
    for sentence in collection:
        by_label.setdefault(sentence.get('label'), sentence.get('sentence'))
    for label, status in DISCLOSURE_PRIORITY:
        if label in by_label:
            return status, by_label[label]
    return ('UNDISCLOSED' if collection else 'NO_POLICY_TEXT'), None

def unambiguous_type_mapping(frame):
    """Do not turn a many-to-many taxonomy relationship into a majority label."""
    grouped = frame.groupby('data_type')['main_data_type'].agg(lambda s: sorted(set(s)))
    return {kind: labels[0] for kind, labels in grouped.items() if len(labels) == 1}
