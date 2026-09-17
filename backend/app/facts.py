"""Project facts, optional legacy fact rendering and nonblocking numeric diagnostics."""
import json
import re
from decimal import Decimal
from functools import lru_cache
from . import config

FACTS_PATH = config.APP_DIR / 'project_facts.json'

@lru_cache(maxsize=1)
def load_facts():
    return json.loads(FACTS_PATH.read_text(encoding='utf-8'))

_UNSET = object()

def numeric_facts(obj=_UNSET, prefix=''):
    if obj is _UNSET:
        obj = load_facts()
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not key.startswith('_'):
                result.update(numeric_facts(value, f'{prefix}.{key}' if prefix else key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            result.update(numeric_facts(value, f'{prefix}.{i}'))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        result[prefix] = obj
    return result

def format_facts_block():
    return 'FACTS (reference data: match each number to its model, metric, category and unit; answer in natural language):\n' + json.dumps(load_facts(), ensure_ascii=False, indent=2)

class GroundingError(ValueError):
    pass

def render_grounded_answer(payload):
    """Numbers and their labels are produced by code, never by LLM prose.

    This validates numerical rendering, not the truth of qualitative prose or
    whether the selected facts completely answer the question.
    """
    if not isinstance(payload, dict) or set(payload) != {'explanation', 'fact_ids'}:
        raise GroundingError('Expected explanation and fact_ids only.')
    explanation, ids = payload['explanation'], payload['fact_ids']
    if not isinstance(explanation, str) or len(explanation) > 4000 or re.search(r'\d', explanation):
        raise GroundingError('Explanation must be short qualitative prose without digits.')
    if not isinstance(ids, list) or len(ids) > 12 or any(not isinstance(x,str) for x in ids):
        raise GroundingError('Expected at most twelve fact IDs.')
    facts = numeric_facts()
    if any(key not in facts for key in ids):
        raise GroundingError('Unknown fact ID.')
    rows = []
    for key in dict.fromkeys(ids):
        value = facts[key]
        # Include full hierarchy so values cannot silently drift to a different model/category.
        parts = key.split('.')
        if len(parts) >= 2 and parts[-2] in ('ci95', 'ci95_wilson') and parts[-1] in ('0','1'):
            parts[-2:] = ['95% confidence interval', 'lower' if parts[-1]=='0' else 'upper']
        label = ' / '.join(part.replace('_', ' ') for part in parts)
        rows.append(f'{label}: {value:g}')
    return '\n\n'.join([explanation.strip(), *rows]).strip()

# Kept as a formatting diagnostic only, never as claim validation.
_NUMBER_RE = re.compile(r'(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?')
def verify_answer_numbers(text):
    known = {Decimal(str(v)) for v in numeric_facts().values()}
    return sorted({n for n in _NUMBER_RE.findall(text) if Decimal(n.replace(',','')) not in known})
