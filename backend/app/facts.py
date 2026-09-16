"""Render numeric claims from fact IDs, retaining each value's own label."""
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
    return ('FACTS (use exact fact IDs as {{fact.id}} placeholders; never type a numeric '
            'value yourself):\n' + json.dumps(numeric_facts(), indent=2))

class GroundingError(ValueError):
    pass

_PLACEHOLDER_RE = re.compile(r'\{\{([^{}]+)\}\}')

def render_grounded_answer(payload):
    """Numeric values are substituted from FACTS by code, never typed by the LLM.

    The LLM writes full prose and embeds every number as a {{fact.id}} placeholder;
    this function verifies no digit appears outside a placeholder (so nothing can be
    invented or miscalculated), resolves each placeholder against FACTS, and returns
    the rendered text. This validates numerical grounding, not the truth of the
    surrounding prose or whether the selected facts fully answer the question.
    """
    if not isinstance(payload, dict) or set(payload) != {'explanation'}:
        raise GroundingError('Expected explanation only.')
    explanation = payload['explanation']
    if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 4000:
        raise GroundingError('Explanation must be non-empty prose under 4000 characters.')
    placeholders = _PLACEHOLDER_RE.findall(explanation)
    if len(placeholders) > 20:
        raise GroundingError('Expected at most twenty fact references.')
    outside_placeholders = _PLACEHOLDER_RE.sub('', explanation)
    if re.search(r'\d', outside_placeholders):
        raise GroundingError('Digits may only appear inside {{fact.id}} placeholders.')
    facts = numeric_facts()
    unknown = sorted({key.strip() for key in placeholders if key.strip() not in facts})
    if unknown:
        raise GroundingError(f'Unknown fact ID(s): {unknown}')

    def resolve(match):
        return f'{facts[match.group(1).strip()]:g}'

    return _PLACEHOLDER_RE.sub(resolve, explanation).strip()

# Kept as a formatting diagnostic only, never as claim validation.
_NUMBER_RE = re.compile(r'(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?')
def verify_answer_numbers(text):
    known = {Decimal(str(v)) for v in numeric_facts().values()}
    return sorted({n for n in _NUMBER_RE.findall(text) if Decimal(n.replace(',','')) not in known})
