"""Project facts, optional legacy fact rendering and nonblocking numeric diagnostics."""
import json
import re
from decimal import Decimal, ROUND_HALF_UP
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
# A number counts as "known" when it equals a FACTS value, or that value as a
# percentage (0.762 -> 76.2), after rounding to the number of decimals written.
# Both English (12,811.5) and Turkish (12.811,5 / %69,9) separators are accepted.
_NUMBER_RE = re.compile(r'(?<![\w.,])-?\d+(?:[.,]\d+)*')

def _readings(token):
    """Possible plain-decimal readings of a written number, e.g. '12,811' -> 12811 or 12.811."""
    sign, body = ('-', token[1:]) if token.startswith('-') else ('', token)
    readings = set()
    for thousands, decimal in ((',', '.'), ('.', ',')):
        parts = body.split(decimal)
        if len(parts) > 2 or (len(parts) == 2 and thousands in parts[1]):
            continue
        groups = parts[0].split(thousands)
        if len(groups) > 1 and (len(groups[0]) > 3 or any(len(g) != 3 for g in groups[1:])):
            continue
        readings.add(sign + ''.join(groups) + ('.' + parts[1] if len(parts) == 2 else ''))
    return readings

def _known_values():
    values = set()
    for v in numeric_facts().values():
        d = Decimal(str(v))
        values.update((d, d * 100))
    return values

def _matches(reading, known):
    number = Decimal(reading)
    step = Decimal(1).scaleb(number.as_tuple().exponent)
    return any(k.quantize(step, rounding=ROUND_HALF_UP) == number for k in known)

def verify_answer_numbers(text):
    known = _known_values()
    return sorted({t for t in _NUMBER_RE.findall(text) if not any(_matches(r, known) for r in _readings(t))})
