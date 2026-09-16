"""Lossless evidence packing and conservative, explicitly estimated budgets.

UTF-8 bytes are a deliberately conservative token proxy, NOT measured tokens.
Provider framing remains estimated; this is not a model context-limit guarantee.
"""
import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field


class ContextPolicy(BaseModel):
    compact: bool = True
    extract_budget: int = Field(default=32000, gt=0)
    verify_budget: int = Field(default=16000, gt=0)
    correct_budget: int = Field(default=16000, gt=0)
    output_reserve: int = Field(default=4096, gt=0)
    framing_reserve: int = Field(default=1024, ge=0)

    def fingerprint(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class ContextBudgetExceeded(RuntimeError):
    def __init__(self, metrics):
        super().__init__('context_budget_exceeded')
        self.context_metrics = metrics


def pack_evidence(payload):
    """Store overlapping exact text once; retain every ID and original interval."""
    data = json.loads(payload)
    spans = data.get('evidence', [])
    if not spans:
        return payload
    blocks = []
    for span in sorted(spans, key=lambda s: (s['start'], s['end'])):
        start, end, value = span['start'], span['end'], span['text']
        if start < 0 or end <= start or len(value) != end - start:
            raise ValueError('Invalid evidence interval')
        if blocks and start <= blocks[-1]['end']:
            block = blocks[-1]
            overlap = min(end, block['end']) - start
            if block['text'][start-block['start']:start-block['start']+overlap] != value[:overlap]:
                raise ValueError('Inconsistent overlapping evidence')
            if end > block['end']:
                block['text'] += value[overlap:]
                block['end'] = end
        else:
            blocks.append(dict(start=start, end=end, text=value))
    for i, block in enumerate(blocks):
        block['block_id'] = f'block-{i}'
    refs = []
    for span in spans:
        block = next(b for b in blocks if b['start'] <= span['start'] and b['end'] >= span['end'])
        refs.append({**{k: v for k, v in span.items() if k != 'text'}, 'block_id': block['block_id']})
    data['evidence'] = refs
    data['evidence_blocks'] = blocks
    data['evidence_encoding'] = 'Evidence text is the block substring at [start-block.start:end-block.start]. Cite original evidence_id, never block_id.'
    packed = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    # Compaction must not increase payload size, including mapping overhead.
    return packed if len(packed.encode()) < len(payload.encode()) else payload


@dataclass
class PreparedContext:
    payload: str
    metrics: dict


def prepare_context(prompt, payload, schema, stage, policy):
    packed = pack_evidence(payload) if policy.compact and stage != 'extract' else payload
    schema_bytes = len(json.dumps(schema.model_json_schema(), ensure_ascii=False).encode())
    fixed = len(prompt.encode()) + schema_bytes + policy.framing_reserve
    before, after = fixed + len(payload.encode()), fixed + len(packed.encode())
    budget = getattr(policy, stage + '_budget')
    metrics = dict(estimator='utf8_bytes_v1_plus_schema_and_framing',
        estimated_input_before=before, estimated_input_after=after,
        output_reserve=policy.output_reserve, budget=budget,
        compaction_ratio=after / before, compacted=packed != payload,
        policy_id=policy.fingerprint(), content_dropped=False)
    if after + policy.output_reserve > budget:
        raise ContextBudgetExceeded(metrics)
    return PreparedContext(packed, metrics)
