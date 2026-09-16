"""Offline paired packing experiment. No model calls or accuracy claims."""
import argparse
import hashlib
import json
from pathlib import Path

from .context import ContextPolicy, ContextBudgetExceeded, prepare_context
from .contracts import Candidate, Fact, EvidenceSpan, VerifyInput, VerifyOutput, validate_evidence


def cases():
    sentence = 'In 2023, Company A partnered with Company B. It did not acquire Company B. '
    source = ''.join(f'Record {i}: {sentence}' for i in range(70))
    layouts = {
        'high_overlap': [(0, 2200), (400, 2600), (800, 3000)],
        'no_overlap': [(0, 700), (800, 1500), (1600, 2300)],
        'short_single': [(0, 150)],
        'nested_overlap': [(0, 3000), (500, 2000), (700, 1200)],
        'over_budget': [(0, 2200), (400, 2600), (800, 3000)],
    }
    for name, intervals in layouts.items():
        candidate = Candidate(candidate_id='fixed-' + name, fact=Fact(
            subject='Company A', predicate='acquired', object='Company B',
            time='2023', quote=sentence.strip()))
        spans = [EvidenceSpan(evidence_id=f'e{i}',start=a,end=b,text=source[a:b])
                 for i, (a,b) in enumerate(intervals)]
        validate_evidence(source, spans)
        yield name, VerifyInput(candidate=candidate, evidence=spans), 1 if name == 'over_budget' else 32000


def assert_lossless(original, packed):
    old, new = json.loads(original), json.loads(packed)
    assert old['candidate'] == new['candidate'], 'Candidate changed'
    assert len(old['evidence']) == len(new['evidence']), 'Evidence missing'
    blocks = {b['block_id']:b for b in new.get('evidence_blocks', [])}
    for a,b in zip(old['evidence'],new['evidence']):
        reconstructed = dict(b)
        if 'block_id' in reconstructed:
            block = blocks[reconstructed.pop('block_id')]
            reconstructed['text'] = block['text'][b['start']-block['start']:b['end']-block['start']]
        assert a == reconstructed, 'Evidence text, ID or metadata changed'


def run_comparison():
    rows = []
    for name, request, budget in cases():
        payload = request.model_dump_json()
        row = dict(case=name, request_sha256=hashlib.sha256(payload.encode()).hexdigest(), modes={})
        for mode in ('budget','compact'):
            policy = ContextPolicy(compact=mode=='compact', verify_budget=budget)
            try:
                # Deliberately a fixed offline harness prompt, not a provider token measurement.
                result = prepare_context('Offline evidence packing comparison.', payload, VerifyOutput, 'verify', policy)
                assert_lossless(payload, result.payload)
                row['modes'][mode] = dict(status='ready', lossless=True,
                    payload_bytes=len(result.payload.encode()), **result.metrics)
            except ContextBudgetExceeded as exc:
                row['modes'][mode] = dict(status='context_budget_exceeded', **exc.context_metrics)
        rows.append(row)
    return dict(experiment='offline-context-packing-v1', provider_calls=0,
        actual_tokens=None, factual_accuracy=None,
        note='Synthetic stress cases, not a representative dataset. Byte estimates are not actual token savings. No verification decisions are generated.', cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON file; existing files are not overwritten')
    args = parser.parse_args()
    report = run_comparison()
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    for row in report['cases']:
        a,b = row['modes']['budget'],row['modes']['compact']
        print(f"{row['case']}: {a['status']} -> {b['status']}; estimated input {a['estimated_input_after']} -> {b['estimated_input_after']}")
    print('Offline checks passed. Provider calls: 0. Actual token savings: not measured.')


if __name__ == '__main__':
    main()
