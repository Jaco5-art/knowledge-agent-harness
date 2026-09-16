"""Six paired repetitions of the unchanged context-extension negation case."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import run_context_extension as experiment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def repeat(emit, model, client=None):
    original = experiment.cases
    selected = next(item for item in original() if item[0] == 'negation')
    # Preserve candidate ID, request bytes and adapter prompt. Alternate order
    # through the original runner's index-based ordering across six repetitions.
    experiment.cases = lambda: iter([selected] * 6)
    started_count = 0
    def tagged(event):
        nonlocal started_count
        if event['event'] == 'attempt_started':
            started_count += 1
        emit(dict(event, repetition=(started_count - 1)//2 + 1))
    try:
        rows = await experiment.run(tagged, model, client)
    finally:
        experiment.cases = original
    for index, row in enumerate(rows):
        row['repetition'] = index//2 + 1
    report = experiment.summary(rows, client is not None)
    report['notes'] = [
        'Six repetitions of one previously failed synthetic negation input; not six independent cases.',
        'Same candidate, evidence and prompts; execution order alternates each repetition.',
        'Diagnostic only. Small sample cannot establish a general error rate or prove causation.',
        'Supported is erroneous for this positive acquisition candidate with explicit negative evidence.',
        'No retry or prompt tuning. All failed requests remain in coverage.',
    ]
    report['pairs'] = []
    for repetition in range(1,7):
        arm = {r['mode']:r for r in rows if r['repetition']==repetition}
        complete = len(arm)==2 and all(r['status']=='succeeded' for r in arm.values())
        report['pairs'].append(dict(repetition=repetition, complete=complete,
            budget_verdict=arm.get('budget',{}).get('decision',{}).get('verdict'),
            compact_verdict=arm.get('compact',{}).get('decision',{}).get('verdict'),
            both_correct=all(r.get('acceptance_matches_expected',False) for r in arm.values()) if complete else None))
    report['request_sha256'] = hashlib.sha256(selected[1].model_dump_json().encode()).hexdigest()
    return rows,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--model',default='gpt-5-mini')
    args=parser.parse_args()
    if args.execute and not os.environ.get('OPENAI_API_KEY'):
        parser.error('Configure OPENAI_API_KEY in this window.')
    # Prepare both modes first with zero provider calls, before opening a client.
    prepared, _ = asyncio.run(repeat(lambda event:None,args.model))
    if len(prepared)!=12 or any(r['status']!='prepared_offline' or not r.get('lossless') for r in prepared):
        raise SystemExit('Offline preparation failed; no API calls made.')
    print('PREPARED: 6 unchanged-input pairs; lossless checks passed; 0 API calls.',flush=True)
    if not args.execute:
        return
    prefix='context_negation_repeat_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    import knowledge_agents.harness.context as context
    protocol=dict(event='protocol',model=args.model,max_provider_requests=12,repetitions=6,
        runner_sha256=sha(Path(__file__)),base_runner_sha256=sha(Path(experiment.__file__)),
        harness_sha256={p.name:sha(p) for p in sorted(Path(context.__file__).parent.glob('*.py'))})
    async def live(emit):
        from openai import AsyncOpenAI
        async with AsyncOpenAI(max_retries=0,timeout=60) as client:
            return await repeat(emit,args.model,client)
    with Path(prefix+'.jsonl').open('x',encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event,ensure_ascii=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
            if event['event']=='attempt_finished':
                print('repeat',event['repetition'],event['mode'],event['status'],
                      event.get('decision',{}).get('verdict',''),flush=True)
        emit(protocol)
        rows,report=asyncio.run(live(emit))
        emit(dict(event='finished',summary=report))
    with Path(prefix+'_summary.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print('Results:',prefix+'.jsonl',prefix+'_summary.json')
    if any(r['status']=='failed' for r in rows):
        raise SystemExit('Stopped after first failure; no retries.')

if __name__=='__main__':
    main()
