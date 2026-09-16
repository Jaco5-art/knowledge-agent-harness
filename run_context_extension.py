"""Six synthetic verification pairs: actual token usage and acceptance quality."""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from knowledge_agents.harness.context_live_compare import Capture, OfflinePrepared
from knowledge_agents.harness.context_adapter import ContextOpenAIAdapter
from knowledge_agents.harness.context import ContextPolicy
from knowledge_agents.harness.context_compare import assert_lossless
from knowledge_agents.harness.contracts import Candidate, Fact, EvidenceSpan, VerifyInput, validate_evidence, validate_decisions


def cases():
    rows = [
        ('year', 'Birch Ltd invested in Aster Ltd in 2020. This record was updated in 2025.', 'Birch Ltd', 'invested in', 'Aster Ltd', '2020', True),
        ('direction', 'Birch Ltd invested in Aster Ltd in 2020. Aster Ltd did not invest in Birch Ltd.', 'Aster Ltd', 'invested in', 'Birch Ltd', '2020', False),
        ('negation', 'Aster Ltd did not acquire Birch Ltd in 2024. The acquisition report was false.', 'Aster Ltd', 'acquired', 'Birch Ltd', '2024', False),
        ('plan', 'Aster Ltd plans to invest in Birch Ltd in 2027. As of 2026, no investment has occurred.', 'Aster Ltd', 'invested in', 'Birch Ltd', '2027', False),
        ('partnership', 'Aster Ltd entered a strategic partnership with Birch Ltd in 2022. No acquisition occurred.', 'Aster Ltd', 'entered a strategic partnership with', 'Birch Ltd', '2022', True),
        ('wrong_year', 'Aster Ltd invested in Birch Ltd in 2021. The ledger was updated in 2025.', 'Aster Ltd', 'invested in', 'Birch Ltd', '2025', False),
    ]
    for i,(name, statement, subject, predicate, obj, year, expected) in enumerate(rows):
        source = 'Administrative page. No transaction information.\n'*35 + statement + '\n' + 'Appendix formatting notes.\n'*40
        # Include a no-overlap control as well as overlapping evidence.
        ranges = [(0,len(source))] if name == 'partnership' else [(0,len(source)-150),(150,len(source)),(300,len(source)-50)]
        evidence = [EvidenceSpan(evidence_id=name+'-'+str(j),start=a,end=b,text=source[a:b]) for j,(a,b) in enumerate(ranges)]
        validate_evidence(source,evidence)
        request = VerifyInput(candidate=Candidate(candidate_id=name,fact=Fact(subject=subject,predicate=predicate,
                    object=obj,time=year,quote=statement)), evidence=evidence)
        yield name,request,expected


def summary(rows, live):
    result = dict(provider_requests_started=sum(r['provider_requests_started'] for r in rows),
                  expected_attempts=12, completed_attempts=sum(r['status']=='succeeded' for r in rows),
                  live=live, arms={})
    for mode in ('budget','compact'):
        arm=[r for r in rows if r['mode']==mode]
        good=[r for r in arm if r['status']=='succeeded']
        metrics={}
        for field in ('input_tokens','output_tokens','total_tokens'):
            values=[r['usage'][field] for r in good if r.get('usage') and isinstance(r['usage'].get(field),int)]
            metrics[field]=dict(recorded_sum=sum(values) if values else None,coverage=len(values),expected=6)
        result['arms'][mode]=dict(successful=len(good),correct_acceptance=sum(r.get('acceptance_matches_expected',False) for r in good),
                                  usage=metrics)
    result['paired_input_token_reduction_fraction']=None
    if all(result['arms'][m]['usage']['input_tokens']['coverage']==6 for m in ('budget','compact')):
        a=result['arms']['budget']['usage']['input_tokens']['recorded_sum']
        b=result['arms']['compact']['usage']['input_tokens']['recorded_sum']
        result['paired_input_token_reduction_fraction']=(a-b)/a if a else None
    result['notes']=['Acceptance-only gold: unsupported, ambiguous and needs_correction are not distinguished as quality labels.',
                     'Citation IDs and lossless payload reconstruction checked; citation entailment not independently assessed.',
                     'Budget means uncompacted evidence with budget enforcement, not context protection disabled.',
                     'Six synthetic cases, one repetition; not an end-to-end extraction or latency benchmark.',
                     'No model quality scores exist for offline preparation.']
    if not live:
        for arm in result['arms'].values():
            arm['correct_acceptance']=None
    return result


async def run(emit, model, client=None):
    rows=[]
    for index,(name,request,expected) in enumerate(cases()):
        for mode in (('budget','compact') if index%2==0 else ('compact','budget')):
            capture=Capture(client)
            adapter=ContextOpenAIAdapter(model,client=capture,policy=ContextPolicy(compact=mode=='compact'))
            row=dict(case=name,mode=mode,expected_accepted=expected,
                     request_sha256=hashlib.sha256(request.model_dump_json().encode()).hexdigest(),
                     policy_id=adapter.policy.fingerprint())
            emit(dict(event='attempt_started',**row))
            started=time.perf_counter()
            try:
                reply=await asyncio.wait_for(adapter.verify(request),65)
                validate_decisions(reply.data,[name])
                decision=reply.data.decisions[0]
                ids=decision.evidence_ids
                if len(ids)!=len(set(ids)) or not set(ids)<={e.evidence_id for e in request.evidence} or (decision.verdict=='supported' and not ids):
                    raise ValueError('Invalid citations')
                row.update(status='succeeded',decision=decision.model_dump(),
                           acceptance_matches_expected=(decision.verdict=='supported')==expected)
            except OfflinePrepared:
                row['status']='prepared_offline'
            except Exception as exc:
                row.update(status='failed',error_type=type(exc).__name__)
            if capture.kwargs:
                try:
                    assert_lossless(request.model_dump_json(),capture.kwargs['input'][1]['content'])
                    row['lossless']=True
                except Exception as exc:
                    row.update(status='failed',error_type=type(exc).__name__,lossless=False)
            usage=getattr(capture.response,'usage',None)
            row.update(provider_requests_started=capture.calls,
                       latency_seconds=time.perf_counter()-started if client else None,
                       usage={k:getattr(usage,k,None) for k in ('input_tokens','output_tokens','total_tokens')} if usage else None)
            rows.append(row)
            emit(dict(event='attempt_finished',**row))
            if row['status']=='failed':
                return rows
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--model',default='gpt-5-mini')
    args=parser.parse_args()
    if args.execute and not os.environ.get('OPENAI_API_KEY'):
        parser.error('Set OPENAI_API_KEY in this window.')
    prefix='context_extension_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    async def work(emit):
        if not args.execute:
            return await run(emit,args.model)
        from openai import AsyncOpenAI
        async with AsyncOpenAI(max_retries=0,timeout=60) as client:
            return await run(emit,args.model,client)
    with Path(prefix+'.jsonl').open('x',encoding='utf-8') as stream:
        def emit(row):
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
            if row['event']=='attempt_finished':
                print(row['case'],row['mode'],row['status'],flush=True)
        import knowledge_agents.harness.context as context
        emit(dict(event='protocol',model=args.model,max_provider_requests=12,live=args.execute,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(context.__file__).parent.glob('*.py'))}))
        rows=asyncio.run(work(emit))
        report=summary(rows,args.execute)
        emit(dict(event='finished',summary=report))
    with Path(prefix+'_summary.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print('Results:',prefix+'.jsonl',prefix+'_summary.json')
    if any(r['status']=='failed' for r in rows):
        raise SystemExit('Stopped after first failure; no automatic retries.')

if __name__=='__main__':
    main()
