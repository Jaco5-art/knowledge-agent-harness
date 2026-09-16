"""Paired verification smoke test. Default offline; --execute allows 6 paid calls."""
import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from .context import ContextPolicy
from .context_adapter import ContextOpenAIAdapter
from .context_compare import assert_lossless
from .contracts import Candidate, Fact, EvidenceSpan, VerifyInput, validate_evidence, validate_decisions


def cases():
    for name, statement, predicate, accepted in [
        ('partnership', 'In 2023, Aster Ltd partnered with Birch Ltd. No investment or acquisition occurred.', 'partnered with', True),
        ('negation', 'In 2023, Aster Ltd partnered with Birch Ltd. No investment or acquisition occurred.', 'acquired', False),
        ('wrong_year', 'Aster Ltd acquired Birch Ltd in 2021. In 2023, no acquisition occurred.', 'acquired', False),
    ]:
        source = ('Administrative pagination. No transaction details here.\n'*30 + statement + '\n' + 'Blank appendix checklist.\n'*50)
        spans = [EvidenceSpan(evidence_id=f'{name}-e{i}', start=a, end=b, text=source[a:b])
                 for i,(a,b) in enumerate([(0,len(source)-300),(300,len(source)),(600,len(source)-100)])]
        validate_evidence(source,spans)
        request = VerifyInput(candidate=Candidate(candidate_id=name, fact=Fact(subject='Aster Ltd',
            predicate=predicate, object='Birch Ltd',time='2023',quote=statement)),evidence=spans)
        yield name,request,accepted


class OfflinePrepared(Exception):
    pass


class Capture:
    def __init__(self,client):
        self.client=client
        self.responses=SimpleNamespace(parse=self.parse)
        self.kwargs=None
        self.response=None
        self.calls=0

    async def parse(self,**kwargs):
        self.kwargs=kwargs
        if self.client is None:
            raise OfflinePrepared()
        self.calls+=1
        self.response=await self.client.responses.parse(**kwargs)
        return self.response


async def run(emit,model,client=None):
    rows=[]
    emit(dict(event='start',model=model,live=client is not None,max_requests=6,dataset='synthetic-pairs-v1'))
    for i,(name,request,expected) in enumerate(cases()):
        for mode in (('budget','compact') if i%2==0 else ('compact','budget')):
            capture=Capture(client)
            adapter=ContextOpenAIAdapter(model,client=capture,policy=ContextPolicy(compact=mode=='compact'))
            row=dict(case=name,mode=mode,expected_accepted=expected,
                request_sha256=hashlib.sha256(request.model_dump_json().encode()).hexdigest(),policy_id=adapter.policy.fingerprint())
            emit(dict(event='attempt_started',**row))
            started=time.perf_counter()
            try:
                reply=await asyncio.wait_for(adapter.verify(request),65)
                validate_decisions(reply.data,[request.candidate.candidate_id])
                decision=reply.data.decisions[0]
                ids={e.evidence_id for e in request.evidence}
                if not set(decision.evidence_ids)<=ids or len(set(decision.evidence_ids))!=len(decision.evidence_ids):
                    raise ValueError('Invalid citations')
                if decision.verdict=='supported' and not decision.evidence_ids:
                    raise ValueError('Missing citations')
                row.update(status='succeeded',decision=decision.model_dump(),context_metrics=reply.context_metrics,
                           acceptance_matches_expected=(decision.verdict=='supported')==expected)
            except OfflinePrepared:
                row.update(status='prepared_offline')
            except Exception as exc:
                row.update(status='failed',error_type=type(exc).__name__)
            usage=getattr(capture.response,'usage',None)
            row.update(provider_requests_started=capture.calls,
                latency_seconds=time.perf_counter()-started if client is not None else None,
                usage={k:getattr(usage,k,None) for k in ('input_tokens','output_tokens','total_tokens')} if usage else None)
            if capture.kwargs:
                payload=capture.kwargs['input'][1]['content']
                assert_lossless(request.model_dump_json(),payload)
                row.update(lossless=True,payload_bytes=len(payload.encode()),
                    prompt_sha256=hashlib.sha256(capture.kwargs['input'][0]['content'].encode()).hexdigest(),
                    output_cap=capture.kwargs['max_output_tokens'])
            emit(dict(event='attempt_finished',**row))
            rows.append(row)
            if row['status']=='failed':
                emit(dict(event='stopped',reason='first_failure_no_retry'))
                return rows
    pairs=[]
    for name,_,_ in cases():
        pair={r['mode']:r for r in rows if r['case']==name}
        a,b=pair['budget'],pair['compact']
        result=dict(case=name,complete=a['status']==b['status']=='succeeded')
        if result['complete']:
            ua,ub=a['usage'],b['usage']
            result.update(same_verdict=a['decision']['verdict']==b['decision']['verdict'],
                same_citation_set=set(a['decision']['evidence_ids'])==set(b['decision']['evidence_ids']),
                input_token_reduction=ua['input_tokens']-ub['input_tokens'] if ua and ub and ua['input_tokens'] is not None and ub['input_tokens'] is not None else None)
        pairs.append(result)
    emit(dict(event='completed',pairs=pairs,provider_requests_started=sum(r['provider_requests_started'] for r in rows)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--model',default='gpt-5-mini')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.execute and not os.environ.get('OPENAI_API_KEY'):
        parser.error('Configure OPENAI_API_KEY locally first')
    with args.output.open('x',encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event,ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            if event['event']=='attempt_finished':
                print(event['case'],event['mode'],event['status'],flush=True)
        async def execute():
            if not args.execute:
                return await run(emit,args.model)
            from openai import AsyncOpenAI
            async with AsyncOpenAI(max_retries=0,timeout=60) as client:
                return await run(emit,args.model,client)
        rows=asyncio.run(execute())
    print('Results:',args.output)
    if any(r['status']=='failed' for r in rows):
        raise SystemExit(1)


if __name__=='__main__':
    main()
