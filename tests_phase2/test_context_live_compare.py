import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from knowledge_agents.harness.context_live_compare import run
from knowledge_agents.harness.contracts import Decision, VerifyOutput


def test_preflight_no_network():
    events=[]
    rows=asyncio.run(run(events.append,'test'))
    assert len(rows)==6
    assert all(r['status']=='prepared_offline' and r['provider_requests_started']==0 for r in rows)
    assert len({r['prompt_sha256'] for r in rows})==1
    for name in ('partnership','negation','wrong_year'):
        pair={r['mode']:r for r in rows if r['case']==name}
        assert pair['compact']['request_sha256']==pair['budget']['request_sha256']
        assert pair['compact']['payload_bytes']<pair['budget']['payload_bytes']


def test_six_mocked_requests():
    async def parse(**kwargs):
        data=json.loads(kwargs['input'][1]['content'])
        cid=data['candidate']['candidate_id']
        return SimpleNamespace(output_parsed=VerifyOutput(decisions=[Decision(candidate_id=cid,
            verdict='supported' if cid=='partnership' else 'unsupported',reason='mock',
            evidence_ids=[data['evidence'][0]['evidence_id']])]),
            usage=SimpleNamespace(input_tokens=100,output_tokens=20,total_tokens=120))
    mock=AsyncMock(side_effect=parse)
    events=[]
    rows=asyncio.run(run(events.append,'test',SimpleNamespace(responses=SimpleNamespace(parse=mock))))
    assert mock.await_count==6
    assert all(r['status']=='succeeded' and r['acceptance_matches_expected'] for r in rows)
    assert all(p['complete'] for p in events[-1]['pairs'])


def test_failure_stops_and_redacts():
    mock=AsyncMock(side_effect=RuntimeError('private-text'))
    events=[]
    rows=asyncio.run(run(events.append,'test',SimpleNamespace(responses=SimpleNamespace(parse=mock))))
    assert len(rows)==1 and mock.await_count==1
    assert 'private-text' not in json.dumps(events)


def test_invalid_citation_preserves_usage():
    mock=AsyncMock(return_value=SimpleNamespace(output_parsed=VerifyOutput(decisions=[Decision(
        candidate_id='partnership',verdict='supported',reason='mock',evidence_ids=['invalid'])]),
        usage=SimpleNamespace(input_tokens=10,output_tokens=5,total_tokens=15)))
    rows=asyncio.run(run(lambda e:None,'test',SimpleNamespace(responses=SimpleNamespace(parse=mock))))
    assert len(rows)==1 and rows[0]['status']=='failed'
    assert rows[0]['usage']['input_tokens']==10
