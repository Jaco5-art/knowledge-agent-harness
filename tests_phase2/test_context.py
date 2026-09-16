import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from knowledge_agents.harness.context import ContextPolicy, ContextBudgetExceeded, pack_evidence, prepare_context
from knowledge_agents.harness.context_adapter import ContextOpenAIAdapter
from knowledge_agents.harness.contracts import ExtractInput, ExtractOutput
import test_runtime
from test_runtime import Scenario


class ContextTests(unittest.TestCase):
    def payload(self):
        text = '甲公司没有收购乙公司，时间为2023年。' * 100
        return json.dumps({'candidate': {'candidate_id': 'keep-me'}, 'evidence': [
            dict(evidence_id='a', start=0, end=1500, text=text[:1500]),
            dict(evidence_id='b', start=500, end=1800, text=text[500:1800])]}, ensure_ascii=False)

    def test_lossless_pack(self):
        original = self.payload()
        packed = pack_evidence(original)
        self.assertLess(len(packed.encode()), len(original.encode()))
        data = json.loads(packed)
        blocks = {b['block_id']: b for b in data['evidence_blocks']}
        for ref, old in zip(data['evidence'], json.loads(original)['evidence']):
            b = blocks[ref['block_id']]
            self.assertEqual(b['text'][ref['start']-b['start']:ref['end']-b['start']], old['text'])
            self.assertEqual(ref['evidence_id'], old['evidence_id'])
        self.assertEqual(data['candidate']['candidate_id'], 'keep-me')

    def test_no_growth(self):
        payload = json.dumps({'evidence': [dict(evidence_id='a',start=0,end=1,text='x')]})
        self.assertEqual(pack_evidence(payload), payload)

    def test_conflicting_overlap(self):
        payload = json.dumps({'evidence': [dict(start=0,end=2,text='ab'),dict(start=1,end=3,text='xy')]})
        with self.assertRaises(ValueError):
            pack_evidence(payload)

    def test_budget_includes_schema_and_reserve(self):
        with self.assertRaises(ContextBudgetExceeded) as caught:
            prepare_context('system', '{}', ExtractOutput, 'extract', ContextPolicy(extract_budget=1))
        self.assertGreater(caught.exception.context_metrics['estimated_input_after'], 2)

    def test_policy_identity(self):
        self.assertNotEqual(ContextPolicy().fingerprint(), ContextPolicy(compact=False).fingerprint())

    def test_extract_never_truncates(self):
        payload = json.dumps({'source': 'not acquired ' * 500})
        prepared = prepare_context('', payload, ExtractOutput, 'extract', ContextPolicy())
        self.assertEqual(prepared.payload, payload)


class AdapterContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_blocks_network(self):
        for role in ('entity', 'event', 'relationship'):
            with self.subTest(role=role):
                parse = AsyncMock()
                adapter = ContextOpenAIAdapter('test', client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
                                              policy=ContextPolicy(extract_budget=1))
                with self.assertRaises(ContextBudgetExceeded):
                    await adapter.extract(ExtractInput(source='test', task='extract', fact_type=role))
                parse.assert_not_awaited()

    async def test_usage_and_output_cap(self):
        parse = AsyncMock(return_value=SimpleNamespace(output_parsed=ExtractOutput(facts=[]),
                         usage=SimpleNamespace(input_tokens=12, output_tokens=3)))
        adapter = ContextOpenAIAdapter('test', client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
        reply = await adapter.extract(ExtractInput(source='test',task='extract',fact_type='entity'))
        self.assertEqual(reply.input_tokens, 12)
        self.assertEqual(parse.call_args.kwargs['max_output_tokens'], 4096)
        self.assertIn('estimated_input_after', reply.context_metrics)


class BudgetRuntimeTests(unittest.IsolatedAsyncioTestCase):
    setUp = test_runtime.RuntimeTests.setUp
    tearDown = test_runtime.RuntimeTests.tearDown
    setup_run = test_runtime.RuntimeTests.setup_run
    async def test_budget_pause_no_retry(self):
        class BudgetScenario(Scenario):
            async def extract(self, req):
                raise ContextBudgetExceeded({'budget': 1})
        harness, state = self.setup_run(BudgetScenario())
        result = await harness.run(state.session_id, 'tenant-a')
        self.assertEqual(result.status, 'needs_review')
        self.assertEqual(result.reviews[-1].reason, 'context_budget_exceeded')
        self.assertEqual(result.call_count, 1)
        self.assertFalse(result.accepted)
