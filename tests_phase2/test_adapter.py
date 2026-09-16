"""Real SDK + FAISS, mocked HTTP boundary: validates integration, not model quality."""
import json
import unittest
from unittest.mock import patch

from knowledge_agents.harness.adapters import OpenAIAdapter, DEMO_SOURCE
from knowledge_agents.harness.contracts import (ExtractInput, RetrieveInput, VerifyInput,
    Candidate, CorrectInput, Fact)
from knowledge_agents.harness.tools import TransientError, PermanentError


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx2
        from openai import AsyncOpenAI
        self.requests = []
        self.error = None

        def handle(request):
            self.requests.append(request)
            if self.error:
                return httpx2.Response(self.error, json={"error": {"message": "injected", "type": "test"}})
            body = json.loads(request.content)
            if request.url.path.endswith("/embeddings"):
                return httpx2.Response(200, json={"object": "list", "model": "test-embedding",
                    "data": [{"object": "embedding", "index": i, "embedding": [1.0, float(i + 1), 0.5]}
                             for i in range(len(body["input"]))],
                    "usage": {"prompt_tokens": 7, "total_tokens": 7}})
            data = json.loads(body["input"][1]["content"])
            name = body["text"]["format"]["name"]
            if name in {"ExtractOutput", "EntityExtractionOutput", "EventExtractionOutput", "RelationshipExtractionOutput"}:
                result = {"facts": [{"subject": "Microsoft", "predicate": "invested in", "object": "OpenAI",
                    "quote": "Microsoft invested in OpenAI in 2023.", "fact_type": data["fact_type"],
                    "time": None, "location": None, "source_agent": "relationship_agent"}]}
            elif name == "VerifyOutput":
                result = {"decisions": [{"candidate_id": data["candidate"]["candidate_id"],
                    "verdict": "supported", "reason": "Mock provider response",
                    "evidence_ids": [data["evidence"][0]["evidence_id"]]}]}
            else:
                result = {"replacement": None}
            return httpx2.Response(200, json={"id": "resp_mock", "object": "response", "created_at": 0,
                "status": "completed", "model": "mock-model", "error": None,
                "incomplete_details": None,
                "output": [{"type": "message", "id": "msg_mock", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps(result), "annotations": []}]}],
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0}}})
        transport = httpx2.MockTransport(handle)
        client = AsyncOpenAI(api_key="offline-test-placeholder", max_retries=0,
                            http_client=httpx2.AsyncClient(transport=transport, trust_env=False))
        self.adapter = OpenAIAdapter("mock-model", client=client)

    async def asyncTearDown(self):
        await self.adapter.close()

    async def test_real_sdk_structured_extraction_retrieval_verification(self):
        extracted = await self.adapter.extract(ExtractInput(source=DEMO_SOURCE, task="relations", fact_type="relationship"))
        fact = extracted.data.facts[0]
        self.assertEqual(extracted.input_tokens, 20)
        retrieved = await self.adapter.retrieve(RetrieveInput(source=DEMO_SOURCE, fact=fact, top_k=3))
        self.assertEqual(retrieved.input_tokens, 14)
        self.assertTrue(retrieved.data.evidence)
        verified = await self.adapter.verify(VerifyInput(candidate=Candidate(candidate_id="f1", fact=fact),
                                                       evidence=retrieved.data.evidence))
        self.assertEqual(verified.data.decisions[0].candidate_id, "f1")
        self.assertEqual(verified.output_tokens, 10)
        corrected = await self.adapter.correct(CorrectInput(candidate=Candidate(candidate_id="f1", fact=fact),
                                        evidence=retrieved.data.evidence, reason="test"))
        self.assertIsNone(corrected.data.replacement)

    async def test_index_cache_is_content_keyed(self):
        fact = Fact(subject="Microsoft", predicate="invested in", object="OpenAI", quote=DEMO_SOURCE)
        first = await self.adapter.retrieve(RetrieveInput(source=DEMO_SOURCE, fact=fact, top_k=1))
        second = await self.adapter.retrieve(RetrieveInput(source=DEMO_SOURCE, fact=fact, top_k=1))
        third = await self.adapter.retrieve(RetrieveInput(source=DEMO_SOURCE + " New document.", fact=fact, top_k=1))
        self.assertEqual([first.input_tokens, second.input_tokens, third.input_tokens], [14, 7, 14])
        self.assertNotEqual(first.data.evidence[0].evidence_id, third.data.evidence[0].evidence_id)

    async def test_http_429_maps_to_runtime_retry_without_sdk_retry(self):
        self.error = 429
        with self.assertRaises(TransientError):
            await self.adapter.extract(ExtractInput(source=DEMO_SOURCE, task="relations", fact_type="relationship"))
        self.assertEqual(len(self.requests), 1)

    async def test_http_401_is_permanent(self):
        self.error = 401
        with self.assertRaises(PermanentError):
            await self.adapter.extract(ExtractInput(source=DEMO_SOURCE, task="relations", fact_type="relationship"))
        self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
