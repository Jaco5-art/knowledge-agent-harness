import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_agents.harness import Harness, RuntimeConfig, StateStore
from knowledge_agents.harness.adapters import DEMO_SOURCE, FixtureAdapter, registry_for
from knowledge_agents.harness.contracts import (CorrectOutput, Decision, RetrieveOutput,
    VerifyOutput, validate_decisions)
from knowledge_agents.harness.store import ConflictError
from knowledge_agents.harness.tools import PermanentError, PolicyDenied, ToolReply, ToolSpec, ToolRegistry


class Scenario(FixtureAdapter):
    def __init__(self, mode="normal"):
        super().__init__(False)
        self.mode, self.verifications = mode, 0

    async def retrieve(self, req):
        if self.mode == "empty":
            return ToolReply(RetrieveOutput(evidence=[]))
        if self.mode == "permanent":
            raise PermanentError("sensitive text should not be logged")
        if self.mode == "timeout":
            await asyncio.sleep(1)
        out = await super().retrieve(req)
        if self.mode == "bad_span":
            out.data.evidence[0].start = 1
        return out

    async def verify(self, req):
        self.verifications += 1
        out = await super().verify(req)
        if self.mode == "duplicate":
            out.data.decisions.append(out.data.decisions[0])
        if self.mode == "unknown_evidence":
            out.data.decisions[0].evidence_ids = ["other-document"]
        if self.mode == "ambiguous":
            out.data.decisions[0].verdict = "ambiguous"
        if self.mode == "malformed_once" and self.verifications == 1:
            return ToolReply({"decisions": "wrong"})
        return out

    async def correct(self, req):
        if self.mode == "cycle":
            return ToolReply(CorrectOutput(replacement=req.candidate.fact))
        return await super().correct(req)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "state.sqlite"
        self.store = StateStore(self.db)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def setup_run(self, adapter=None, **config):
        adapter = adapter or Scenario()
        h = Harness(self.store, registry_for(adapter))
        s = h.create(DEMO_SOURCE, "extract entities, events and relationships", "tenant-a",
                     RuntimeConfig(retry_delay_seconds=0, **config))
        return h, s

    async def test_recovery_correction_and_preserve_accepted(self):
        h, s = self.setup_run(FixtureAdapter(True))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.status, "completed")
        accepted = [result.candidates[c] for c in result.accepted]
        self.assertEqual(len(accepted), 2)
        self.assertEqual(sorted(c.round for c in accepted), [0, 1])
        self.assertNotIn("acquired in", [c.fact.predicate for c in accepted])
        corrected = next(c for c in accepted if c.parent_id)
        self.assertIn(corrected.parent_id, result.rejected)
        events = self.store.events(s.session_id, "tenant-a")
        failures = [e for e in events if e["event"] == "tool_failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error_type"], "TransientError")
        self.assertTrue(all(e["input_tokens"] is None for e in events if e["event"] == "tool_succeeded"))

    async def test_resume_new_store_without_reextracting(self):
        h, s = self.setup_run()
        before = await h.run(s.session_id, "tenant-a", max_steps=3)
        ids = set(before.candidates)
        self.store.close()
        self.store = StateStore(self.db)
        class NoExtract(Scenario):
            async def extract(self, req):
                raise AssertionError("Completed extraction must not repeat")
        h = Harness(self.store, registry_for(NoExtract()))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.status, "completed")
        self.assertTrue(ids <= set(result.candidates))

    async def test_crash_after_tool_output_commit_before_domain_apply(self):
        h, s = self.setup_run()
        original = self.store.save
        class Crash(BaseException):
            pass
        def crash(state, event):
            if event["event"] == "extraction_applied":
                raise Crash()
            return original(state, event)
        self.store.save = crash
        with self.assertRaises(Crash):
            await h.run(s.session_id, "tenant-a")
        self.store.save = original
        before = self.store.load(s.session_id, "tenant-a")
        self.assertEqual(before.call_count, 1)
        self.assertEqual(before.candidates, {})
        after = await h.run(s.session_id, "tenant-a", max_steps=1)
        self.assertEqual(after.call_count, 1)
        self.assertEqual(len(after.candidates), 1)

    async def test_malformed_output_retry_recovers(self):
        h, s = self.setup_run(Scenario("malformed_once"))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.status, "completed")
        self.assertTrue(any(e.get("error_type") == "ValidationError" for e in self.store.events(s.session_id, "tenant-a")))

    async def test_duplicate_decisions_pause_without_acceptance(self):
        h, s = self.setup_run(Scenario("duplicate"))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.accepted, [])
        self.assertEqual(result.reviews[-1].reason, "attempts_exhausted")

    async def test_unknown_citations_not_accepted(self):
        h, s = self.setup_run(Scenario("unknown_evidence"))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.accepted, [])

    async def test_empty_retrieval_does_not_silently_widen_context(self):
        h, s = self.setup_run(Scenario("empty"))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.reviews[-1].reason, "empty_retrieval")
        self.assertEqual(result.empty_retrieval_fallbacks, [])

    async def test_opt_in_fallback_recorded(self):
        h, s = self.setup_run(Scenario("empty"), full_document_fallback=True)
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.empty_retrieval_fallbacks), 3)

    async def test_permanent_failure_not_retried_and_redacted(self):
        h, s = self.setup_run(Scenario("permanent"))
        await h.run(s.session_id, "tenant-a")
        events = self.store.events(s.session_id, "tenant-a")
        failures = [e for e in events if e["event"] == "tool_failed"]
        self.assertEqual(len(failures), 1)
        self.assertNotIn("sensitive text", json.dumps(events))

    async def test_timeout_enforced_and_bounded(self):
        h, s = self.setup_run(Scenario("timeout"), timeout_seconds=0.01)
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.reviews[-1].reason, "attempts_exhausted")
        failures = [e for e in self.store.events(s.session_id, "tenant-a") if e["event"] == "tool_failed"]
        self.assertEqual(len(failures), 2)

    async def test_no_progress_pauses_and_preserves_good_fact(self):
        h, s = self.setup_run(Scenario("cycle"))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.reviews[-1].reason, "cycle_duplicate_or_no_progress")
        self.assertEqual(len(result.accepted), 1)

    async def test_zero_correction_rounds(self):
        h, s = self.setup_run(max_correction_rounds=0)
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.reviews[-1].reason, "correction_budget_exhausted")
        self.assertFalse(any(c.parent_id for c in result.candidates.values()))

    async def test_call_budget_survives_human_retry(self):
        h, s = self.setup_run(max_tool_calls=1)
        result = await h.run(s.session_id, "tenant-a")
        h.resolve_review(s.session_id, "tenant-a", result.reviews[-1].review_id, result.version, "operator", "retry")
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.call_count, 1)
        self.assertEqual(result.reviews[-1].reason, "tool_call_budget_exhausted")

    async def test_human_review_no_accept_bypass_and_stale_version(self):
        h, s = self.setup_run(Scenario("ambiguous"))
        result = await h.run(s.session_id, "tenant-a")
        rid = result.reviews[-1].review_id
        with self.assertRaises(ValueError):
            h.resolve_review(s.session_id, "tenant-a", rid, result.version, "operator", "accept")
        with self.assertRaises(ConflictError):
            h.resolve_review(s.session_id, "tenant-a", rid, result.version - 1, "operator", "reject")
        resolved = h.resolve_review(s.session_id, "tenant-a", rid, result.version, "operator", "reject")
        self.assertEqual(len(resolved.rejected), 1)
        self.assertEqual(resolved.accepted, [])

    async def test_retry_revalidates_before_acceptance(self):
        adapter = Scenario("ambiguous")
        h, s = self.setup_run(adapter)
        result = await h.run(s.session_id, "tenant-a")
        h.resolve_review(s.session_id, "tenant-a", result.reviews[-1].review_id, result.version, "operator", "retry")
        adapter.mode = "normal"
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.accepted), 2)
        self.assertGreaterEqual(adapter.verifications, 4)

    async def test_offsets_checked(self):
        h, s = self.setup_run(Scenario("bad_span"))
        result = await h.run(s.session_id, "tenant-a")
        self.assertEqual(result.accepted, [])
        self.assertEqual(result.status, "needs_review")

    async def test_top_k_is_forwarded(self):
        seen = []
        class Recording(Scenario):
            async def retrieve(self, req):
                seen.append(req.top_k)
                return await super().retrieve(req)
        h, s = self.setup_run(Recording(), top_k=5)
        await h.run(s.session_id, "tenant-a")
        self.assertEqual(set(seen), {5})

    async def test_tenant_isolation(self):
        h, s = self.setup_run()
        with self.assertRaises(KeyError):
            await h.run(s.session_id, "tenant-b")
        with self.assertRaises(KeyError):
            self.store.events(s.session_id, "tenant-b")

    async def test_conflicting_state_writer_rejected(self):
        h, s = self.setup_run()
        stale = self.store.load(s.session_id, "tenant-a")
        await h.run(s.session_id, "tenant-a", max_steps=1)
        with self.assertRaises(ConflictError):
            self.store.save(stale, {"event": "bad_write"})

    async def test_session_lock_rejects_second_worker(self):
        h, s = self.setup_run()
        with self.store.session_lock(s.session_id):
            with self.assertRaises(ConflictError):
                await h.run(s.session_id, "tenant-a")

    async def test_tool_version_change_blocks_resume(self):
        h, s = self.setup_run()
        a = Scenario()
        a.version = "changed"
        with self.assertRaises(ValueError):
            await Harness(self.store, registry_for(a)).run(s.session_id, "tenant-a")

    def test_unknown_tool_and_write_tool_denied(self):
        registry = registry_for(Scenario())
        with self.assertRaises(PolicyDenied):
            registry.get("shell", "verify")
        with self.assertRaises(PolicyDenied):
            registry.get("extract", "verify")
        original = registry.get("extract", "extract")
        with self.assertRaises(PolicyDenied):
            ToolRegistry().register(ToolSpec(name="write", version="1", input_model=original.input_model,
                output_model=original.output_model, handler=original.handler,
                allowed_stages=frozenset({"extract"}), read_only=False))


if __name__ == "__main__":
    unittest.main()
