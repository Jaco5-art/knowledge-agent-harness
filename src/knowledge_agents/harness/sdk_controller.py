
"""Independent offline SDK scheduling; shared Harness business policies."""
import asyncio
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from agents import Agent, Runner, RunConfig, FunctionTool
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
)
from .sdk_runner_check import OfflineModel
from .runtime import Harness
from .contracts import (
    ExtractInput, RetrieveInput, VerifyInput, CorrectInput, CorrectOutput,
)
from .store import StateStore
from .adapters import FixtureAdapter, registry_for, DEMO_SOURCE

async def advance(self, state):
    if state.stage == 'extract':
        role = state.extraction_plan[state.extraction_index]
        output = await self._call(state, ExtractInput(source=state.source, task=state.task, fact_type=role), lambda o: self._check_extraction(state, o))
        if output is None:
            return
        duplicate_count = 0
        for fact in output.facts:
            if fact.claim_key() not in state.seen:
                self._add(state, fact)
            else:
                duplicate_count += 1
        state.extraction_index += 1
        if state.extraction_index == len(state.extraction_plan):
            state.stage = 'retrieve' if state.pending else 'finish'
        self._save(state, 'extraction_applied', agent=role + '_agent', duplicates_removed=duplicate_count)
        return
    if not state.pending or state.stage == 'finish':
        state.stage, state.status = ('finish', 'completed')
        self._save(state, 'session_completed', accepted=len(state.accepted), rejected=len(state.rejected))
        return
    candidate = state.candidates[state.pending[0]]
    cid = candidate.candidate_id
    if state.stage == 'retrieve':
        output = await self._call(state, RetrieveInput(source=state.source, fact=candidate.fact, top_k=state.config.top_k), lambda o: self._check_retrieval(state, o))
        if output is None:
            return
        if not output.evidence:
            if not state.config.full_document_fallback:
                self._pause(state, 'empty_retrieval')
                return
            from .contracts import EvidenceSpan
            output.evidence = [EvidenceSpan(evidence_id='full:' + state.source_hash, start=0, end=len(state.source), text=state.source)]
            state.empty_retrieval_fallbacks.append(cid)
            self._save(state, 'full_document_fallback', candidate_id=cid)
        state.evidence[cid] = output.evidence
        state.stage = 'verify'
        self._save(state, 'evidence_bound', candidate_id=cid)
    elif state.stage == 'verify':
        evidence = state.evidence[cid]
        output = await self._call(state, VerifyInput(candidate=candidate, evidence=evidence), lambda o: self._check_verification(candidate, evidence, o))
        if output is None:
            return
        decision = output.decisions[0]
        state.decisions.append(decision)
        if decision.verdict == 'supported':
            state.accepted.append(cid)
            state.pending.pop(0)
            state.stage = 'retrieve'
        elif decision.verdict == 'ambiguous':
            self._pause(state, 'ambiguous_fact')
            return
        elif candidate.round >= state.config.max_correction_rounds:
            self._pause(state, 'correction_budget_exhausted')
            return
        else:
            state.stage = 'correct'
        self._save(state, 'verification_applied', candidate_id=cid, verdict=decision.verdict)
    elif state.stage == 'correct':

        def check(output: CorrectOutput):
            f = output.replacement
            if f and (f.quote not in state.source or f.subject != candidate.fact.subject or f.object != candidate.fact.object or (f.fact_type != candidate.fact.fact_type)):
                raise ValueError('Correction must retain entity pair/type and source grounding')
        output = await self._call(state, CorrectInput(candidate=candidate, evidence=state.evidence[cid], reason=state.decisions[-1].reason), check)
        if output is None:
            return
        if output.replacement and output.replacement.claim_key() in state.seen:
            self._pause(state, 'cycle_duplicate_or_no_progress')
            return
        state.pending.pop(0)
        state.rejected.append(cid)
        if output.replacement:
            self._add(state, output.replacement, candidate)
        state.stage = 'retrieve'
        self._save(state, 'correction_applied', candidate_id=cid, replacement_created=output.replacement is not None)


class SchedulerModel(OfflineModel):
    def __init__(self, state):
        super().__init__()
        self.state = state

    async def get_response(self, *args, **kwargs):
        self.turns += 1
        if self.state.status == "running":
            cid = uuid.uuid4().hex
            items = [ResponseFunctionToolCall(
                id="item_" + cid, call_id=cid,
                name="step_" + self.state.stage,
                arguments="{}", type="function_call",
            )]
        else:
            items = [ResponseOutputMessage(
                id="final_" + uuid.uuid4().hex,
                type="message", role="assistant", status="completed",
                content=[ResponseOutputText(
                    type="output_text",
                    text=json.dumps({"status": self.state.status}),
                    annotations=[],
                )],
            )]
        return ModelResponse(output=items, usage=Usage(), response_id=None)


class SDKController:
    def __init__(self, store, registry):
        # Reuse persistence, validation and retry policies, not its run loop.
        self.policies = Harness(store, registry)
        self.store = store
        self.registry = registry
        self.steps = 0

    async def run(self, session_id, tenant_id, max_steps=200):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.steps = 0
        with self.store.session_lock(session_id):
            state = self.store.load(session_id, tenant_id)
            if state.tool_versions != self.registry.versions():
                raise ValueError("Tool versions changed; start a new session")
            if state.status != "running":
                return state

            def handler_for(stage):
                async def invoke(context, arguments):
                    if json.loads(arguments) != {}:
                        raise ValueError("Stage tools accept no model-supplied state")
                    if state.status != "running" or state.stage != stage:
                        raise ValueError("Invalid stage transition")
                    if self.steps >= max_steps:
                        raise RuntimeError("SDK step budget exhausted")
                    self.steps += 1
                    await advance(self.policies, state)
                    return json.dumps({
                        "stage": state.stage,
                        "status": state.status,
                        "state_version": state.version,
                    })
                return invoke

            tools = [
                FunctionTool(
                    name="step_" + stage,
                    description="Advance the trusted " + stage + " stage.",
                    params_json_schema={
                        "type": "object", "properties": {},
                        "required": [], "additionalProperties": False,
                    },
                    on_invoke_tool=handler_for(stage),
                    strict_json_schema=True,
                )
                for stage in ("extract", "retrieve", "verify", "correct", "finish")
            ]
            agent = Agent(
                name="OfflineKnowledgeScheduler",
                instructions="Advance one permitted business stage at a time.",
                model=SchedulerModel(state),
                tools=tools,
            )
            await Runner.run(
                agent, "Complete the fixture knowledge task.",
                max_turns=max_steps + 1,
                run_config=RunConfig(tracing_disabled=True),
            )
            return state


def summary(state):
    return {
        "status": state.status,
        "facts": sorted(
            [state.candidates[cid].fact.model_dump() for cid in state.accepted],
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
        "tool_calls": state.call_count,
        "corrected_candidates": sum(
            candidate.round > 0 for candidate in state.candidates.values()
        ),
        "pending_reviews": [
            review.reason for review in state.reviews if not review.resolved
        ],
    }


async def forbidden(*args, **kwargs):
    raise AssertionError("SDK path called the original Harness execution loop")


async def evaluate():
    with tempfile.TemporaryDirectory() as folder:
        results = []
        for use_sdk in (False, True):
            store = StateStore(Path(folder) / f"case_{use_sdk}.sqlite")
            try:
                registry = registry_for(FixtureAdapter(False))
                harness = Harness(store, registry)
                initial = harness.create(
                    DEMO_SOURCE, "extract entities and relationships", "offline"
                )
                if use_sdk:
                    controller = SDKController(store, registry)
                    with patch.object(Harness, "_run", forbidden):
                        state = await controller.run(initial.session_id, "offline")
                    steps = controller.steps
                else:
                    state = await harness.run(initial.session_id, "offline")
                    steps = None
                results.append(summary(state))
            finally:
                store.close()

    direct, sdk = results
    assert direct == sdk, (direct, sdk)
    assert sdk["status"] == "completed"
    assert sdk["facts"]
    assert sdk["corrected_candidates"] > 0
    assert not sdk["pending_reviews"]

    report = {
        "evaluation": "independent_sdk_scheduler_fixture",
        "api_calls": 0,
        "original_loop_blocked_during_sdk_run": True,
        "sdk_steps": steps,
        "direct": direct,
        "sdk": sdk,
        "limitations": [
            "One deterministic fixture, not a production benchmark.",
            "SDK model chooses stages from trusted state; no live LLM planning.",
            "Business policies, retry logic and persistence are shared.",
            "Shared transition code was extracted from the current runtime.",
            "Resume, cancellation and error-path parity need separate evaluation.",
        ],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(f"sdk_independent_{stamp}.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("SCHEDULER OK: independent SDK loop matches fixture; 0 API calls.")
    print(f"SDK steps: {steps}")
    print(f"Report: {path.resolve()}")


def main():
    asyncio.run(asyncio.wait_for(evaluate(), timeout=120))


if __name__ == "__main__":
    main()
