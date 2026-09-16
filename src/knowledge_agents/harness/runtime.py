from __future__ import annotations

import asyncio
import hashlib
from json import JSONDecodeError
import time
import uuid

from pydantic import ValidationError

from .contracts import (Candidate, CorrectInput, CorrectOutput, Decision, ExtractInput,
    ExtractOutput, RetrieveInput, RetrieveOutput, ReviewItem, RuntimeConfig, Session,
    VerifyInput, VerifyOutput, validate_decisions, validate_evidence)
from .store import ConflictError, StateStore
from .tools import InvalidOutput, PermanentError, ToolRegistry, TransientError


class Harness:
    """Single-worker runtime. Registered async tools must support cancellation.

    Successful outputs are checkpointed before application. A crash during a read-only
    model request can repeat that request on resume; exactly-once billing is not claimed.
    """
    def __init__(self, store: StateStore, registry: ToolRegistry):
        self.store, self.registry = store, registry

    def create(self, source: str, task: str, tenant_id: str,
               config: RuntimeConfig | None = None) -> Session:
        config = config or RuntimeConfig()
        if not source.strip() or len(source) > config.max_document_chars:
            raise ValueError("Empty document or document size limit exceeded")
        if not tenant_id.strip():
            raise ValueError("Tenant scope is required")
        for stage in ("extract", "retrieve", "verify", "correct"):
            self.registry.get(stage, stage)
        from ..agents.planner import plan_task
        plan = [p for p in plan_task({"task": task})["plan"] if p in {"entity", "event", "relationship"}]
        state = Session(session_id=uuid.uuid4().hex, tenant_id=tenant_id, source=source,
            source_hash=hashlib.sha256(source.encode()).hexdigest(), task=task,
            config=config, tool_versions=self.registry.versions(), extraction_plan=plan)
        self.store.create(state)
        return state

    def _save(self, state: Session, event: str, **details):
        self.store.save(state, {"event": event, "stage": state.stage, **details})

    def _pause(self, state: Session, reason: str):
        candidate_id = state.pending[0] if state.pending and state.stage != "extract" else None
        state.status = "needs_review"
        state.reviews.append(ReviewItem(review_id=uuid.uuid4().hex,
            candidate_id=candidate_id, stage=state.stage, reason=reason,
            state_version=state.version + 1))
        self._save(state, "review_required", reason=reason, candidate_id=candidate_id)

    async def _call(self, state: Session, request, check):
        name = state.stage
        spec = self.registry.get(name, name)
        payload = request.model_dump_json()
        digest = hashlib.sha256(payload.encode()).hexdigest()
        key = f"{state.revision}:{name}:{spec.version}:{digest}"
        if key in state.outputs:
            output = spec.output_model.model_validate(state.outputs[key])
            check(output)
            return output
        validated = spec.input_model.model_validate_json(payload)
        while state.attempts.get(key, 0) < state.config.max_attempts:
            if state.call_count >= state.config.max_tool_calls:
                self._pause(state, "tool_call_budget_exhausted")
                return None
            attempt = state.attempts.get(key, 0) + 1
            state.attempts[key] = attempt
            state.call_count += 1
            call_id = uuid.uuid4().hex
            self._save(state, "tool_started", tool=name, tool_version=spec.version,
                call_id=call_id, request_hash=digest, attempt=attempt,
                input_chars=len(payload),
                agent=state.extraction_plan[state.extraction_index] + "_agent" if name == "extract" else name,
                candidate_id=state.pending[0] if state.pending and name != "extract" else None)
            started = time.perf_counter()
            reply = None
            try:
                reply = await asyncio.wait_for(spec.handler(validated), state.config.timeout_seconds)
                data = reply.data.model_dump() if hasattr(reply.data, "model_dump") else reply.data
                output = spec.output_model.model_validate(data)
                try:
                    check(output)
                except ValueError as exc:
                    raise InvalidOutput("Business output contract failed") from exc
            except asyncio.CancelledError:
                self._save(state, "tool_interrupted", call_id=call_id,
                           latency_ms=(time.perf_counter() - started) * 1000)
                raise
            except Exception as exc:
                from .context import ContextBudgetExceeded
                if isinstance(exc, ContextBudgetExceeded):
                    self._save(state, "context_budget_exceeded", call_id=call_id,
                               context_metrics=exc.context_metrics)
                    self._pause(state, "context_budget_exceeded")
                    return None
                retryable = isinstance(exc, (TimeoutError, TransientError, InvalidOutput, ValidationError, JSONDecodeError))
                # Never log raw SDK exception bodies; they may contain customer text/secrets.
                self._save(state, "tool_failed", tool=name, call_id=call_id, attempt=attempt,
                    error_type=type(exc).__name__, retryable=retryable,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    input_tokens=reply.input_tokens if reply else None,
                    output_tokens=reply.output_tokens if reply else None)
                if not retryable:
                    self._pause(state, f"permanent_tool_error:{type(exc).__name__}")
                    return None
                if attempt < state.config.max_attempts:
                    await asyncio.sleep(state.config.retry_delay_seconds * 2 ** (attempt - 1))
                continue
            state.outputs[key] = output.model_dump()
            self._save(state, "tool_succeeded", tool=name, call_id=call_id, attempt=attempt,
                output_chars=len(output.model_dump_json()),
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                **({"context_metrics": reply.context_metrics} if reply.context_metrics else {}))
            return output
        self._pause(state, "attempts_exhausted")
        return None

    def _add(self, state: Session, fact, parent: Candidate | None = None):
        if fact.quote not in state.source:
            raise ValueError("Fact quote must occur in source")
        # Provenance is assigned by runtime, not trusted from model-generated labels.
        fact = fact.model_copy(update={"source_agent": "correction_agent" if parent else fact.fact_type + "_agent"})
        candidate = Candidate(candidate_id=uuid.uuid4().hex, fact=fact,
            parent_id=parent.candidate_id if parent else None, round=parent.round + 1 if parent else 0)
        state.candidates[candidate.candidate_id] = candidate
        state.pending.append(candidate.candidate_id)
        state.seen.append(fact.claim_key())

    def _check_extraction(self, state, output: ExtractOutput):
        if len(state.candidates) + len(output.facts) > state.config.max_facts:
            raise ValueError("Too many facts")
        if any(f.fact_type != state.extraction_plan[state.extraction_index] for f in output.facts):
            raise ValueError("Extraction returned wrong fact type")
        if any(f.quote not in state.source for f in output.facts):
            raise ValueError("Unbound extraction evidence")

    def _check_retrieval(self, state, output: RetrieveOutput):
        if len(output.evidence) > state.config.top_k:
            raise ValueError("Top-K contract exceeded")
        validate_evidence(state.source, output.evidence)

    def _check_verification(self, candidate, evidence, output: VerifyOutput):
        validate_decisions(output, [candidate.candidate_id])
        decision = output.decisions[0]
        available = {e.evidence_id for e in evidence}
        if len(set(decision.evidence_ids)) != len(decision.evidence_ids):
            raise ValueError("Duplicate citations")
        if not set(decision.evidence_ids) <= available:
            raise ValueError("Unknown evidence ID")
        if decision.verdict == "supported" and not decision.evidence_ids:
            raise ValueError("Supported decision must cite evidence")

    async def run(self, session_id: str, tenant_id: str, *, max_steps: int | None = None) -> Session:
        with self.store.session_lock(session_id):
            return await self._run(session_id, tenant_id, max_steps=max_steps)

    async def _run(self, session_id: str, tenant_id: str, *, max_steps: int | None = None) -> Session:
        state = self.store.load(session_id, tenant_id)
        if state.tool_versions != self.registry.versions():
            raise ValueError("Tool versions changed; start a new session")
        steps = 0
        while state.status == "running" and (max_steps is None or steps < max_steps):
            steps += 1
            if state.stage == "extract":
                role = state.extraction_plan[state.extraction_index]
                output = await self._call(state, ExtractInput(source=state.source, task=state.task, fact_type=role),
                                           lambda o: self._check_extraction(state, o))
                if output is None:
                    break
                duplicate_count = 0
                for fact in output.facts:
                    if fact.claim_key() not in state.seen:
                        self._add(state, fact)
                    else:
                        duplicate_count += 1
                state.extraction_index += 1
                if state.extraction_index == len(state.extraction_plan):
                    state.stage = "retrieve" if state.pending else "finish"
                self._save(state, "extraction_applied", agent=role + "_agent", duplicates_removed=duplicate_count)
                continue
            if not state.pending or state.stage == "finish":
                state.stage, state.status = "finish", "completed"
                self._save(state, "session_completed", accepted=len(state.accepted), rejected=len(state.rejected))
                break
            candidate = state.candidates[state.pending[0]]
            cid = candidate.candidate_id
            if state.stage == "retrieve":
                output = await self._call(state, RetrieveInput(source=state.source,
                    fact=candidate.fact, top_k=state.config.top_k), lambda o: self._check_retrieval(state, o))
                if output is None:
                    break
                if not output.evidence:
                    if not state.config.full_document_fallback:
                        self._pause(state, "empty_retrieval")
                        break
                    from .contracts import EvidenceSpan
                    output.evidence = [EvidenceSpan(evidence_id="full:" + state.source_hash,
                        start=0, end=len(state.source), text=state.source)]
                    state.empty_retrieval_fallbacks.append(cid)
                    self._save(state, "full_document_fallback", candidate_id=cid)
                state.evidence[cid] = output.evidence
                state.stage = "verify"
                self._save(state, "evidence_bound", candidate_id=cid)
            elif state.stage == "verify":
                evidence = state.evidence[cid]
                output = await self._call(state, VerifyInput(candidate=candidate, evidence=evidence),
                    lambda o: self._check_verification(candidate, evidence, o))
                if output is None:
                    break
                decision = output.decisions[0]
                state.decisions.append(decision)
                if decision.verdict == "supported":
                    state.accepted.append(cid)
                    state.pending.pop(0)
                    state.stage = "retrieve"
                elif decision.verdict == "ambiguous":
                    self._pause(state, "ambiguous_fact")
                    break
                elif candidate.round >= state.config.max_correction_rounds:
                    self._pause(state, "correction_budget_exhausted")
                    break
                else:
                    state.stage = "correct"
                self._save(state, "verification_applied", candidate_id=cid, verdict=decision.verdict)
            elif state.stage == "correct":
                def check(output: CorrectOutput):
                    f = output.replacement
                    if f and (f.quote not in state.source or f.subject != candidate.fact.subject
                              or f.object != candidate.fact.object or f.fact_type != candidate.fact.fact_type):
                        raise ValueError("Correction must retain entity pair/type and source grounding")
                output = await self._call(state, CorrectInput(candidate=candidate,
                    evidence=state.evidence[cid], reason=state.decisions[-1].reason), check)
                if output is None:
                    break
                if output.replacement and output.replacement.claim_key() in state.seen:
                    self._pause(state, "cycle_duplicate_or_no_progress")
                    break
                state.pending.pop(0)
                state.rejected.append(cid)
                if output.replacement:
                    self._add(state, output.replacement, candidate)
                state.stage = "retrieve"
                self._save(state, "correction_applied", candidate_id=cid,
                           replacement_created=output.replacement is not None)
        return state

    def resolve_review(self, session_id: str, tenant_id: str, review_id: str,
                       expected_version: int, actor: str, action: str) -> Session:
        """Local trusted operator API. Resume/reject only; no 'accept as fact' bypass."""
        with self.store.session_lock(session_id):
            return self._resolve_review(session_id, tenant_id, review_id, expected_version, actor, action)

    def _resolve_review(self, session_id, tenant_id, review_id, expected_version, actor, action):
        if not actor.strip() or action not in {"retry", "reject", "cancel"}:
            raise ValueError("Require operator identity and retry/reject/cancel")
        state = self.store.load(session_id, tenant_id)
        if state.version != expected_version:
            raise ConflictError("Review targets a stale state version")
        item = next((r for r in state.reviews if r.review_id == review_id and not r.resolved), None)
        if state.status != "needs_review" or item is None:
            raise ValueError("Review is not pending")
        if item.candidate_id != (state.pending[0] if state.pending and state.stage != "extract" else None):
            raise ConflictError("Review candidate has changed")
        item.resolved, item.actor, item.action = True, actor, action
        if action == "cancel":
            state.status = "cancelled"
        elif action == "reject":
            if item.candidate_id:
                state.rejected.append(state.pending.pop(0))
                state.stage, state.status = "retrieve", "running"
            else:
                state.status = "failed"
        else:
            # Human retry does not reset global call/correction budgets.
            state.revision += 1
            state.status = "running"
        self._save(state, "review_resolved", review_id=review_id, actor=actor, action=action)
        return state
