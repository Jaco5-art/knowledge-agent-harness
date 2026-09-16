"""Runtime contracts. Evidence grounding is necessary, not semantic proof."""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Fact(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str | None = None
    fact_type: Literal["entity", "event", "relationship"] = "relationship"
    time: str | None = None
    location: str | None = None
    quote: str = Field(min_length=1)
    source_agent: str = "relationship_agent"
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)

    def claim_key(self) -> str:
        import json
        values = [self.subject, self.predicate, self.object, self.time, self.location]
        normalized = [(v or "").strip().casefold() for v in values]
        return hashlib.sha256(json.dumps(normalized).encode()).hexdigest()


class Candidate(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str
    parent_id: str | None = None
    round: int = 0
    fact: Fact


class EvidenceSpan(Contract):
    evidence_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    score: float | None = None


class RuntimeConfig(Contract):
    max_correction_rounds: int = Field(default=2, ge=0, le=10)
    max_attempts: int = Field(default=2, ge=1, le=5)
    max_tool_calls: int = Field(default=100, ge=1)
    timeout_seconds: float = Field(default=60, gt=0)
    retry_delay_seconds: float = Field(default=0.1, ge=0, le=5)
    top_k: int = Field(default=3, ge=1, le=20)
    max_facts: int = Field(default=100, ge=1, le=1000)
    max_document_chars: int = Field(default=100000, ge=1)
    # Fallback may widen evidence scope. Explicitly off for comparable RAG runs.
    full_document_fallback: bool = False


class ExtractInput(Contract):
    source: str
    task: str
    fact_type: Literal["entity", "event", "relationship"]


class ExtractOutput(Contract):
    facts: list[Fact]


class RetrieveInput(Contract):
    source: str
    fact: Fact
    top_k: int = Field(ge=1)


class RetrieveOutput(Contract):
    evidence: list[EvidenceSpan]


class VerifyInput(Contract):
    candidate: Candidate
    evidence: list[EvidenceSpan]


class Decision(Contract):
    candidate_id: str
    verdict: Literal["supported", "unsupported", "needs_correction", "ambiguous"]
    reason: str = Field(min_length=1)
    evidence_ids: list[str]
    confidence: float | None = Field(default=None, ge=0, le=1)


class VerifyOutput(Contract):
    decisions: list[Decision]


class CorrectInput(Contract):
    candidate: Candidate
    evidence: list[EvidenceSpan]
    reason: str


class CorrectOutput(Contract):
    replacement: Fact | None


class ReviewItem(Contract):
    review_id: str
    candidate_id: str | None
    stage: str
    reason: str
    state_version: int
    resolved: bool = False
    actor: str | None = None
    action: str | None = None


class Session(Contract):
    session_id: str
    tenant_id: str
    source: str
    source_hash: str
    task: str
    config: RuntimeConfig
    tool_versions: dict[str, str]
    extraction_plan: list[str]
    extraction_index: int = 0
    version: int = 0
    status: Literal["running", "completed", "needs_review", "failed", "cancelled"] = "running"
    stage: Literal["extract", "retrieve", "verify", "correct", "finish"] = "extract"
    candidates: dict[str, Candidate] = Field(default_factory=dict)
    pending: list[str] = Field(default_factory=list)
    accepted: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    seen: list[str] = Field(default_factory=list)
    evidence: dict[str, list[EvidenceSpan]] = Field(default_factory=dict)
    decisions: list[Decision] = Field(default_factory=list)
    reviews: list[ReviewItem] = Field(default_factory=list)
    # Successful step results are durable before being applied to domain state.
    outputs: dict[str, dict] = Field(default_factory=dict)
    attempts: dict[str, int] = Field(default_factory=dict)
    call_count: int = 0
    revision: int = 0
    empty_retrieval_fallbacks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def document_integrity(self):
        if hashlib.sha256(self.source.encode()).hexdigest() != self.source_hash:
            raise ValueError("Document hash mismatch")
        return self


def validate_decisions(output: VerifyOutput, ids: list[str]) -> None:
    got = [d.candidate_id for d in output.decisions]
    if len(got) != len(ids) or len(set(got)) != len(got) or set(got) != set(ids):
        raise ValueError("Require exactly one decision for each candidate ID")


def validate_evidence(source: str, spans: list[EvidenceSpan]) -> None:
    ids = [e.evidence_id for e in spans]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evidence IDs")
    for e in spans:
        if e.end <= e.start or e.end > len(source) or source[e.start:e.end] != e.text:
            raise ValueError("Evidence offsets do not match the source document")
