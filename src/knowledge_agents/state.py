from __future__ import annotations

from typing import TypedDict

from .models import ExtractedFact, VerificationResult


class WorkflowState(TypedDict, total=False):
    source_text: str
    task: str
    plan: list[str]
    entity_facts: list[ExtractedFact]
    event_facts: list[ExtractedFact]
    relationship_facts: list[ExtractedFact]
    fused_facts: list[ExtractedFact]
    retrieved_chunks: dict[str, list[str]]
    verification_results: list[VerificationResult]
    verified_facts: list[ExtractedFact]
    rejected_facts: list[ExtractedFact]
    correction_round: int
    errors: list[str]
