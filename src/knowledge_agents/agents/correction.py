from __future__ import annotations

from ..provider import ExtractionProvider
from ..state import WorkflowState


def correct_knowledge(state: WorkflowState, provider: ExtractionProvider) -> dict:
    next_round = state.get("correction_round", 0) + 1
    corrected = provider.correct(
        source_text=state["source_text"],
        facts=state.get("fused_facts", []),
        results=state.get("verification_results", []),
        correction_round=next_round,
    )
    return {
        "fused_facts": corrected,
        "correction_round": next_round,
        "verification_results": [],
        "verified_facts": [],
        "rejected_facts": [],
        "retrieved_chunks": {},
    }
