from __future__ import annotations

from ..models import ExtractedFact
from ..state import WorkflowState


def fuse_knowledge(state: WorkflowState) -> dict:
    """Deduplicate facts without silently rewriting conflicting predicates."""
    candidates = (
        state.get("entity_facts", [])
        + state.get("event_facts", [])
        + state.get("relationship_facts", [])
    )
    fused: dict[tuple[str, str, str, str, str], ExtractedFact] = {}
    for fact in candidates:
        key = fact.canonical_key()
        current = fused.get(key)
        if current is None or fact.extraction_confidence > current.extraction_confidence:
            fused[key] = fact
    return {"fused_facts": list(fused.values())}

