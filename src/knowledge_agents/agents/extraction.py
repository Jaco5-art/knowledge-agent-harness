from __future__ import annotations

from ..models import FactType
from ..provider import ExtractionProvider
from ..state import WorkflowState


def extract_selected(state: WorkflowState, provider: ExtractionProvider) -> dict:
    plan = state.get("plan", [])
    output: dict[str, list] = {
        "entity_facts": [],
        "event_facts": [],
        "relationship_facts": [],
    }
    mapping = {
        "entity": (FactType.ENTITY, "entity_facts"),
        "event": (FactType.EVENT, "event_facts"),
        "relationship": (FactType.RELATIONSHIP, "relationship_facts"),
    }
    for agent_name, (fact_type, state_key) in mapping.items():
        if agent_name in plan:
            output[state_key] = provider.extract(state["source_text"], fact_type)
    return output

