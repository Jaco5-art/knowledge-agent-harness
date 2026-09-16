from __future__ import annotations

from ..rag import FAISSKnowledgeBase
from ..state import WorkflowState


def retrieve_evidence(state: WorkflowState, knowledge_base: FAISSKnowledgeBase, top_k: int = 3) -> dict:
    retrieved: dict[str, list[str]] = {}
    for fact in state.get("fused_facts", []):
        query = " ".join(part for part in (fact.subject, fact.predicate, fact.object, fact.time) if part)
        retrieved[fact.fact_id] = [chunk.text for chunk in knowledge_base.search(query, top_k)]
    return {"retrieved_chunks": retrieved}
