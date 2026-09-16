from __future__ import annotations

import os

from .agents.correction import correct_knowledge
from .agents.extraction import extract_selected
from .agents.fusion import fuse_knowledge
from .agents.planner import plan_task
from .agents.retrieval import retrieve_evidence
from .agents.verification import verify_knowledge
from .provider import ExtractionProvider
from .rag import FAISSKnowledgeBase
from .state import WorkflowState


def build_workflow(
    provider: ExtractionProvider,
    max_correction_rounds: int | None = None,
    knowledge_base: FAISSKnowledgeBase | None = None,
):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before building the workflow.") from exc

    limit = max_correction_rounds
    if limit is None:
        limit = int(os.getenv("MAX_CORRECTION_ROUNDS", "2"))

    graph = StateGraph(WorkflowState)
    graph.add_node("planner", plan_task)
    graph.add_node("extractors", lambda state: extract_selected(state, provider))
    graph.add_node("fusion", fuse_knowledge)
    graph.add_node("verification", verify_knowledge)
    graph.add_node("correction", lambda state: correct_knowledge(state, provider))
    if knowledge_base is not None:
        graph.add_node("retrieval", lambda state: retrieve_evidence(state, knowledge_base))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "extractors")
    graph.add_edge("extractors", "fusion")
    graph.add_edge("fusion", "retrieval" if knowledge_base is not None else "verification")
    if knowledge_base is not None:
        graph.add_edge("retrieval", "verification")

    def route_after_verification(state: WorkflowState) -> str:
        has_failures = bool(state.get("rejected_facts"))
        within_limit = state.get("correction_round", 0) < limit
        return "correction" if has_failures and within_limit else END

    graph.add_conditional_edges("verification", route_after_verification)
    graph.add_edge("correction", "retrieval" if knowledge_base is not None else "verification")
    return graph.compile()


def run_workflow(
    source_text: str,
    task: str,
    provider: ExtractionProvider,
    max_correction_rounds: int | None = None,
    knowledge_base: FAISSKnowledgeBase | None = None,
) -> WorkflowState:
    app = build_workflow(provider, max_correction_rounds, knowledge_base)
    return app.invoke({"source_text": source_text, "task": task, "correction_round": 0})
