from __future__ import annotations

from ..state import WorkflowState


def plan_task(state: WorkflowState) -> dict:
    task = state.get("task", "extract entities, events and relationships").casefold()
    plan = ["entity"]
    if any(term in task for term in ("event", "happen", "investment", "acquisition")):
        plan.append("event")
    if any(term in task for term in ("relationship", "partnership", "collaboration", "affiliation")):
        plan.append("relationship")
    if len(plan) == 1 and "entity" not in task:
        plan.extend(["event", "relationship"])
    plan.extend(["fusion", "verification"])
    return {"plan": plan, "correction_round": state.get("correction_round", 0)}

