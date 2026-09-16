
"""Controlled semantic-gate ablation; no provider calls."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile

from .runtime import Harness
from .store import StateStore
from .adapters import FixtureAdapter, registry_for, DEMO_SOURCE
from .semantic import from_session
from .semantic_workflow import SemanticGate, fixture_catalog
from .semantic_tools import SemanticCatalog


def validated_catalog(original, change):
    data = original.model_dump()
    change(data)
    return SemanticCatalog.model_validate(data)


async def evaluate():
    with tempfile.TemporaryDirectory() as folder:
        store = StateStore(Path(folder) / "sessions.sqlite")
        try:
            harness = Harness(store, registry_for(FixtureAdapter(False)))
            initial = harness.create(
                DEMO_SOURCE,
                "extract entities and relationships",
                "semantic-evaluation",
            )
            state = await asyncio.wait_for(
                harness.run(initial.session_id, "semantic-evaluation"),
                timeout=60,
            )
            assert state.status == "completed"
            original_state = state.model_dump_json()
            baseline = from_session(state)
            assert not baseline["needs_review"]
            assert len(baseline["business_facts"]) == 2
            catalog = fixture_catalog(state)

            def alias(data):
                entity = next(e for e in data["entities"] if e["entity_id"] == "ms")
                if "aliases" not in entity:
                    raise ValueError("Entity schema has no aliases field")
                entity["name"] = "Microsoft Corporation"
                entity["aliases"] = ["Microsoft"]
                data["revision"] = "alias-case"

            def ambiguous(data):
                entity = dict(next(
                    e for e in data["entities"] if e["entity_id"] == "ms"
                ))
                entity["entity_id"] = "ms-other"
                data["entities"].append(entity)
                data["revision"] = "ambiguous-case"

            def wrong_type(data):
                entity = next(e for e in data["entities"] if e["entity_id"] == "ms")
                entity["entity_type"] = "City"
                data["revision"] = "type-conflict-case"

            def missing(data):
                data["entities"] = [
                    e for e in data["entities"] if e["entity_id"] != "ms"
                ]
                data["revision"] = "unknown-case"

            cases = [
                ("clean", catalog, 2, 0),
                ("alias", validated_catalog(catalog, alias), 2, 0),
                ("ambiguous", validated_catalog(catalog, ambiguous), 1, 1),
                ("type_conflict", validated_catalog(catalog, wrong_type), 1, 1),
                ("unknown_entity", validated_catalog(catalog, missing), 1, 1),
            ]
            results = []
            baseline_ids = {
                fact["view_id"] for fact in baseline["business_facts"]
            }

            for name, scoped_catalog, ready_count, review_count in cases:
                output = await SemanticGate(scoped_catalog).apply(state)
                ready_ids = {f["view_id"] for f in output["ready_facts"]}
                review_ids = {f["view_id"] for f in output["needs_review"]}

                assert len(output["ready_facts"]) == ready_count, name
                assert len(output["needs_review"]) == review_count, name
                assert ready_ids.isdisjoint(review_ids), name
                assert ready_ids | review_ids == baseline_ids, name
                assert output["status"] == (
                    "needs_review" if review_count else "completed"
                ), name
                assert state.model_dump_json() == original_state, name

                if name in ("clean", "alias"):
                    microsoft = next(
                        f for f in output["ready_facts"]
                        if f["subject"] == "Microsoft"
                    )
                    assert microsoft["subject_entity_id"] == "ms", name
                else:
                    assert output["ready_facts"][0]["subject"] == "OpenAI", name

                results.append({
                    "case": name,
                    "without_gate": {
                        "business_fact_count": len(baseline_ids),
                        "entity_identity_checked": False,
                        "entity_types_checked": False,
                    },
                    "with_gate": {
                        "status": output["status"],
                        "ready_count": ready_count,
                        "review_count": review_count,
                        "ready_facts": output["ready_facts"],
                        "needs_review": output["needs_review"],
                        "trace": output["trace"],
                    },
                    "upstream_state_unchanged": True,
                })
                print(
                    f"{name}: without_gate={len(baseline_ids)} unchecked; "
                    f"with_gate={ready_count} ready, {review_count} review"
                )
        finally:
            store.close()

    report = {
        "evaluation": "semantic_gate_controlled_ablation",
        "api_calls": 0,
        "adapter": "FixtureAdapter",
        "baseline_definition": "same business-fact projection without semantic gate",
        "results": results,
        "limitations": [
            "One fixed fixture; catalog conditions vary.",
            "Not a noisy-document extraction benchmark.",
            "Review routing is not a factual error-rate estimate.",
            "Ready facts in a needs_review batch are not automatically published.",
        ],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(f"semantic_ablation_{stamp}.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("EVALUATION OK: 5 semantic cases passed; 0 API calls.")
    print(f"Report: {path.resolve()}")


def main():
    asyncio.run(evaluate())


if __name__ == "__main__":
    main()
