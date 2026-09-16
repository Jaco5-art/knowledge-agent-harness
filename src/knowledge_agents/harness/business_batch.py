
"""Business evaluation with durable sessions and explicit paid execution."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from .adapters import OpenAIAdapter, registry_for
from .scoped_alias_adapter import ScopedAliasAdapter
from .contracts import RuntimeConfig
from .runtime import Harness
from .store import StateStore
from .semantic_workflow import SemanticGate
from .semantic_tools import SemanticCatalog, Entity, RelationRule
from .business_score import evaluate

TENANT = "business-evaluation"
COMPANIES = ["Aster Labs", "Aster Holdings", "Birch Systems"]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def catalog_for(doc):
    # Operator-provided taxonomy; never derived from gold labels.
    aliases = {name: [] for name in COMPANIES}
    for alias, targets in doc["trusted_aliases"].items():
        targets = [targets] if isinstance(targets, str) else targets
        for target in targets:
            if target not in aliases:
                raise ValueError("Unknown trusted alias target")
            aliases[target].append(alias)
    return SemanticCatalog(
        tenant=TENANT,
        scope=hashlib.sha256(doc["source"].encode()).hexdigest(),
        revision="business-catalog-v1",
        entities=[
            Entity(
                entity_id=name, name=name, entity_type="Company",
                aliases=aliases[name],
            )
            for name in COMPANIES
        ],
        relations={
            predicate: RelationRule(
                subject_types=["Company"], object_types=["Company"]
            )
            for predicate in (
                "invested_in", "strategic_partnership_with",
                "acquired_by",
            )
        },
    )


async def run(args, dataset, runtime_docs, fingerprint):
    checkpoint_path = Path(f"{args.run_id}_checkpoint.json")
    db = Path(f"{args.run_id}_sessions.sqlite")
    predictions_path = Path(f"{args.run_id}_predictions.json")
    scores_path = Path(f"{args.run_id}_scores.json")

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Dataset/model/code changed; existing batch cannot be resumed")
        if not db.exists():
            raise FileNotFoundError("Checkpoint exists but session database is missing")
    else:
        if any(p.exists() for p in (db, predictions_path, scores_path)):
            raise FileExistsError("Batch artifacts exist without their checkpoint")
        checkpoint = {"fingerprint": fingerprint, "documents": {}}
        save(checkpoint_path, checkpoint)

    adapter = None
    store = None
    try:
        store = StateStore(db)
        attempted = 0
        for doc in runtime_docs:
            did = doc["document_id"]
            entry = checkpoint["documents"].get(did)
            if entry and entry.get("prediction") is not None:
                continue
            if attempted >= args.limit:
                break
            attempted += 1
            if adapter is not None:
                await adapter.close()
                adapter = None
            adapter = ScopedAliasAdapter(
                args.model, args.embedding_model,
                source=doc["source"], aliases=doc["trusted_aliases"],
            )
            harness = Harness(store, registry_for(adapter))

            if entry is None:
                task = (
                    "extract entities and relationships. "
                    "Extract only completed positive business facts supported by the source. "
                    "Do not turn negation, plans, or CRM status into completed relations. "
                    "Trusted alias mappings (not additional facts): "
                    + json.dumps(doc["trusted_aliases"], ensure_ascii=False)
                )
                state = harness.create(
                    doc["source"], task, TENANT,
                    RuntimeConfig(
                        max_attempts=2, max_tool_calls=40,
                        max_correction_rounds=2,
                    ),
                )
                entry = {"session_id": state.session_id, "prediction": None}
                checkpoint["documents"][did] = entry
                save(checkpoint_path, checkpoint)

            # Existing durable running sessions resume without new extraction sessions.
            state = await harness.run(entry["session_id"], TENANT)
            trace = Path(f"{args.run_id}_trace_{state.session_id}_v{state.version}.jsonl")
            if not trace.exists():
                store.export_trace(state.session_id, TENANT, str(trace))

            facts = []
            review = False
            status = "failed"
            reasons = [r.reason for r in state.reviews if not r.resolved]

            if state.status == "completed":
                gate = await SemanticGate(catalog_for(doc)).apply(state)
                review = bool(gate["needs_review"])
                status = "succeeded"
                # Export publishable facts only: review blocks the entire batch.
                if not review:
                    for fact in gate["ready_facts"]:
                        subject = fact["subject_entity_id"]
                        obj = fact["object_entity_id"]
                        predicate = fact["predicate"]
                        # Evaluator uses active 'acquired'; gate uses 'acquired_by'.
                        if predicate == "acquired_by":
                            subject, obj = obj, subject
                            predicate = "acquired"
                        facts.append({
                            "subject": subject, "predicate": predicate,
                            "object": obj, "time": fact.get("time"),
                        })
                save(Path(f"{args.run_id}_gate_{did}.json"), gate)
                reasons = [r["reason"] for r in gate["needs_review"]]
            elif (
                state.status == "needs_review"
                and reasons
                and all(reason == "ambiguous_fact" for reason in reasons)
            ):
                status = "succeeded"
                review = True

            entry["prediction"] = {
                "document_id": did,
                "status": status,
                "review_required": review,
                "facts": facts,
                "session_id": state.session_id,
                "runtime_status": state.status,
                "reasons": reasons,
            }
            save(checkpoint_path, checkpoint)
            print(f"{did}: {status}; review={review}; facts={len(facts)}")

            # Stop the batch after an operational failure.
            if status == "failed":
                print("Batch stopped after failure. Review the saved session before retrying.")
                break
    finally:
        if store is not None:
            store.close()
        if adapter is not None:
            await adapter.close()

    predictions = [
        entry["prediction"]
        for entry in checkpoint["documents"].values()
        if entry.get("prediction") is not None
    ]
    save(predictions_path, predictions)
    # Gold becomes accessible only to the scoring step.
    save(scores_path, evaluate(dataset, predictions))
    print(f"Predictions: {predictions_path.resolve()}")
    print(f"Scores: {scores_path.resolve()}")
    print("Missing documents are included in scoring coverage; batch may be incomplete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("business_eval_v1.json"))
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--limit", type=int, default=1,
                        help="Maximum new/resumed documents in this invocation")
    parser.add_argument("--execute", action="store_true")
    # BUSINESS_RUN_ID_V1
    parser.add_argument("--run-id", required=True,
                        help="Unique prefix for this evaluation run")
    # SCOPED_ALIAS_BATCH_V1
    parser.add_argument("--document-id", action="append",
                        help="Only process these document IDs")
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("--run-id must contain only letters, digits, _ or -")
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be between 1 and 10")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    runtime_docs = [
        {
            "document_id": doc["document_id"],
            "source": doc["source"],
            "trusted_aliases": doc["trusted_aliases"],
        }
        for doc in dataset["documents"]
    ]
    ids = [doc["document_id"] for doc in runtime_docs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document IDs")
    for doc in runtime_docs:
        if not doc["document_id"].replace("_", "").isalnum():
            raise ValueError("Unsafe document ID")
        if not isinstance(doc["source"], str) or not doc["source"].strip():
            raise ValueError("Empty source")
        catalog_for(doc)

    if args.document_id:
        requested = set(args.document_id)
        if not requested <= set(ids):
            parser.error("Unknown --document-id")
        runtime_docs = [
            doc for doc in runtime_docs
            if doc["document_id"] in requested
        ]

    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    identity = {
        "dataset": dataset,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "harness_sources": source_hashes,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()

    if not args.execute:
        print(
            f"PREPARED: {len(runtime_docs)} documents; "
            "catalogs validated; 0 API calls."
        )
        print("No predictions or model scores generated.")
        return
    asyncio.run(run(args, dataset, runtime_docs, fingerprint))


if __name__ == "__main__":
    main()
