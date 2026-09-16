
"""Score business predictions; never invokes a model."""
import argparse
import json
from pathlib import Path

def key(fact):
    return tuple(fact.get(k) for k in ("subject", "predicate", "object", "time"))

def ratio(a, b):
    return a / b if b else None

def evaluate(dataset, predictions):
    docs = {d["document_id"]: d for d in dataset["documents"]}
    if len(docs) != len(dataset["documents"]):
        raise ValueError("Duplicate dataset document IDs")
    indexed = {}
    for prediction in predictions:
        did = prediction["document_id"]
        if did not in docs or did in indexed:
            raise ValueError("Unknown or duplicate prediction document ID")
        indexed[did] = prediction

    groups = {
        name: dict(
            expected_documents=0, scored_documents=0, failed_documents=0,
            tp=0, fp=0, fn=0, duplicates=0,
            forbidden_outputs=0, review_correct=0, exact_documents=0,
        )
        for name in ("clean", "noisy")
    }
    details = []
    for did, doc in docs.items():
        group = groups[doc["variant"]]
        group["expected_documents"] += 1
        prediction = indexed.get(did)
        if prediction is None or prediction.get("status") != "succeeded":
            group["failed_documents"] += 1
            details.append({"document_id": did, "scored": False})
            continue

        if type(prediction.get("review_required")) is not bool:
            raise ValueError(f"{did}: review_required must be boolean")
        facts = prediction.get("facts")
        if not isinstance(facts, list):
            raise ValueError(f"{did}: facts must be a list")
        for fact in facts:
            if not isinstance(fact, dict):
                raise ValueError(f"{did}: each fact must be an object")
            for field in ("subject", "predicate", "object"):
                if not isinstance(fact.get(field), str) or not fact[field].strip():
                    raise ValueError(f"{did}: invalid {field}")
            if fact.get("time") is not None and not isinstance(fact["time"], str):
                raise ValueError(f"{did}: time must be string or null")

        actual = {key(f) for f in facts}
        gold = {key(f) for f in doc["gold"]["positive_facts"]}
        tp, fp, fn = len(actual & gold), len(actual - gold), len(gold - actual)
        duplicates = len(facts) - len(actual)
        forbidden = {
            tuple(item) for item in doc["gold"]["forbidden_positive_relations"]
        }
        prohibited = sum(f[:3] in forbidden for f in actual)
        review_ok = prediction["review_required"] == doc["gold"]["review_required"]
        exact = actual == gold and review_ok and duplicates == 0

        group["scored_documents"] += 1
        for field, value in (
            ("tp", tp), ("fp", fp), ("fn", fn),
            ("duplicates", duplicates), ("forbidden_outputs", prohibited),
            ("review_correct", int(review_ok)), ("exact_documents", int(exact)),
        ):
            group[field] += value
        details.append({
            "document_id": did, "scored": True,
            "tp": tp, "fp": fp, "fn": fn,
            "duplicate_outputs": duplicates,
            "forbidden_outputs": prohibited,
            "review_correct": review_ok,
            "exact": exact,
        })

    for group in groups.values():
        tp, fp, fn = group["tp"], group["fp"], group["fn"]
        group["precision"] = ratio(tp, tp + fp)
        group["recall"] = ratio(tp, tp + fn)
        group["f1"] = ratio(2 * tp, 2 * tp + fp + fn)
        group["review_accuracy"] = ratio(
            group["review_correct"], group["scored_documents"]
        )
        group["exact_success_rate_all_documents"] = ratio(
            group["exact_documents"], group["expected_documents"]
        )

    return {
        "dataset_version": dataset["version"],
        "groups": groups,
        "documents": details,
        "notes": [
            "Exact canonical subject/predicate/object/time matching.",
            "Failed or missing runs excluded from relation metrics; coverage reported.",
            "Failed or missing runs count as unsuccessful in all-document exact success.",
            "Forbidden outputs are not a complete unsupported-fact metric.",
            "Evidence entailment is not scored.",
            "Metrics concern this synthetic development set only.",
        ],
    }

def self_test():
    fact = {"subject": "A", "predicate": "invested_in", "object": "B", "time": "2023"}
    dataset = {
        "version": "unit-fixture",
        "documents": [{
            "document_id": "one", "variant": "clean",
            "gold": {
                "positive_facts": [fact],
                "forbidden_positive_relations": [["A", "acquired", "B"]],
                "review_required": False,
            },
        }],
    }
    prediction = {
        "document_id": "one", "status": "succeeded",
        "facts": [fact], "review_required": False,
    }
    result = evaluate(dataset, [prediction])["groups"]["clean"]
    assert result["f1"] == 1 and result["exact_documents"] == 1

    wrong = dict(prediction, facts=[dict(fact, time="2024")])
    result = evaluate(dataset, [wrong])["groups"]["clean"]
    assert result["tp"] == 0 and result["fp"] == result["fn"] == 1

    wrong = dict(prediction, facts=[dict(fact, predicate="acquired")])
    assert evaluate(dataset, [wrong])["groups"]["clean"]["forbidden_outputs"] == 1

    repeated = dict(prediction, facts=[fact, fact])
    result = evaluate(dataset, [repeated])["groups"]["clean"]
    assert result["duplicates"] == 1 and result["exact_documents"] == 0

    result = evaluate(dataset, [])["groups"]["clean"]
    assert result["precision"] is None
    assert result["failed_documents"] == 1
    assert result["exact_success_rate_all_documents"] == 0

    try:
        evaluate(dataset, [prediction, prediction])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate prediction IDs accepted")

    print("SCORER OK: offline scoring checks passed; no model scores generated.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("business_eval_v1.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.predictions or not args.output:
        parser.error("Provide --predictions and --output")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    result = evaluate(dataset, predictions)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"SCORED: {args.output.resolve()}")

if __name__ == "__main__":
    main()
