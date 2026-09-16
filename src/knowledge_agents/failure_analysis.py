from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .evaluation import EvaluationSample
from .redocred_predictor import IndexedPrediction


def _relations_by_sample(predictions: list[IndexedPrediction]) -> dict[str, dict[tuple[int, int, str], str]]:
    return {
        prediction.sample_id: {
            relation.key(): relation.evidence_quote for relation in prediction.relations
        }
        for prediction in predictions
    }


def _category(gold: bool, baseline: bool, verification: bool, rag: bool) -> str:
    if gold:
        if baseline and verification and rag:
            return "persistent_true_positive"
        if not baseline and not verification and not rag:
            return "baseline_false_negative"
        if not baseline and (verification or rag):
            return "correction_recovered_true_positive"
        if baseline and not verification and rag:
            return "rag_prevented_false_rejection"
        if baseline and verification and not rag:
            return "rag_false_rejection"
        if baseline and not verification and not rag:
            return "verification_false_rejection"
        return "mixed_true_relation"
    if baseline and not verification and not rag:
        return "verification_removed_false_positive"
    if baseline and verification and not rag:
        return "rag_removed_false_positive"
    if baseline and rag:
        return "persistent_false_positive"
    if not baseline and (verification or rag):
        return "correction_created_false_positive"
    return "not_predicted"


def analyze_failures(
    samples: list[EvaluationSample],
    baseline: list[IndexedPrediction],
    verification_only: list[IndexedPrediction],
    rag_verification: list[IndexedPrediction],
    relation_map: dict[str, str],
) -> tuple[dict, list[dict]]:
    baseline_map = _relations_by_sample(baseline)
    verification_map = _relations_by_sample(verification_only)
    rag_map = _relations_by_sample(rag_verification)
    rows: list[dict] = []
    for sample in samples:
        gold = {
            (item["head_index"], item["tail_index"], item["relation_id"])
            for item in sample.benchmark_metadata.get("indexed_relations", [])
        }
        base = baseline_map.get(sample.sample_id, {})
        verified = verification_map.get(sample.sample_id, {})
        ragged = rag_map.get(sample.sample_id, {})
        entities = sample.benchmark_metadata.get("entities", [])
        universe = gold | set(base) | set(verified) | set(ragged)
        for head, tail, relation_id in sorted(universe):
            in_gold = (head, tail, relation_id) in gold
            in_baseline = (head, tail, relation_id) in base
            in_verification = (head, tail, relation_id) in verified
            in_rag = (head, tail, relation_id) in ragged
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "head_index": head,
                    "head_entity": entities[head] if head < len(entities) else "",
                    "tail_index": tail,
                    "tail_entity": entities[tail] if tail < len(entities) else "",
                    "relation_id": relation_id,
                    "relation_label": relation_map.get(relation_id, relation_id),
                    "gold": in_gold,
                    "baseline": in_baseline,
                    "verification_only": in_verification,
                    "rag_verification": in_rag,
                    "category": _category(in_gold, in_baseline, in_verification, in_rag),
                    "baseline_evidence": base.get((head, tail, relation_id), ""),
                    "verification_evidence": verified.get((head, tail, relation_id), ""),
                    "rag_evidence": ragged.get((head, tail, relation_id), ""),
                }
            )
    counts = Counter(row["category"] for row in rows)
    review_rows = [row for row in rows if row["category"] != "persistent_true_positive"]
    summary = {
        "category_counts": dict(sorted(counts.items())),
        "relations_analyzed": len(rows),
        "rows_requiring_review": len(review_rows),
        "verification_removed_baseline_false_positives": sum(
            1 for row in rows if not row["gold"] and row["baseline"] and not row["verification_only"]
        ),
        "rag_removed_baseline_false_positives": sum(
            1 for row in rows if not row["gold"] and row["baseline"] and not row["rag_verification"]
        ),
        "verification_lost_baseline_true_positives": sum(
            1 for row in rows if row["gold"] and row["baseline"] and not row["verification_only"]
        ),
        "rag_lost_baseline_true_positives": sum(
            1 for row in rows if row["gold"] and row["baseline"] and not row["rag_verification"]
        ),
        "verification_recovered_true_relations": sum(
            1 for row in rows if row["gold"] and not row["baseline"] and row["verification_only"]
        ),
        "rag_recovered_true_relations": sum(
            1 for row in rows if row["gold"] and not row["baseline"] and row["rag_verification"]
        ),
    }
    return summary, review_rows


def write_failure_csv(path: str | Path, rows: list[dict]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "head_index", "head_entity", "tail_index", "tail_entity",
        "relation_id", "relation_label", "gold", "baseline", "verification_only",
        "rag_verification", "category", "baseline_evidence",
        "verification_evidence", "rag_evidence",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination
