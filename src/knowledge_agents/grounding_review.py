from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .evaluation import EvaluationSample, load_jsonl


GROUNDING_LABELS = {
    "directly_supported",
    "inferred_or_world_knowledge",
    "annotation_anomaly",
    "uncertain",
}


def _strict_grounding_metrics(rows: list[dict], verdict_field: str) -> dict:
    """Score whether a verifier accepts only facts directly supported by the text."""
    tp = fp = tn = fn = 0
    for row in rows:
        expected_supported = row.get("human_grounding_label") == "directly_supported"
        predicted_supported = row.get(verdict_field) == "supported"
        if expected_supported and predicted_supported:
            tp += 1
        elif expected_supported:
            fn += 1
        elif predicted_supported:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {
        "sample_count": total,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
    }


def _case_summary(row: dict) -> dict:
    return {
        "sample_id": row.get("sample_id", ""),
        "head_entity": row.get("head_entity", ""),
        "relation": row.get("relation_label", ""),
        "tail_entity": row.get("tail_entity", ""),
        "human_label": row.get("human_grounding_label", ""),
        "verification_only_verdict": row.get("verification_only_verdict", ""),
        "rag_verdict": row.get("rag_verdict", ""),
    }


def load_trace(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_review_rows(
    samples: list[EvaluationSample],
    trace_rows: list[dict],
    relation_map: dict[str, str],
) -> list[dict]:
    samples_by_id = {sample.sample_id: sample for sample in samples}
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in trace_rows:
        if row.get("case_type") != "clean":
            continue
        key = (
            row["sample_id"], row["head_index"], row["tail_index"],
            row["candidate_relation_id"], row["evidence_quote"],
        )
        grouped[key][row["condition"]] = row

    output = []
    for key, conditions in sorted(grouped.items()):
        sample_id, head, tail, relation_id, evidence_quote = key
        sample = samples_by_id[sample_id]
        entities = sample.benchmark_metadata.get("entities", [])
        no_rag = conditions.get("verification_only", {})
        rag = conditions.get("rag_verification", {})
        output.append(
            {
                "sample_id": sample_id,
                "document_title": sample.benchmark_metadata.get("title", ""),
                "head_index": head,
                "head_entity": entities[head] if head < len(entities) else "",
                "relation_id": relation_id,
                "relation_label": relation_map.get(relation_id, relation_id),
                "tail_index": tail,
                "tail_entity": entities[tail] if tail < len(entities) else "",
                "annotated_evidence": evidence_quote,
                "source_document": sample.text,
                "verification_only_verdict": no_rag.get("verdict", ""),
                "verification_only_reason": no_rag.get("reason", ""),
                "rag_verdict": rag.get("verdict", ""),
                "rag_reason": rag.get("reason", ""),
                "review_status": "needs_human_review",
                "human_grounding_label": "",
                "reviewed_by": "",
                "reviewer_notes": "",
            }
        )
    return output


def write_review_csv(path: str | Path, rows: list[dict]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No clean challenge cases were found in the trace.")
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def audit_review_csv(path: str | Path) -> dict:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    issues = []
    label_counts = Counter()
    approved_label_counts = Counter()
    status_counts = Counter()
    for line_number, row in enumerate(rows, start=2):
        status = row.get("review_status", "")
        label = row.get("human_grounding_label", "")
        reviewer = row.get("reviewed_by", "").strip()
        status_counts[status or "missing"] += 1
        if status == "approved":
            if label not in GROUNDING_LABELS:
                issues.append(f"row {line_number}: approved row has invalid grounding label")
            if not reviewer:
                issues.append(f"row {line_number}: approved row has no reviewer")
            if label in GROUNDING_LABELS:
                approved_label_counts[label] += 1
        elif status not in {"needs_human_review", "needs_revision", "rejected"}:
            issues.append(f"row {line_number}: invalid review status {status!r}")
        label_counts[label or "unlabeled"] += 1

    approved_rows = status_counts["approved"]
    approved = [row for row in rows if row.get("review_status") == "approved"]
    review_coverage = approved_rows / len(rows) if rows else 0.0
    approved_rows_valid = approved_rows > 0 and not any(
        issue.startswith("row") and "approved row" in issue for issue in issues
    )
    verification_only_metrics = _strict_grounding_metrics(
        approved, "verification_only_verdict"
    )
    rag_metrics = _strict_grounding_metrics(approved, "rag_verdict")
    negative_cases = [
        row for row in approved
        if row.get("human_grounding_label") != "directly_supported"
    ]
    rag_removed_false_positives = [
        _case_summary(row) for row in negative_cases
        if row.get("verification_only_verdict") == "supported"
        and row.get("rag_verdict") != "supported"
    ]
    persistent_rag_false_positives = [
        _case_summary(row) for row in negative_cases
        if row.get("verification_only_verdict") == "supported"
        and row.get("rag_verdict") == "supported"
    ]
    rag_introduced_false_positives = [
        _case_summary(row) for row in negative_cases
        if row.get("verification_only_verdict") != "supported"
        and row.get("rag_verdict") == "supported"
    ]
    return {
        "rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "approved_label_counts": dict(sorted(approved_label_counts.items())),
        "human_review": {
            "approved_rows": approved_rows,
            "coverage": round(review_coverage, 4),
            "approved_rows_valid": approved_rows_valid,
            "interpretation": (
                "valid_partial_human_review" if approved_rows_valid and approved_rows < len(rows)
                else "complete_human_review" if approved_rows_valid and approved_rows == len(rows)
                else "no_valid_human_review"
            ),
        },
        "human_grounded_verifier_evaluation": {
            "policy": (
                "Only directly_supported is a positive text-grounding case; "
                "inferred/world-knowledge, annotation anomalies, and uncertain cases "
                "must not be accepted as directly supported."
            ),
            "verification_only": verification_only_metrics,
            "rag_verification": rag_metrics,
            "rag_vs_verification_only": {
                "accuracy_delta": round(
                    rag_metrics["accuracy"] - verification_only_metrics["accuracy"], 4
                ),
                "precision_delta": round(
                    rag_metrics["precision"] - verification_only_metrics["precision"], 4
                ),
                "false_positives_removed": (
                    verification_only_metrics["false_positive"]
                    - rag_metrics["false_positive"]
                ),
                "true_positives_lost": (
                    verification_only_metrics["true_positive"]
                    - rag_metrics["true_positive"]
                ),
            },
            "case_analysis": {
                "rag_removed_false_positives": rag_removed_false_positives,
                "persistent_rag_false_positives": persistent_rag_false_positives,
                "rag_introduced_false_positives": rag_introduced_false_positives,
            },
        },
        "issues": issues,
        "ready_for_reporting": bool(rows) and not issues and all(
            row.get("review_status") == "approved" for row in rows
        ),
    }


def markdown_audit_report(report: dict) -> str:
    human = report["human_review"]
    evaluation = report["human_grounded_verifier_evaluation"]
    verification = evaluation["verification_only"]
    rag = evaluation["rag_verification"]
    comparison = evaluation["rag_vs_verification_only"]
    cases = evaluation["case_analysis"]
    labels = report["approved_label_counts"]
    return f"""# Human-Grounded Verification Report

Status: **{human['interpretation']}**

## Review scope

- Reviewed relations: {human['approved_rows']} / {report['rows']}
- Human-review coverage: {human['coverage']:.2%}
- Directly supported: {labels.get('directly_supported', 0)}
- Inferred or world knowledge: {labels.get('inferred_or_world_knowledge', 0)}
- Annotation anomalies: {labels.get('annotation_anomaly', 0)}
- Uncertain: {labels.get('uncertain', 0)}
- Ready for reporting within this reviewed challenge set: {str(report['ready_for_reporting']).lower()}

## Strict text-grounding evaluation

Only `directly_supported` is treated as a positive case. All other labels must
not be accepted as directly supported by the supplied text.

| Condition | Accuracy | Precision | Recall | Specificity | False positives |
|---|---:|---:|---:|---:|---:|
| Verification only | {verification['accuracy']:.2%} | {verification['precision']:.2%} | {verification['recall']:.2%} | {verification['specificity']:.2%} | {verification['false_positive']} |
| RAG verification | {rag['accuracy']:.2%} | {rag['precision']:.2%} | {rag['recall']:.2%} | {rag['specificity']:.2%} | {rag['false_positive']} |

## Observed difference

- Accuracy delta: {comparison['accuracy_delta']:+.2%}
- Precision delta: {comparison['precision_delta']:+.2%}
- False positives removed: {comparison['false_positives_removed']}
- True positives lost: {comparison['true_positives_lost']}

## Case analysis

- Verification false positives removed by RAG: {len(cases['rag_removed_false_positives'])}
- False positives remaining after RAG: {len(cases['persistent_rag_false_positives'])}
- False positives introduced by RAG: {len(cases['rag_introduced_false_positives'])}

### Examples corrected by RAG

{chr(10).join(f"- {case['head_entity']} — {case['relation']} — {case['tail_entity']} ({case['human_label']})" for case in cases['rag_removed_false_positives']) or '- None'}

### Remaining false-positive examples

{chr(10).join(f"- {case['head_entity']} — {case['relation']} — {case['tail_entity']} ({case['human_label']})" for case in cases['persistent_rag_false_positives']) or '- None'}

## Reporting boundary

These metrics describe the human-reviewed text-grounding challenge cases only.
They are separate from end-to-end relation-extraction Precision, Recall, and F1,
and must not be presented as results from the 100-document formal experiment.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or audit the text-grounding human review sheet.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--dataset", required=True)
    export.add_argument("--trace", required=True)
    export.add_argument("--relation-map", required=True)
    export.add_argument("--output", default="outputs/redocred/grounding_review.csv")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--csv", required=True)
    audit.add_argument("--output", help="Optionally save the audit result as JSON.")
    audit.add_argument(
        "--markdown-output", help="Optionally save a human-readable Markdown report."
    )
    args = parser.parse_args()
    if args.command == "export":
        samples = load_jsonl(args.dataset, EvaluationSample)
        relation_map = json.loads(Path(args.relation_map).read_text(encoding="utf-8"))
        rows = build_review_rows(samples, load_trace(args.trace), relation_map)
        print(f"Wrote {len(rows)} cases to {write_review_csv(args.output, rows)}")
    else:
        report = audit_review_csv(args.csv)
        rendered = json.dumps(report, indent=2)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered + "\n", encoding="utf-8")
        if args.markdown_output:
            destination = Path(args.markdown_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(markdown_audit_report(report), encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
