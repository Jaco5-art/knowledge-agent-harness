from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .evaluation import EvaluationSample, LabeledFact, load_jsonl


ALLOWED_STATUSES = {"needs_human_review", "approved", "rejected", "needs_revision", "public_benchmark"}


def audit_samples(samples: list[EvaluationSample]) -> dict:
    ids = Counter(sample.sample_id for sample in samples)
    texts = Counter(sample.text.casefold().strip() for sample in samples)
    issues: list[dict[str, str]] = []
    for sample in samples:
        if ids[sample.sample_id] > 1:
            issues.append({"sample_id": sample.sample_id, "issue": "duplicate_id"})
        if texts[sample.text.casefold().strip()] > 1:
            issues.append({"sample_id": sample.sample_id, "issue": "duplicate_text"})
        if sample.review_status not in ALLOWED_STATUSES:
            issues.append({"sample_id": sample.sample_id, "issue": "invalid_review_status"})
        if sample.review_status == "approved" and not sample.reviewed_by:
            issues.append({"sample_id": sample.sample_id, "issue": "approved_without_reviewer"})
        if not sample.gold_facts and "no_positive_fact" not in sample.phenomena:
            issues.append({"sample_id": sample.sample_id, "issue": "empty_gold_without_label"})
    status_counts = Counter(sample.review_status for sample in samples)
    return {
        "samples": len(samples),
        "issues": issues,
        "issue_count": len(issues),
        "review_status": dict(status_counts),
        "ready_for_final_reporting": bool(samples) and not issues and status_counts.get("approved", 0) == len(samples),
    }


def export_review_csv(samples: list[EvaluationSample], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "text", "gold_facts_json", "source_type", "difficulty", "phenomena", "review_status", "reviewed_by", "reviewer_notes"]
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "sample_id": sample.sample_id,
                "text": sample.text,
                "gold_facts_json": json.dumps([fact.model_dump() for fact in sample.gold_facts], ensure_ascii=False),
                "source_type": sample.source_type,
                "difficulty": sample.difficulty,
                "phenomena": ",".join(sample.phenomena),
                "review_status": sample.review_status,
                "reviewed_by": sample.reviewed_by or "",
                "reviewer_notes": sample.reviewer_notes or "",
            })
    return destination


def import_review_csv(path: str | Path) -> list[EvaluationSample]:
    samples = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            facts = [LabeledFact.model_validate(item) for item in json.loads(row["gold_facts_json"])]
            status = row["review_status"].strip()
            reviewed_by = row["reviewed_by"].strip() or None
            if status == "approved" and not reviewed_by:
                raise ValueError(f"{row['sample_id']} cannot be approved without reviewed_by.")
            samples.append(EvaluationSample(
                sample_id=row["sample_id"], text=row["text"], gold_facts=facts,
                source_type=row["source_type"], difficulty=row["difficulty"],
                phenomena=[item for item in row["phenomena"].split(",") if item],
                review_status=status, reviewed_by=reviewed_by,
                reviewer_notes=row["reviewer_notes"].strip() or None,
            ))
    return samples


def write_jsonl(samples: list[EvaluationSample], path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text("".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and manage benchmark review status.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--dataset", required=True)
    export_parser.add_argument("--csv", required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--csv", required=True)
    import_parser.add_argument("--dataset", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_review_csv(load_jsonl(args.dataset, EvaluationSample), args.csv)
    elif args.command == "import":
        samples = import_review_csv(args.csv)
        report = audit_samples(samples)
        if report["issues"]:
            raise SystemExit(json.dumps(report, indent=2))
        write_jsonl(samples, args.dataset)
    else:
        print(json.dumps(audit_samples(load_jsonl(args.dataset, EvaluationSample)), indent=2))


if __name__ == "__main__":
    main()
