from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field


class LabeledFact(BaseModel):
    subject: str
    predicate: str
    object: str | None = None

    def key(self) -> tuple[str, str, str]:
        return tuple((value or "").casefold().strip() for value in (self.subject, self.predicate, self.object))


class EvaluationSample(BaseModel):
    sample_id: str
    text: str
    gold_facts: list[LabeledFact] = Field(default_factory=list)
    source_type: str = "unspecified"
    difficulty: str = "unspecified"
    phenomena: list[str] = Field(default_factory=list)
    review_status: str = "needs_human_review"
    reviewed_by: str | None = None
    reviewer_notes: str | None = None
    benchmark_metadata: dict = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    sample_id: str
    facts: list[LabeledFact] = Field(default_factory=list)


def load_jsonl(path: str | Path, model):
    with Path(path).open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def score_predictions(
    samples: Iterable[EvaluationSample],
    predictions: Iterable[PredictionRecord],
) -> dict[str, float | int]:
    predicted_by_id = {record.sample_id: {fact.key() for fact in record.facts} for record in predictions}
    true_positive = false_positive = false_negative = 0
    sample_count = 0
    for sample in samples:
        sample_count += 1
        gold = {fact.key() for fact in sample.gold_facts}
        predicted = predicted_by_id.get(sample.sample_id, set())
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": sample_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compare_systems(
    samples: list[EvaluationSample],
    baseline: list[PredictionRecord],
    verified: list[PredictionRecord],
) -> dict:
    baseline_metrics = score_predictions(samples, baseline)
    verified_metrics = score_predictions(samples, verified)
    baseline_fp = baseline_metrics["false_positive"]
    removed_fp = baseline_fp - verified_metrics["false_positive"]
    error_detection_rate = removed_fp / baseline_fp if baseline_fp else 0.0
    return {
        "baseline": baseline_metrics,
        "verified_system": verified_metrics,
        "false_positives_removed": removed_fp,
        "error_detection_rate": round(error_detection_rate, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score baseline and verified extraction predictions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--verified", required=True)
    args = parser.parse_args()
    samples = load_jsonl(args.dataset, EvaluationSample)
    baseline = load_jsonl(args.baseline, PredictionRecord)
    verified = load_jsonl(args.verified, PredictionRecord)
    print(json.dumps(compare_systems(samples, baseline, verified), indent=2))


if __name__ == "__main__":
    main()
