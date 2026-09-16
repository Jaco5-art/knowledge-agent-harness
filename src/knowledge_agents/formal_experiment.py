from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .evaluation import EvaluationSample, load_jsonl
from .failure_analysis import analyze_failures, write_failure_csv
from .redocred_predictor import (
    IndexedPrediction,
    OpenAIConstrainedRelationPredictor,
    gold_prediction,
    score_indexed,
)
from .redocred_rag import FAISSRelationEvidenceRetriever, retrieval_metrics
from .redocred_verification import OpenAIRelationVerifier, RelationVerifier, verify_and_correct

WORKFLOW_VERSION = "correction-suggestion-cycle-safe-v3"


class RelationPredictor(Protocol):
    def predict(self, sample: EvaluationSample) -> IndexedPrediction: ...


def _write_jsonl(path: Path, records: list[IndexedPrediction]) -> None:
    path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")


def _write_verification_trace(path: Path, records: list[IndexedPrediction]) -> None:
    rows = (
        {"sample_id": record.sample_id, **decision}
        for record in records
        for decision in record.verification_trace
    )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _run_checkpointed(
    name: str,
    samples: list[EvaluationSample],
    path: Path,
    producer,
    *,
    resume: bool,
) -> list[IndexedPrediction]:
    expected_ids = {sample.sample_id for sample in samples}
    completed: dict[str, IndexedPrediction] = {}
    if resume and path.exists():
        records = load_jsonl(path, IndexedPrediction)
        for record in records:
            if record.sample_id not in expected_ids:
                raise ValueError(f"Checkpoint {path} contains an unexpected sample: {record.sample_id}")
            if record.sample_id in completed:
                raise ValueError(f"Checkpoint {path} contains duplicate sample: {record.sample_id}")
            completed[record.sample_id] = record
    else:
        path.write_text("", encoding="utf-8")

    for position, sample in enumerate(samples, start=1):
        if sample.sample_id in completed:
            print(f"[{name}] {position}/{len(samples)} {sample.sample_id} (checkpoint)")
            continue
        prediction = producer(sample)
        if prediction.sample_id != sample.sample_id:
            raise ValueError(f"{name} returned the wrong sample ID for {sample.sample_id}.")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(prediction.model_dump_json() + "\n")
        completed[sample.sample_id] = prediction
        print(f"[{name}] {position}/{len(samples)} {sample.sample_id} saved")
    return [completed[sample.sample_id] for sample in samples]


def _sample_fingerprint(samples: list[EvaluationSample]) -> str:
    payload = "\n".join(sample.model_dump_json() for sample in samples).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_external_baseline(
    path: str | Path, samples: list[EvaluationSample]
) -> list[IndexedPrediction]:
    source = Path(path)
    records = load_jsonl(source, IndexedPrediction)
    by_id: dict[str, IndexedPrediction] = {}
    expected_ids = {sample.sample_id for sample in samples}
    for record in records:
        if record.sample_id not in expected_ids:
            raise ValueError(f"External baseline contains an unexpected sample: {record.sample_id}")
        if record.sample_id in by_id:
            raise ValueError(f"External baseline contains duplicate sample: {record.sample_id}")
        by_id[record.sample_id] = record
    missing = expected_ids - set(by_id)
    if missing:
        raise ValueError(f"External baseline is missing samples: {sorted(missing)}")
    return [by_id[sample.sample_id] for sample in samples]


def _validate_or_write_run_config(path: Path, config: dict, resume: bool) -> None:
    if resume:
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume without {path}.")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                "Resume settings do not match the checkpoint. Use the original "
                "settings or choose a new output directory."
            )
        return
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _delta(before: dict, after: dict) -> dict[str, float | int]:
    return {
        "precision_delta": round(after["precision"] - before["precision"], 4),
        "recall_delta": round(after["recall"] - before["recall"], 4),
        "f1_delta": round(after["f1"] - before["f1"], 4),
        "false_positives_removed": before["false_positive"] - after["false_positive"],
        "true_positives_lost": before["true_positive"] - after["true_positive"],
    }


def _workflow_actions(records: list[IndexedPrediction]) -> dict[str, int]:
    traces = [decision for record in records for decision in record.verification_trace]
    verification = [item for item in traces if item.get("phase", "verification") == "verification"]
    suggestions = [item for item in traces if item.get("phase") == "correction_suggestion"]
    return {
        "verification_decisions": len(verification),
        "supported_decisions": sum(item.get("verdict") == "supported" for item in verification),
        "unsupported_decisions": sum(item.get("verdict") == "unsupported" for item in verification),
        "needs_correction_decisions": sum(item.get("verdict") == "needs_correction" for item in verification),
        "correction_suggestion_decisions": len(suggestions),
        "corrections_proposed": sum(
            item.get("verdict") == "needs_correction" and bool(item.get("corrected_relation_id"))
            for item in traces
        ),
        "corrections_blocked_as_cycles_or_duplicates": sum(
            item.get("correction_blocked_reason") == "correction_cycle_or_duplicate"
            for item in traces
        ),
        "corrected_candidates_reverified": sum(
            item.get("phase", "verification") == "verification" and item.get("round_index", 0) > 0
            for item in traces
        ),
        "corrected_candidates_accepted": sum(
            item.get("phase", "verification") == "verification"
            and item.get("round_index", 0) > 0
            and item.get("verdict") == "supported"
            for item in traces
        ),
    }


def _markdown_report(report: dict) -> str:
    conditions = report["conditions"]
    rows = []
    for name in ("baseline", "verification_only", "rag_verification"):
        metrics = conditions[name]
        rows.append(
            f"| {name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['false_positive']} | {metrics['false_negative']} |"
        )
    retrieval = report["retrieval"]
    failure = report["failure_analysis"]
    actions = report["workflow_actions"]
    return f"""# Re-DocRED Formal Experiment Report

Status: **machine-generated; inspect before using results in a resume**

## Reproducibility

- Run time (UTC): {report['run']['created_at_utc']}
- Samples: {report['run']['sample_count']}
- Dataset SHA-256: `{report['run']['dataset_sha256']}`
- LLM model: `{report['run']['llm_model']}`
- Workflow version: `{report['run']['workflow_version']}`
- Embedding model: `{report['run']['embedding_model']}`
- Top K: {report['run']['top_k']}
- Chunk size / overlap: {report['run']['chunk_size']} / {report['run']['chunk_overlap']}

## Relation extraction results

| Condition | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Retrieval

- Evidence Recall@{retrieval['top_k']}: {retrieval['evidence_recall_at_k']:.4f}
- Covered annotated relations: {retrieval['covered_relations']} / {retrieval['gold_relations_with_evidence']}

## Failure analysis

- Verification removed baseline false positives: {failure['verification_removed_baseline_false_positives']}
- RAG removed baseline false positives: {failure['rag_removed_baseline_false_positives']}
- Verification lost baseline true positives: {failure['verification_lost_baseline_true_positives']}
- RAG lost baseline true positives: {failure['rag_lost_baseline_true_positives']}
- Relations requiring human review: {failure['rows_requiring_review']}

See `failure_cases.csv` for entity names, relation labels, condition membership
and evidence quotes for each non-persistent result.

## Verification and correction actions

| Condition | Unsupported | Correction suggestions | Corrections proposed | Cycles blocked | Reverified | Accepted after correction |
|---|---:|---:|---:|---:|---:|---:|
| Verification only | {actions['verification_only']['unsupported_decisions']} | {actions['verification_only']['correction_suggestion_decisions']} | {actions['verification_only']['corrections_proposed']} | {actions['verification_only']['corrections_blocked_as_cycles_or_duplicates']} | {actions['verification_only']['corrected_candidates_reverified']} | {actions['verification_only']['corrected_candidates_accepted']} |
| RAG verification | {actions['rag_verification']['unsupported_decisions']} | {actions['rag_verification']['correction_suggestion_decisions']} | {actions['rag_verification']['corrections_proposed']} | {actions['rag_verification']['corrections_blocked_as_cycles_or_duplicates']} | {actions['rag_verification']['corrected_candidates_reverified']} | {actions['rag_verification']['corrected_candidates_accepted']} |

## Interpretation guardrail

These numbers are valid only for the dataset fingerprint and settings above.
Do not describe an improvement if the relevant delta in `report.json` is zero
or negative. Human review of failure cases remains required before publishing
resume claims.
"""


def run_formal_experiment(
    samples: list[EvaluationSample],
    predictor: RelationPredictor,
    verifier: RelationVerifier,
    retriever: FAISSRelationEvidenceRetriever,
    output_directory: str | Path,
    *,
    llm_model: str,
    embedding_model: str,
    relation_map: dict[str, str] | None = None,
    created_at: datetime | None = None,
    resume: bool = False,
    dataset_sample_count: int | None = None,
    baseline_checkpoint: str | Path | None = None,
) -> dict:
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not samples:
        raise ValueError("The formal experiment requires at least one sample.")
    run_config = {
        "workflow_version": WORKFLOW_VERSION,
        "dataset_sha256": _sample_fingerprint(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "top_k": retriever.top_k,
        "chunk_size": retriever.chunk_size,
        "chunk_overlap": retriever.overlap,
        "baseline_source_sha256": (
            _file_sha256(Path(baseline_checkpoint)) if baseline_checkpoint else None
        ),
    }
    _validate_or_write_run_config(output_dir / "run_config.json", run_config, resume)
    baseline_path = output_dir / "baseline_predictions.jsonl"
    verification_path = output_dir / "verification_only_predictions.jsonl"
    rag_path = output_dir / "rag_verification_predictions.jsonl"
    if baseline_checkpoint:
        baseline = _load_external_baseline(baseline_checkpoint, samples)
        _write_jsonl(baseline_path, baseline)
        print(f"[baseline] reused {len(baseline)} predictions from {baseline_checkpoint}")
    else:
        baseline = _run_checkpointed(
            "baseline", samples, baseline_path, predictor.predict, resume=resume
        )
    baseline_by_id = {prediction.sample_id: prediction for prediction in baseline}
    verification_only = _run_checkpointed(
        "verification_only",
        samples,
        verification_path,
        lambda sample: verify_and_correct(sample, baseline_by_id[sample.sample_id], verifier),
        resume=resume,
    )
    rag_verification = _run_checkpointed(
        "rag_verification",
        samples,
        rag_path,
        lambda sample: verify_and_correct(
            sample,
            baseline_by_id[sample.sample_id],
            verifier,
            evidence_retriever=retriever,
        ),
        resume=resume,
    )
    gold = [gold_prediction(sample) for sample in samples]
    baseline_metrics = score_indexed(gold, baseline)
    verification_metrics = score_indexed(gold, verification_only)
    rag_metrics = score_indexed(gold, rag_verification)
    failure_summary, failure_rows = analyze_failures(
        samples,
        baseline,
        verification_only,
        rag_verification,
        relation_map or retriever.relation_map,
    )
    report = {
        "run": {
            "created_at_utc": (created_at or datetime.now(timezone.utc)).isoformat(),
            "sample_count": len(samples),
            "dataset_sample_count": dataset_sample_count or len(samples),
            "run_scope": "smoke_test" if len(samples) < (dataset_sample_count or len(samples)) else "formal_full",
            "dataset_sha256": _sample_fingerprint(samples),
            "llm_model": llm_model,
            "workflow_version": WORKFLOW_VERSION,
            "embedding_model": embedding_model,
            "top_k": retriever.top_k,
            "chunk_size": retriever.chunk_size,
            "chunk_overlap": retriever.overlap,
            "baseline_source_sha256": run_config["baseline_source_sha256"],
        },
        "conditions": {
            "baseline": baseline_metrics,
            "verification_only": verification_metrics,
            "rag_verification": rag_metrics,
        },
        "comparisons": {
            "verification_vs_baseline": _delta(baseline_metrics, verification_metrics),
            "rag_verification_vs_baseline": _delta(baseline_metrics, rag_metrics),
            "rag_vs_verification_only": _delta(verification_metrics, rag_metrics),
        },
        "retrieval": retrieval_metrics(samples, retriever),
        "workflow_actions": {
            "verification_only": _workflow_actions(verification_only),
            "rag_verification": _workflow_actions(rag_verification),
        },
        "failure_analysis": failure_summary,
    }
    # Rewrite checkpoints in canonical dataset order after successful completion.
    _write_jsonl(baseline_path, baseline)
    _write_jsonl(verification_path, verification_only)
    _write_jsonl(rag_path, rag_verification)
    _write_verification_trace(output_dir / "verification_only_trace.jsonl", verification_only)
    _write_verification_trace(output_dir / "rag_verification_trace.jsonl", rag_verification)
    write_failure_csv(output_dir / "failure_cases.csv", failure_rows)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three formal Re-DocRED experiment conditions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--relation-map", required=True)
    parser.add_argument("--output", default="outputs/redocred/formal")
    parser.add_argument("--model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--chunk-overlap", type=int, default=75)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N frozen samples (for smoke tests).")
    parser.add_argument("--resume", action="store_true", help="Reuse per-condition JSONL checkpoints in the output directory.")
    parser.add_argument(
        "--baseline-checkpoint",
        help="Reuse a compatible baseline_predictions.jsonl from an earlier workflow run.",
    )
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set before running the formal experiment.")

    from .rag import OpenAIEmbeddingProvider

    all_samples = load_jsonl(args.dataset, EvaluationSample)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero.")
    samples = all_samples[: args.limit] if args.limit is not None else all_samples
    relation_map = json.loads(Path(args.relation_map).read_text(encoding="utf-8"))
    predictor = OpenAIConstrainedRelationPredictor(relation_map, args.model)
    verifier = OpenAIRelationVerifier(relation_map, args.model)
    embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    retriever = FAISSRelationEvidenceRetriever(
        OpenAIEmbeddingProvider(embedding_model),
        relation_map,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    report = run_formal_experiment(
        samples,
        predictor,
        verifier,
        retriever,
        args.output,
        llm_model=args.model or os.getenv("LLM_MODEL", "gpt-5-mini"),
        embedding_model=embedding_model,
        relation_map=relation_map,
        resume=args.resume,
        dataset_sample_count=len(all_samples),
        baseline_checkpoint=args.baseline_checkpoint,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
