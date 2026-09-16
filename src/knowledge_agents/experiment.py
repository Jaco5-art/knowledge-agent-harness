from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluation import EvaluationSample, LabeledFact, PredictionRecord, compare_systems, load_jsonl
from .models import ExtractedFact, FactType
from .provider import ExtractionProvider, OpenAIProvider
from .rag import EmbeddingProvider, FAISSKnowledgeBase, OpenAIEmbeddingProvider
from .workflow import run_workflow


def _prediction(sample_id: str, facts: list[ExtractedFact]) -> PredictionRecord:
    labeled = [
        LabeledFact(subject=fact.subject, predicate=fact.predicate, object=fact.object)
        for fact in facts
        if fact.fact_type in (FactType.EVENT, FactType.RELATIONSHIP)
    ]
    return PredictionRecord(sample_id=sample_id, facts=labeled)


def run_experiment(
    samples: list[EvaluationSample],
    provider: ExtractionProvider,
    embedder: EmbeddingProvider,
    output_directory: str | Path,
    top_k: int = 3,
) -> dict:
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_records: list[PredictionRecord] = []
    verified_records: list[PredictionRecord] = []

    for sample in samples:
        baseline_facts = provider.extract(sample.text, FactType.EVENT)
        baseline_facts += provider.extract(sample.text, FactType.RELATIONSHIP)
        baseline_records.append(_prediction(sample.sample_id, baseline_facts))

        knowledge_base = FAISSKnowledgeBase(embedder)
        knowledge_base.add_documents({sample.sample_id: sample.text})
        state = run_workflow(
            sample.text,
            "extract events and relationships",
            provider,
            knowledge_base=knowledge_base,
        )
        verified_records.append(_prediction(sample.sample_id, state.get("verified_facts", [])))

    baseline_path = output_dir / "baseline_predictions.jsonl"
    verified_path = output_dir / "verified_predictions.jsonl"
    summary_path = output_dir / "metrics.json"
    baseline_path.write_text("".join(record.model_dump_json() + "\n" for record in baseline_records), encoding="utf-8")
    verified_path.write_text("".join(record.model_dump_json() + "\n" for record in verified_records), encoding="utf-8")
    metrics = compare_systems(samples, baseline_records, verified_records)
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and verification-enhanced extraction experiments.")
    parser.add_argument("--dataset", default="data/eval_samples_synthetic_100.jsonl")
    parser.add_argument("--output", default="outputs/latest")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set before running the LLM experiment.")
    samples = load_jsonl(args.dataset, EvaluationSample)
    metrics = run_experiment(samples, OpenAIProvider(args.model), OpenAIEmbeddingProvider(), args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

