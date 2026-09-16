from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model

from .evaluation import EvaluationSample, load_jsonl


class IndexedRelation(BaseModel):
    head_index: int = Field(ge=0)
    tail_index: int = Field(ge=0)
    relation_id: str
    evidence_quote: str

    def key(self) -> tuple[int, int, str]:
        return self.head_index, self.tail_index, self.relation_id


class IndexedPrediction(BaseModel):
    sample_id: str
    relations: list[IndexedRelation] = Field(default_factory=list)
    verification_trace: list[dict] = Field(default_factory=list)


def build_output_model(allowed_relation_ids: list[str]):
    if not allowed_relation_ids:
        raise ValueError("At least one relation ID is required.")
    relation_literal = Literal.__getitem__(tuple(sorted(set(allowed_relation_ids))))
    constrained_relation = create_model(
        "ConstrainedIndexedRelation",
        head_index=(int, Field(ge=0)),
        tail_index=(int, Field(ge=0)),
        relation_id=(relation_literal, ...),
        evidence_quote=(str, ...),
    )
    return create_model(
        "ConstrainedRelationBatch",
        relations=(list[constrained_relation], Field(default_factory=list)),
    )


class OpenAIConstrainedRelationPredictor:
    def __init__(self, relation_map: dict[str, str], model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install the OpenAI extra: pip install -e ".[openai]"') from exc
        self.client = OpenAI()
        self.model = model or os.getenv("LLM_MODEL", "gpt-5-mini")
        self.relation_map = relation_map
        self.output_model = build_output_model(list(relation_map))

    def predict(self, sample: EvaluationSample) -> IndexedPrediction:
        entities = sample.benchmark_metadata.get("entities", [])
        entity_catalog = "\n".join(f"{index}: {name}" for index, name in enumerate(entities))
        relation_catalog = "\n".join(f"{relation_id}: {label}" for relation_id, label in self.relation_map.items())
        prompt = f"""DOCUMENT:\n{sample.text}\n\nENTITIES:\n{entity_catalog}\n\nRELATIONS:\n{relation_catalog}"""
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Extract document-level relations supported by the document. "
                        "Use only the supplied entity indices and relation IDs. Copy an "
                        "exact supporting quote. Return an empty list when no listed "
                        "relation is supported; do not infer from entity names alone."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text_format=self.output_model,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("The model returned no parsed relation output.")
        relations = [IndexedRelation.model_validate(item.model_dump()) for item in parsed.relations]
        entity_count = len(entities)
        for relation in relations:
            if relation.head_index >= entity_count or relation.tail_index >= entity_count:
                raise ValueError(f"Entity index out of range in sample {sample.sample_id}.")
            if relation.evidence_quote not in sample.text:
                raise ValueError(f"Evidence quote not found in sample {sample.sample_id}.")
        return IndexedPrediction(sample_id=sample.sample_id, relations=relations)


def gold_prediction(sample: EvaluationSample) -> IndexedPrediction:
    relations = [IndexedRelation(**item, evidence_quote="") for item in sample.benchmark_metadata.get("indexed_relations", [])]
    return IndexedPrediction(sample_id=sample.sample_id, relations=relations)


def score_indexed(gold: list[IndexedPrediction], predicted: list[IndexedPrediction]) -> dict:
    predicted_by_id = {item.sample_id: {relation.key() for relation in item.relations} for item in predicted}
    tp = fp = fn = 0
    for item in gold:
        expected = {relation.key() for relation in item.relations}
        actual = predicted_by_id.get(item.sample_id, set())
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained Re-DocRED relation prediction.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--relation-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set before running predictions.")
    samples = load_jsonl(args.dataset, EvaluationSample)
    relation_map = json.loads(Path(args.relation_map).read_text(encoding="utf-8"))
    predictor = OpenAIConstrainedRelationPredictor(relation_map, args.model)
    predictions = [predictor.predict(sample) for sample in samples]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(item.model_dump_json() + "\n" for item in predictions), encoding="utf-8")
    metrics = score_indexed([gold_prediction(sample) for sample in samples], predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
