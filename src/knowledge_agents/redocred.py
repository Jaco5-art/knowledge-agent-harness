from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .evaluation import EvaluationSample, LabeledFact


def _entity_name(vertex_set: list[list[dict]], index: int) -> str:
    mentions = vertex_set[index]
    if not mentions:
        return f"entity_{index}"
    return mentions[0]["name"]


def convert_document(document: dict, relation_map: dict[str, str] | None = None, index: int = 0) -> EvaluationSample:
    relation_map = relation_map or {}
    sentences = document["sents"]
    text = " ".join(" ".join(tokens) for tokens in sentences)
    facts = []
    for label in document.get("labels", []):
        relation_id = label["r"]
        facts.append(
            LabeledFact(
                subject=_entity_name(document["vertexSet"], label["h"]),
                predicate=relation_map.get(relation_id, relation_id),
                object=_entity_name(document["vertexSet"], label["t"]),
            )
        )
    title = document.get("title", f"document-{index}")
    entities = [_entity_name(document["vertexSet"], entity_index) for entity_index in range(len(document["vertexSet"]))]
    indexed_relations = []
    for label in document.get("labels", []):
        evidence_sentence_ids = label.get("evidence", [])
        indexed_relations.append(
            {
                "head_index": label["h"],
                "tail_index": label["t"],
                "relation_id": label["r"],
                # Retained for retrieval evaluation. The indexed scorer ignores
                # these additional benchmark-only fields.
                "evidence_sentence_ids": evidence_sentence_ids,
                "evidence_quotes": [
                    " ".join(sentences[sentence_id])
                    for sentence_id in evidence_sentence_ids
                    if 0 <= sentence_id < len(sentences)
                ],
            }
        )
    return EvaluationSample(
        sample_id=f"redocred-{index:04d}",
        text=text,
        gold_facts=facts,
        source_type="Re-DocRED",
        difficulty="document_level",
        phenomena=["document_relation_extraction", "coreference", "cross_sentence"],
        review_status="public_benchmark",
        reviewed_by="Re-DocRED authors",
        reviewer_notes=f"Original document title: {title}",
        benchmark_metadata={
            "title": title,
            "entities": entities,
            "indexed_relations": indexed_relations,
        },
    )


def load_and_sample(
    dataset_path: str | Path,
    relation_map_path: str | Path | None = None,
    sample_size: int = 100,
    seed: int = 42,
) -> list[EvaluationSample]:
    documents = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    relation_map = {}
    if relation_map_path:
        relation_map = json.loads(Path(relation_map_path).read_text(encoding="utf-8"))
    if sample_size > len(documents):
        raise ValueError(f"Requested {sample_size} documents from a split containing {len(documents)}.")
    selected_indices = sorted(random.Random(seed).sample(range(len(documents)), sample_size))
    return [convert_document(documents[i], relation_map, i) for i in selected_indices]


def write_jsonl(samples: list[EvaluationSample], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8")
    return destination


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    samples: list[EvaluationSample],
    dataset_path: str | Path,
    source_path: str | Path,
    destination: str | Path,
    *,
    relation_map_path: str | Path | None,
    sample_size: int,
    seed: int,
    source_revision: str | None = None,
) -> Path:
    manifest = {
        "benchmark": "Re-DocRED",
        "source_repository": "https://github.com/tonytan48/Re-DocRED",
        "source_revision": source_revision,
        "source_split_file": Path(source_path).name,
        "source_sha256": _sha256(source_path),
        "sample_size": sample_size,
        "sample_seed": seed,
        "sample_ids": [sample.sample_id for sample in samples],
        "derived_dataset_sha256": _sha256(dataset_path),
        "relation_map_file": Path(relation_map_path).name if relation_map_path else None,
        "relation_map_sha256": _sha256(relation_map_path) if relation_map_path else None,
        "gold_relation_count": sum(
            len(sample.benchmark_metadata.get("indexed_relations", [])) for sample in samples
        ),
        "gold_relations_with_annotated_evidence": sum(
            1
            for sample in samples
            for relation in sample.benchmark_metadata.get("indexed_relations", [])
            if relation.get("evidence_quotes")
        ),
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a deterministic Re-DocRED subset to the project evaluation format.")
    parser.add_argument("--input", required=True, help="Path to dev_revised.json or test_revised.json")
    parser.add_argument("--relation-map", default=None, help="Optional JSON mapping from relation IDs to readable labels")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/redocred_eval_100.jsonl")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--source-revision", default=None)
    args = parser.parse_args()
    samples = load_and_sample(args.input, args.relation_map, args.sample_size, args.seed)
    path = write_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} Re-DocRED documents to {path}")
    if args.manifest:
        manifest_path = write_manifest(
            samples,
            path,
            args.input,
            args.manifest,
            relation_map_path=args.relation_map,
            sample_size=args.sample_size,
            seed=args.seed,
            source_revision=args.source_revision,
        )
        print(f"Wrote reproducibility manifest to {manifest_path}")


if __name__ == "__main__":
    main()
