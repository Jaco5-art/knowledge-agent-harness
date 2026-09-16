from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluation import EvaluationSample, load_jsonl
from .redocred_predictor import IndexedRelation
from .redocred_rag import FAISSRelationEvidenceRetriever
from .redocred_verification import OpenAIRelationVerifier, RelationVerdict, RelationVerifier


def build_cases(
    sample: EvaluationSample,
    relation_ids: list[str],
) -> tuple[list[IndexedRelation], list[dict]]:
    metadata = sample.benchmark_metadata.get("indexed_relations", [])
    gold_keys = {
        (item["head_index"], item["tail_index"], item["relation_id"])
        for item in metadata
    }
    candidates: list[IndexedRelation] = []
    cases: list[dict] = []
    for gold_index, item in enumerate(metadata):
        quotes = [quote for quote in item.get("evidence_quotes", []) if quote]
        if not quotes:
            continue
        original = IndexedRelation(
            head_index=item["head_index"],
            tail_index=item["tail_index"],
            relation_id=item["relation_id"],
            evidence_quote=quotes[0],
        )
        candidates.append(original)
        cases.append({"case_type": "clean", "original_relation_id": original.relation_id})

        reversed_key = (original.tail_index, original.head_index, original.relation_id)
        if gold_index % 2 and original.head_index != original.tail_index and reversed_key not in gold_keys:
            corrupted = original.model_copy(
                update={"head_index": original.tail_index, "tail_index": original.head_index}
            )
            case_type = "head_tail_reversal"
        else:
            replacement = next(
                relation_id
                for relation_id in relation_ids
                if (original.head_index, original.tail_index, relation_id) not in gold_keys
            )
            corrupted = original.model_copy(update={"relation_id": replacement})
            case_type = "relation_replacement"
        candidates.append(corrupted)
        cases.append(
            {
                "case_type": case_type,
                "original_relation_id": original.relation_id,
            }
        )
    return candidates, cases


def _score(cases: list[dict], verdicts: list[RelationVerdict]) -> dict:
    clean = [(case, verdict) for case, verdict in zip(cases, verdicts) if case["case_type"] == "clean"]
    corrupted = [(case, verdict) for case, verdict in zip(cases, verdicts) if case["case_type"] != "clean"]
    retained = sum(verdict.verdict == "supported" for _, verdict in clean)
    detected = sum(verdict.verdict != "supported" for _, verdict in corrupted)
    replace_cases = [(case, verdict) for case, verdict in corrupted if case["case_type"] == "relation_replacement"]
    corrected = sum(
        verdict.verdict == "needs_correction"
        and verdict.corrected_relation_id == case["original_relation_id"]
        for case, verdict in replace_cases
    )
    return {
        "clean_cases": len(clean),
        "clean_retained": retained,
        "clean_retention_rate": round(retained / len(clean), 4) if clean else 0.0,
        "corrupted_cases": len(corrupted),
        "corrupted_detected": detected,
        "error_detection_rate": round(detected / len(corrupted), 4) if corrupted else 0.0,
        "relation_replacement_cases": len(replace_cases),
        "correct_relation_suggested": corrected,
        "correction_suggestion_accuracy": round(corrected / len(replace_cases), 4) if replace_cases else 0.0,
    }


def run_challenge(
    samples: list[EvaluationSample],
    relation_map: dict[str, str],
    verifier: RelationVerifier,
    retriever: FAISSRelationEvidenceRetriever | None = None,
) -> tuple[dict, list[dict]]:
    all_cases: list[dict] = []
    all_verdicts: list[RelationVerdict] = []
    trace_rows: list[dict] = []
    relation_ids = sorted(relation_map)
    for sample in samples:
        candidates, cases = build_cases(sample, relation_ids)
        if not candidates:
            continue
        # Evaluate clean and corrupted candidates in separate calls so the
        # verifier cannot identify an error merely by comparing duplicate pairs.
        for clean_group in (True, False):
            group = [
                (candidate, case)
                for candidate, case in zip(candidates, cases)
                if (case["case_type"] == "clean") == clean_group
            ]
            group_candidates = [candidate for candidate, _ in group]
            group_cases = [case for _, case in group]
            if retriever is None:
                verdicts = verifier.verify(sample, group_candidates)
                retrieved = {}
            else:
                retrieved = retriever.retrieve(sample, group_candidates)
                verify_with_evidence = getattr(verifier, "verify_with_evidence", None)
                if verify_with_evidence is None:
                    raise TypeError("RAG challenge requires verify_with_evidence().")
                verdicts = verify_with_evidence(sample, group_candidates, retrieved)
            all_cases.extend(group_cases)
            all_verdicts.extend(verdicts)
            for index, (candidate, case, verdict) in enumerate(zip(group_candidates, group_cases, verdicts)):
                trace_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        **case,
                        "head_index": candidate.head_index,
                        "tail_index": candidate.tail_index,
                        "candidate_relation_id": candidate.relation_id,
                        "evidence_quote": candidate.evidence_quote,
                        "verdict": verdict.verdict,
                        "confidence": verdict.confidence,
                        "reason": verdict.reason,
                        "corrected_relation_id": verdict.corrected_relation_id,
                        "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved.get(index, [])],
                    }
                )
    return _score(all_cases, all_verdicts), trace_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verification on controlled Re-DocRED relation errors.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--relation-map", required=True)
    parser.add_argument("--output", default="outputs/redocred/verification-challenge")
    parser.add_argument("--model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set before running the verification challenge.")
    from .rag import OpenAIEmbeddingProvider

    samples = load_jsonl(args.dataset, EvaluationSample)[: args.limit]
    relation_map = json.loads(Path(args.relation_map).read_text(encoding="utf-8"))
    verifier = OpenAIRelationVerifier(relation_map, args.model)
    no_rag, no_rag_trace = run_challenge(samples, relation_map, verifier)
    embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    retriever = FAISSRelationEvidenceRetriever(
        OpenAIEmbeddingProvider(embedding_model), relation_map, top_k=args.top_k
    )
    rag, rag_trace = run_challenge(samples, relation_map, verifier, retriever)
    report = {
        "sample_count": len(samples),
        "model": args.model or os.getenv("LLM_MODEL", "gpt-5-mini"),
        "verification_only": no_rag,
        "rag_verification": rag,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [dict(condition="verification_only", **row) for row in no_rag_trace]
    rows += [dict(condition="rag_verification", **row) for row in rag_trace]
    (output / "decision_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
