from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, create_model

from .evaluation import EvaluationSample, load_jsonl
from .models import RetrievedChunk
from .redocred_predictor import IndexedPrediction, IndexedRelation, gold_prediction, score_indexed


class RelationVerdict(BaseModel):
    candidate_index: int = Field(ge=0)
    verdict: Literal["supported", "unsupported", "needs_correction"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    corrected_relation_id: str | None = None


class RelationDecisionTrace(BaseModel):
    round_index: int
    phase: Literal["verification", "correction_suggestion"] = "verification"
    head_index: int
    tail_index: int
    relation_id: str
    evidence_quote: str
    verdict: Literal["supported", "unsupported", "needs_correction"]
    confidence: float
    reason: str
    corrected_relation_id: str | None = None
    correction_blocked_reason: str | None = None
    retrieved_chunks: list[dict] = Field(default_factory=list)


class RelationVerifier(Protocol):
    def verify(self, sample: EvaluationSample, candidates: list[IndexedRelation]) -> list[RelationVerdict]: ...


class RelationEvidenceRetriever(Protocol):
    def retrieve(
        self, sample: EvaluationSample, candidates: list[IndexedRelation]
    ) -> dict[int, list[RetrievedChunk]]: ...


def build_verification_model(allowed_relation_ids: list[str]):
    if not allowed_relation_ids:
        raise ValueError("At least one relation ID is required.")
    relation_literal = Literal.__getitem__(tuple(sorted(set(allowed_relation_ids))))
    verdict_model = create_model(
        "ConstrainedRelationVerdict",
        candidate_index=(int, Field(ge=0)),
        verdict=(Literal["supported", "unsupported", "needs_correction"], ...),
        confidence=(float, Field(ge=0.0, le=1.0)),
        reason=(str, ...),
        corrected_relation_id=(relation_literal | None, ...),
    )
    return create_model("ConstrainedVerificationBatch", verdicts=(list[verdict_model], Field(default_factory=list)))


class OpenAIRelationVerifier:
    def __init__(self, relation_map: dict[str, str], model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install the OpenAI extra: pip install -e ".[openai]"') from exc
        self.client = OpenAI()
        self.model = model or os.getenv("LLM_MODEL", "gpt-5-mini")
        self.relation_map = relation_map
        self.output_model = build_verification_model(list(relation_map))

    def verify(self, sample: EvaluationSample, candidates: list[IndexedRelation]) -> list[RelationVerdict]:
        return self._verify(sample, candidates, None)

    def verify_with_evidence(
        self,
        sample: EvaluationSample,
        candidates: list[IndexedRelation],
        evidence_by_candidate: dict[int, list[RetrievedChunk]],
    ) -> list[RelationVerdict]:
        return self._verify(sample, candidates, evidence_by_candidate)

    def _verify(
        self,
        sample: EvaluationSample,
        candidates: list[IndexedRelation],
        evidence_by_candidate: dict[int, list[RetrievedChunk]] | None,
        *,
        correction_mode: bool = False,
    ) -> list[RelationVerdict]:
        entities = sample.benchmark_metadata.get("entities", [])
        catalog = "\n".join(f"{key}: {value}" for key, value in self.relation_map.items())
        strict_relation_rules = []
        if "P1344" in self.relation_map:
            strict_relation_rules.append(
                "P1344 participant in applies to participation in an event; membership in an "
                "organization, holding office, or having a seat in a legislature is not P1344."
            )
        if "P39" in self.relation_map:
            strict_relation_rules.append(
                "P39 position held requires the tail entity itself to be an office or role; a "
                "place, jurisdiction, or employing organization is not a position."
            )
        if "P108" in self.relation_map:
            strict_relation_rules.append(
                "P108 employer requires textual support that the tail employed the person; merely "
                "serving in a place or jurisdiction does not establish an employer relation."
            )
        strict_relation_policy = " ".join(strict_relation_rules)
        candidate_text = "\n".join(
            f"{i}: head={candidate.head_index} ({entities[candidate.head_index]}), "
            f"tail={candidate.tail_index} ({entities[candidate.tail_index]}), "
            f"relation={candidate.relation_id} ({self.relation_map[candidate.relation_id]}), "
            f"evidence={candidate.evidence_quote!r}"
            for i, candidate in enumerate(candidates)
        )
        if evidence_by_candidate is None:
            evidence_context = sample.text
            evidence_instruction = "Verify against the supplied document and quoted evidence."
        else:
            evidence_context = "\n\n".join(
                f"CANDIDATE {index} RETRIEVED EVIDENCE:\n"
                + "\n".join(
                    f"- [{chunk.chunk_id}; score={chunk.score:.4f}] {chunk.text}"
                    for chunk in evidence_by_candidate.get(index, [])
                )
                for index in range(len(candidates))
            )
            evidence_instruction = (
                "Verify each candidate only against its retrieved evidence section. "
                "If that section lacks sufficient evidence, mark it unsupported."
            )
        if correction_mode:
            system_instruction = (
                "Each candidate has already failed verification. Determine whether the exact "
                "same head-tail pair is supported by the evidence under a different relation "
                "from the catalog. If so, return needs_correction and that relation ID. "
                "Otherwise return unsupported. Never return supported, never change the entity "
                "pair, and do not use world knowledge or entity-name plausibility. Return one "
                "verdict for every candidate index."
            )
        else:
            system_instruction = (
                f"{evidence_instruction} Apply a strict textual-entailment standard. "
                "Entity co-occurrence, related background knowledge, or a plausible "
                "association is not support. Check relation direction and distinguish "
                "nearby labels such as location versus country, participant versus "
                "part-of, and creator versus performer. If the evidence supports a "
                "different catalog relation, return needs_correction and its relation ID. "
                f"Otherwise mark unsupported. {strict_relation_policy} Return one verdict for "
                "every candidate index."
            )
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {"role": "user", "content": f"EVIDENCE CONTEXT:\n{evidence_context}\n\nRELATION CATALOG:\n{catalog}\n\nCANDIDATES:\n{candidate_text}"},
            ],
            text_format=self.output_model,
        )
        if response.output_parsed is None:
            raise RuntimeError("The verifier returned no parsed output.")
        verdicts = [RelationVerdict.model_validate(item.model_dump()) for item in response.output_parsed.verdicts]
        if {item.candidate_index for item in verdicts} != set(range(len(candidates))):
            raise ValueError("Verifier must return exactly one verdict for every candidate.")
        verdicts = sorted(verdicts, key=lambda item: item.candidate_index)
        if correction_mode and any(item.verdict == "supported" for item in verdicts):
            raise ValueError("Correction suggestion must not directly accept a rejected candidate.")
        return verdicts

    def suggest_corrections(
        self, sample: EvaluationSample, candidates: list[IndexedRelation]
    ) -> list[RelationVerdict]:
        return self._verify(sample, candidates, None, correction_mode=True)

    def suggest_corrections_with_evidence(
        self,
        sample: EvaluationSample,
        candidates: list[IndexedRelation],
        evidence_by_candidate: dict[int, list[RetrievedChunk]],
    ) -> list[RelationVerdict]:
        return self._verify(
            sample, candidates, evidence_by_candidate, correction_mode=True
        )


def _grounded_candidates(sample: EvaluationSample, candidates: list[IndexedRelation]) -> list[IndexedRelation]:
    entity_count = len(sample.benchmark_metadata.get("entities", []))
    return [
        candidate
        for candidate in candidates
        if candidate.head_index < entity_count
        and candidate.tail_index < entity_count
        and candidate.evidence_quote
        and candidate.evidence_quote in sample.text
    ]


def verify_and_correct(
    sample: EvaluationSample,
    prediction: IndexedPrediction,
    verifier: RelationVerifier,
    max_correction_rounds: int = 2,
    evidence_retriever: RelationEvidenceRetriever | None = None,
) -> IndexedPrediction:
    pending = _grounded_candidates(sample, prediction.relations)
    seen_relations_by_pair: dict[tuple[int, int], set[str]] = {}
    for candidate in pending:
        seen_relations_by_pair.setdefault(
            (candidate.head_index, candidate.tail_index), set()
        ).add(candidate.relation_id)
    accepted: list[IndexedRelation] = []
    trace: list[RelationDecisionTrace] = []
    for round_index in range(max_correction_rounds + 1):
        if not pending:
            break
        if evidence_retriever is not None:
            evidence = evidence_retriever.retrieve(sample, pending)
            verify_with_evidence = getattr(verifier, "verify_with_evidence", None)
            if verify_with_evidence is None:
                raise TypeError("RAG verification requires a verifier with verify_with_evidence().")
            verdicts = verify_with_evidence(sample, pending, evidence)
        else:
            evidence = {}
            verdicts = verifier.verify(sample, pending)
        corrected: list[IndexedRelation] = []
        unsupported: list[tuple[int, IndexedRelation]] = []
        for candidate, verdict in zip(pending, verdicts):
            correction_blocked_reason = None
            corrected_candidate = None
            if (
                verdict.verdict == "needs_correction"
                and verdict.corrected_relation_id
                and round_index < max_correction_rounds
            ):
                pair = (candidate.head_index, candidate.tail_index)
                if verdict.corrected_relation_id in seen_relations_by_pair.setdefault(pair, set()):
                    correction_blocked_reason = "correction_cycle_or_duplicate"
                else:
                    seen_relations_by_pair[pair].add(verdict.corrected_relation_id)
                    corrected_candidate = candidate.model_copy(
                        update={"relation_id": verdict.corrected_relation_id}
                    )
            trace.append(
                RelationDecisionTrace(
                    round_index=round_index,
                    phase="verification",
                    head_index=candidate.head_index,
                    tail_index=candidate.tail_index,
                    relation_id=candidate.relation_id,
                    evidence_quote=candidate.evidence_quote,
                    verdict=verdict.verdict,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                    corrected_relation_id=verdict.corrected_relation_id,
                    correction_blocked_reason=correction_blocked_reason,
                    retrieved_chunks=[chunk.model_dump() for chunk in evidence.get(verdict.candidate_index, [])],
                )
            )
            if verdict.verdict == "supported":
                accepted.append(candidate)
            elif corrected_candidate is not None:
                corrected.append(corrected_candidate)
            elif verdict.verdict == "unsupported" and round_index < max_correction_rounds:
                unsupported.append((verdict.candidate_index, candidate))

        if unsupported:
            correction_candidates = [candidate for _, candidate in unsupported]
            if evidence_retriever is not None:
                suggest = getattr(verifier, "suggest_corrections_with_evidence", None)
                correction_evidence = {
                    new_index: evidence.get(old_index, [])
                    for new_index, (old_index, _) in enumerate(unsupported)
                }
                suggestions = (
                    suggest(sample, correction_candidates, correction_evidence)
                    if suggest is not None else []
                )
            else:
                suggest = getattr(verifier, "suggest_corrections", None)
                correction_evidence = {}
                suggestions = suggest(sample, correction_candidates) if suggest is not None else []
            if suggestions and len(suggestions) != len(correction_candidates):
                raise ValueError("Correction suggester must return one verdict per rejected candidate.")
            for candidate, suggestion in zip(correction_candidates, suggestions):
                correction_blocked_reason = None
                corrected_candidate = None
                if suggestion.verdict == "needs_correction" and suggestion.corrected_relation_id:
                    pair = (candidate.head_index, candidate.tail_index)
                    if suggestion.corrected_relation_id in seen_relations_by_pair.setdefault(pair, set()):
                        correction_blocked_reason = "correction_cycle_or_duplicate"
                    else:
                        seen_relations_by_pair[pair].add(suggestion.corrected_relation_id)
                        corrected_candidate = candidate.model_copy(
                            update={"relation_id": suggestion.corrected_relation_id}
                        )
                trace.append(
                    RelationDecisionTrace(
                        round_index=round_index,
                        phase="correction_suggestion",
                        head_index=candidate.head_index,
                        tail_index=candidate.tail_index,
                        relation_id=candidate.relation_id,
                        evidence_quote=candidate.evidence_quote,
                        verdict=suggestion.verdict,
                        confidence=suggestion.confidence,
                        reason=suggestion.reason,
                        corrected_relation_id=suggestion.corrected_relation_id,
                        correction_blocked_reason=correction_blocked_reason,
                        retrieved_chunks=[
                            chunk.model_dump()
                            for chunk in correction_evidence.get(suggestion.candidate_index, [])
                        ],
                    )
                )
                if corrected_candidate is not None:
                    corrected.append(corrected_candidate)
        pending = list({relation.key(): relation for relation in corrected}.values())
    unique = {relation.key(): relation for relation in accepted}
    return IndexedPrediction(
        sample_id=prediction.sample_id,
        relations=list(unique.values()),
        verification_trace=[item.model_dump() for item in trace],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and correct constrained Re-DocRED predictions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--relation-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--rag", action="store_true", help="Use FAISS retrieval before every verification round.")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--chunk-overlap", type=int, default=75)
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set before running verification.")
    samples = load_jsonl(args.dataset, EvaluationSample)
    predictions = load_jsonl(args.predictions, IndexedPrediction)
    prediction_by_id = {item.sample_id: item for item in predictions}
    relation_map = json.loads(Path(args.relation_map).read_text(encoding="utf-8"))
    verifier = OpenAIRelationVerifier(relation_map, args.model)
    retriever = None
    if args.rag:
        from .rag import OpenAIEmbeddingProvider
        from .redocred_rag import FAISSRelationEvidenceRetriever

        retriever = FAISSRelationEvidenceRetriever(
            OpenAIEmbeddingProvider(args.embedding_model),
            relation_map,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
    verified = [
        verify_and_correct(
            sample,
            prediction_by_id.get(sample.sample_id, IndexedPrediction(sample_id=sample.sample_id)),
            verifier,
            evidence_retriever=retriever,
        )
        for sample in samples
    ]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(item.model_dump_json() + "\n" for item in verified), encoding="utf-8")
    report = {
        "baseline": score_indexed([gold_prediction(sample) for sample in samples], predictions),
        "verified": score_indexed([gold_prediction(sample) for sample in samples], verified),
    }
    if retriever is not None:
        from .redocred_rag import retrieval_metrics

        report["retrieval"] = retrieval_metrics(samples, retriever)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
