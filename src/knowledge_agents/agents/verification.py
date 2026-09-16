from __future__ import annotations

import re

from ..models import Evidence, ExtractedFact, VerificationResult
from ..state import WorkflowState


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def verify_fact(fact: ExtractedFact, source_text: str) -> VerificationResult:
    """Evidence-grounded verifier used by tests and as a guard before LLM review.

    It does not claim semantic proof. It rejects missing evidence and flags facts
    whose core arguments are absent from their quoted source span.
    """
    if not fact.evidence:
        return VerificationResult(
            fact_id=fact.fact_id,
            verdict="unsupported",
            confidence=0.95,
            reasons=["No source evidence was attached to the extracted fact."],
        )

    valid_spans = [e for e in fact.evidence if e.quote and e.quote in source_text]
    if not valid_spans:
        return VerificationResult(
            fact_id=fact.fact_id,
            verdict="unsupported",
            confidence=0.98,
            reasons=["The quoted evidence does not occur in the source text."],
        )

    evidence_tokens = _tokens(" ".join(e.quote for e in valid_spans))
    argument_tokens = _tokens(fact.subject) | _tokens(fact.object)
    missing = sorted(argument_tokens - evidence_tokens)
    if missing:
        return VerificationResult(
            fact_id=fact.fact_id,
            verdict="needs_correction",
            confidence=0.8,
            reasons=[f"Core argument tokens are absent from evidence: {', '.join(missing)}"],
            matched_evidence=valid_spans,
        )

    predicate_tokens = _tokens(fact.predicate)
    predicate_overlap = predicate_tokens & evidence_tokens
    if predicate_tokens and not predicate_overlap:
        return VerificationResult(
            fact_id=fact.fact_id,
            verdict="needs_correction",
            confidence=0.75,
            reasons=["The claimed predicate is not lexically grounded in the evidence."],
            matched_evidence=valid_spans,
        )

    return VerificationResult(
        fact_id=fact.fact_id,
        verdict="supported",
        confidence=min(0.99, (fact.extraction_confidence + 1.0) / 2),
        reasons=["Subject, predicate and object are grounded in a source span."],
        matched_evidence=valid_spans,
    )


def verify_knowledge(state: WorkflowState) -> dict:
    facts = state.get("fused_facts", [])
    retrieved = state.get("retrieved_chunks", {})
    results = []
    for fact in facts:
        contexts = retrieved.get(fact.fact_id)
        verification_source = "\n".join(contexts) if contexts else state["source_text"]
        results.append(verify_fact(fact, verification_source))
    by_id = {result.fact_id: result for result in results}
    verified = [fact for fact in facts if by_id[fact.fact_id].verdict == "supported"]
    rejected = [fact for fact in facts if by_id[fact.fact_id].verdict != "supported"]
    return {
        "verification_results": results,
        "verified_facts": verified,
        "rejected_facts": rejected,
    }
