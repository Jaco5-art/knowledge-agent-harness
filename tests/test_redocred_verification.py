from knowledge_agents.evaluation import EvaluationSample
from knowledge_agents.redocred_predictor import IndexedPrediction, IndexedRelation
from knowledge_agents.redocred_verification import RelationVerdict, build_verification_model, verify_and_correct


class CorrectThenSupportVerifier:
    def verify(self, sample, candidates):
        output = []
        for index, candidate in enumerate(candidates):
            if candidate.relation_id == "P112":
                output.append(RelationVerdict(candidate_index=index, verdict="needs_correction", confidence=0.9, reason="The evidence states location, not founder.", corrected_relation_id="P159"))
            else:
                output.append(RelationVerdict(candidate_index=index, verdict="supported", confidence=0.95, reason="Evidence supports headquarters location."))
        return output


class RejectSuggestThenSupportVerifier:
    def verify(self, sample, candidates):
        return [
            RelationVerdict(
                candidate_index=index,
                verdict="supported" if candidate.relation_id == "P159" else "unsupported",
                confidence=0.95,
                reason="The corrected location relation is supported."
                if candidate.relation_id == "P159" else "The founder relation is unsupported.",
            )
            for index, candidate in enumerate(candidates)
        ]

    def suggest_corrections(self, sample, candidates):
        return [
            RelationVerdict(
                candidate_index=index,
                verdict="needs_correction",
                confidence=0.9,
                reason="The same pair expresses a headquarters location.",
                corrected_relation_id="P159",
            )
            for index, _ in enumerate(candidates)
        ]


class OscillatingVerifier:
    def verify(self, sample, candidates):
        return [
            RelationVerdict(
                candidate_index=index,
                verdict="needs_correction",
                confidence=0.9,
                reason="oscillating suggestion",
                corrected_relation_id="P159" if candidate.relation_id == "P112" else "P112",
            )
            for index, candidate in enumerate(candidates)
        ]

def sample():
    return EvaluationSample(
        sample_id="s1",
        text="Apex is based in Paris.",
        benchmark_metadata={"entities": ["Apex", "Paris"]},
    )


def test_verification_schema_constrains_correction_label():
    model = build_verification_model(["P112", "P159"])
    value = model.model_validate({"verdicts": [{"candidate_index": 0, "verdict": "needs_correction", "confidence": 0.9, "reason": "wrong relation", "corrected_relation_id": "P159"}]})
    assert value.verdicts[0].corrected_relation_id == "P159"


def test_correction_is_reverified_before_acceptance():
    prediction = IndexedPrediction(sample_id="s1", relations=[IndexedRelation(head_index=0, tail_index=1, relation_id="P112", evidence_quote="Apex is based in Paris")])
    verified = verify_and_correct(sample(), prediction, CorrectThenSupportVerifier())
    assert len(verified.relations) == 1
    assert verified.relations[0].relation_id == "P159"


def test_fabricated_evidence_is_rejected_before_llm_verification():
    prediction = IndexedPrediction(sample_id="s1", relations=[IndexedRelation(head_index=0, tail_index=1, relation_id="P159", evidence_quote="Apex acquired Paris")])
    verified = verify_and_correct(sample(), prediction, CorrectThenSupportVerifier())
    assert verified.relations == []


def test_rejected_candidate_can_be_corrected_but_must_be_reverified():
    prediction = IndexedPrediction(
        sample_id="s1",
        relations=[
            IndexedRelation(
                head_index=0,
                tail_index=1,
                relation_id="P112",
                evidence_quote="Apex is based in Paris.",
            )
        ],
    )
    verified = verify_and_correct(sample(), prediction, RejectSuggestThenSupportVerifier())
    assert [relation.relation_id for relation in verified.relations] == ["P159"]
    assert [item["phase"] for item in verified.verification_trace] == [
        "verification",
        "correction_suggestion",
        "verification",
    ]


def test_correction_cycle_is_blocked_instead_of_reaccepting_original_relation():
    prediction = IndexedPrediction(
        sample_id="s1",
        relations=[
            IndexedRelation(
                head_index=0,
                tail_index=1,
                relation_id="P112",
                evidence_quote="Apex is based in Paris.",
            )
        ],
    )
    verified = verify_and_correct(sample(), prediction, OscillatingVerifier())
    assert verified.relations == []
    assert len(verified.verification_trace) == 2
    assert verified.verification_trace[-1]["correction_blocked_reason"] == "correction_cycle_or_duplicate"
