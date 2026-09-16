from datetime import datetime, timezone

import pytest

from knowledge_agents.evaluation import EvaluationSample
from knowledge_agents.models import RetrievedChunk
from knowledge_agents.rag import HashEmbeddingProvider
from knowledge_agents.redocred_predictor import IndexedPrediction, IndexedRelation
from knowledge_agents.redocred_rag import FAISSRelationEvidenceRetriever
from knowledge_agents.redocred_verification import RelationVerdict
from knowledge_agents.formal_experiment import run_formal_experiment


class FixedPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, sample):
        self.calls += 1
        return IndexedPrediction(
            sample_id=sample.sample_id,
            relations=[IndexedRelation(head_index=0, tail_index=1, relation_id="P159", evidence_quote=sample.text)],
        )


class AlwaysSupportsVerifier:
    def verify(self, sample, candidates):
        return self._verdicts(candidates)

    def verify_with_evidence(self, sample, candidates, evidence_by_candidate):
        assert evidence_by_candidate[0]
        return self._verdicts(candidates)

    @staticmethod
    def _verdicts(candidates):
        return [RelationVerdict(candidate_index=i, verdict="supported", confidence=1.0, reason="supported") for i in range(len(candidates))]


class WrongRelationPredictor:
    def predict(self, sample):
        return IndexedPrediction(
            sample_id=sample.sample_id,
            relations=[IndexedRelation(head_index=0, tail_index=1, relation_id="P112", evidence_quote=sample.text)],
        )


class MustNotRunPredictor:
    def predict(self, sample):
        raise AssertionError("External baseline should prevent predictor calls.")


class FakeRetriever:
    top_k = 1
    chunk_size = 350
    overlap = 75
    relation_map = {"P112": "founded by", "P159": "headquarters location"}

    def retrieve(self, sample, candidates):
        return {
            index: [RetrievedChunk(chunk_id="c1", document_id=sample.sample_id, text=sample.text, start=0, end=len(sample.text), score=1.0)]
            for index, _ in enumerate(candidates)
        }


class ClosedLoopVerifier:
    def verify(self, sample, candidates):
        return self._verify(candidates)

    def verify_with_evidence(self, sample, candidates, evidence_by_candidate):
        return self._verify(candidates)

    @staticmethod
    def _verify(candidates):
        return [
            RelationVerdict(
                candidate_index=index,
                verdict="supported" if candidate.relation_id == "P159" else "unsupported",
                confidence=1.0,
                reason="checked",
            )
            for index, candidate in enumerate(candidates)
        ]

    def suggest_corrections(self, sample, candidates):
        return self._suggest(candidates)

    def suggest_corrections_with_evidence(self, sample, candidates, evidence_by_candidate):
        return self._suggest(candidates)

    @staticmethod
    def _suggest(candidates):
        return [
            RelationVerdict(
                candidate_index=index,
                verdict="needs_correction",
                confidence=1.0,
                reason="The pair expresses a headquarters location.",
                corrected_relation_id="P159",
            )
            for index, _ in enumerate(candidates)
        ]


def test_formal_experiment_writes_three_conditions_and_report(tmp_path):
    sample = EvaluationSample(
        sample_id="redocred-0001",
        text="Apex is based in Paris.",
        source_type="Re-DocRED",
        benchmark_metadata={
            "entities": ["Apex", "Paris"],
            "indexed_relations": [{
                "head_index": 0,
                "tail_index": 1,
                "relation_id": "P159",
                "evidence_quotes": ["Apex is based in Paris."],
            }],
        },
    )
    retriever = FAISSRelationEvidenceRetriever(
        HashEmbeddingProvider(), {"P159": "headquarters location"}, top_k=1
    )
    report = run_formal_experiment(
        [sample], FixedPredictor(), AlwaysSupportsVerifier(), retriever, tmp_path,
        llm_model="test-model", embedding_model="hash-test",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert set(report["conditions"]) == {"baseline", "verification_only", "rag_verification"}
    assert report["retrieval"]["evidence_recall_at_k"] == 1.0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "failure_cases.csv").exists()
    assert report["failure_analysis"]["category_counts"]["persistent_true_positive"] == 1
    assert report["workflow_actions"]["verification_only"]["supported_decisions"] == 1
    assert report["workflow_actions"]["rag_verification"]["supported_decisions"] == 1
    assert "inspect before using results in a resume" in (tmp_path / "report.md").read_text()


def test_formal_experiment_resumes_without_repeating_baseline_calls(tmp_path):
    sample = EvaluationSample(
        sample_id="redocred-0001",
        text="Apex is based in Paris.",
        benchmark_metadata={
            "entities": ["Apex", "Paris"],
            "indexed_relations": [{
                "head_index": 0, "tail_index": 1, "relation_id": "P159",
                "evidence_quotes": ["Apex is based in Paris."],
            }],
        },
    )
    retriever = FAISSRelationEvidenceRetriever(
        HashEmbeddingProvider(), {"P159": "headquarters location"}, top_k=1
    )
    first_predictor = FixedPredictor()
    run_formal_experiment(
        [sample], first_predictor, AlwaysSupportsVerifier(), retriever, tmp_path,
        llm_model="test", embedding_model="hash",
    )
    assert first_predictor.calls == 1
    resumed_predictor = FixedPredictor()
    report = run_formal_experiment(
        [sample], resumed_predictor, AlwaysSupportsVerifier(), retriever, tmp_path,
        llm_model="test", embedding_model="hash", resume=True,
        dataset_sample_count=100,
    )
    assert resumed_predictor.calls == 0
    assert report["run"]["run_scope"] == "smoke_test"


def test_resume_rejects_changed_settings(tmp_path):
    sample = EvaluationSample(
        sample_id="s1", text="Apex is based in Paris.",
        benchmark_metadata={
            "entities": ["Apex", "Paris"],
            "indexed_relations": [{"head_index": 0, "tail_index": 1, "relation_id": "P159"}],
        },
    )
    retriever = FAISSRelationEvidenceRetriever(
        HashEmbeddingProvider(), {"P159": "headquarters location"}, top_k=1
    )
    run_formal_experiment(
        [sample], FixedPredictor(), AlwaysSupportsVerifier(), retriever, tmp_path,
        llm_model="model-a", embedding_model="hash",
    )
    with pytest.raises(ValueError, match="Resume settings do not match"):
        run_formal_experiment(
            [sample], FixedPredictor(), AlwaysSupportsVerifier(), retriever, tmp_path,
            llm_model="model-b", embedding_model="hash", resume=True,
        )


def test_formal_experiment_reject_correct_reverify_changes_final_prediction(tmp_path):
    sample = EvaluationSample(
        sample_id="s1",
        text="Apex is based in Paris.",
        benchmark_metadata={
            "entities": ["Apex", "Paris"],
            "indexed_relations": [{
                "head_index": 0,
                "tail_index": 1,
                "relation_id": "P159",
                "evidence_quotes": ["Apex is based in Paris."],
            }],
        },
    )
    report = run_formal_experiment(
        [sample], WrongRelationPredictor(), ClosedLoopVerifier(), FakeRetriever(), tmp_path,
        llm_model="test", embedding_model="fake",
    )
    assert report["conditions"]["baseline"]["f1"] == 0.0
    assert report["conditions"]["verification_only"]["f1"] == 1.0
    assert report["conditions"]["rag_verification"]["f1"] == 1.0
    assert report["workflow_actions"]["verification_only"]["corrections_proposed"] == 1
    assert report["workflow_actions"]["verification_only"]["corrected_candidates_accepted"] == 1


def test_formal_experiment_can_reuse_external_baseline_checkpoint(tmp_path):
    sample = EvaluationSample(
        sample_id="s1",
        text="Apex is based in Paris.",
        benchmark_metadata={
            "entities": ["Apex", "Paris"],
            "indexed_relations": [{
                "head_index": 0,
                "tail_index": 1,
                "relation_id": "P159",
                "evidence_quotes": ["Apex is based in Paris."],
            }],
        },
    )
    baseline_path = tmp_path / "external_baseline.jsonl"
    baseline_path.write_text(
        IndexedPrediction(
            sample_id="s1",
            relations=[IndexedRelation(
                head_index=0, tail_index=1, relation_id="P159",
                evidence_quote="Apex is based in Paris.",
            )],
        ).model_dump_json() + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "reused"
    report = run_formal_experiment(
        [sample], MustNotRunPredictor(), AlwaysSupportsVerifier(), FakeRetriever(), output,
        llm_model="test", embedding_model="fake", baseline_checkpoint=baseline_path,
    )
    assert report["conditions"]["baseline"]["f1"] == 1.0
    assert report["run"]["baseline_source_sha256"]
