from pathlib import Path

from knowledge_agents.evaluation import EvaluationSample
from knowledge_agents.rag import HashEmbeddingProvider
from knowledge_agents.redocred import load_and_sample
from knowledge_agents.redocred_predictor import IndexedPrediction, IndexedRelation
from knowledge_agents.redocred_rag import FAISSRelationEvidenceRetriever, retrieval_metrics
from knowledge_agents.redocred_verification import RelationVerdict, verify_and_correct


FIXTURES = Path(__file__).parent / "fixtures"


class EvidenceAwareVerifier:
    def __init__(self):
        self.calls = []

    def verify_with_evidence(self, sample, candidates, evidence_by_candidate):
        self.calls.append((candidates, evidence_by_candidate))
        return [
            RelationVerdict(
                candidate_index=index,
                verdict="supported",
                confidence=0.95,
                reason="Retrieved evidence supports the relation.",
            )
            for index in range(len(candidates))
        ]


def test_rag_verification_retrieves_evidence_before_acceptance():
    sample = EvaluationSample(
        sample_id="s1",
        text="Apex is based in Paris. A separate sentence discusses weather.",
        benchmark_metadata={"entities": ["Apex", "Paris"]},
    )
    prediction = IndexedPrediction(
        sample_id="s1",
        relations=[
            IndexedRelation(
                head_index=0,
                tail_index=1,
                relation_id="P159",
                evidence_quote="Apex is based in Paris",
            )
        ],
    )
    retriever = FAISSRelationEvidenceRetriever(
        HashEmbeddingProvider(), {"P159": "headquarters location"}, top_k=1, chunk_size=35, overlap=5
    )
    verifier = EvidenceAwareVerifier()
    result = verify_and_correct(sample, prediction, verifier, evidence_retriever=retriever)
    assert len(result.relations) == 1
    assert verifier.calls[0][1][0]
    assert "Apex" in verifier.calls[0][1][0][0].text


def test_retrieval_metrics_use_public_gold_evidence_sentences():
    samples = load_and_sample(
        FIXTURES / "redocred_fixture.json",
        FIXTURES / "docred_relations.json",
        sample_size=2,
        seed=42,
    )
    retriever = FAISSRelationEvidenceRetriever(
        HashEmbeddingProvider(),
        {"P159": "headquarters location", "P112": "founded by"},
        top_k=1,
        chunk_size=100,
        overlap=10,
    )
    metrics = retrieval_metrics(samples, retriever)
    assert metrics["gold_relations_with_evidence"] == 2
    assert metrics["evidence_recall_at_k"] == 1.0
