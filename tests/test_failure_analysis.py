from knowledge_agents.evaluation import EvaluationSample
from knowledge_agents.failure_analysis import analyze_failures
from knowledge_agents.redocred_predictor import IndexedPrediction, IndexedRelation


def prediction(sample_id, relations):
    return IndexedPrediction(
        sample_id=sample_id,
        relations=[IndexedRelation(head_index=h, tail_index=t, relation_id=r, evidence_quote="e") for h, t, r in relations],
    )


def test_failure_analysis_distinguishes_helpful_filtering_from_false_rejection():
    sample = EvaluationSample(
        sample_id="s1",
        text="text",
        benchmark_metadata={
            "entities": ["A", "B"],
            "indexed_relations": [{"head_index": 0, "tail_index": 1, "relation_id": "P1"}],
        },
    )
    baseline = [prediction("s1", [(0, 1, "P1"), (1, 0, "P2")])]
    verified = [prediction("s1", [(0, 1, "P1")])]
    rag = [prediction("s1", [])]
    summary, rows = analyze_failures([sample], baseline, verified, rag, {"P1": "true", "P2": "false"})
    assert summary["verification_removed_baseline_false_positives"] == 1
    assert summary["rag_lost_baseline_true_positives"] == 1
    assert {row["category"] for row in rows} == {"verification_removed_false_positive", "rag_false_rejection"}
