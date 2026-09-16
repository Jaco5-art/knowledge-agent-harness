from knowledge_agents.evaluation import EvaluationSample, LabeledFact, PredictionRecord, compare_systems, score_predictions


def test_exact_fact_metrics():
    samples = [EvaluationSample(sample_id="1", text="x", gold_facts=[LabeledFact(subject="A", predicate="supports", object="B")])]
    predictions = [PredictionRecord(sample_id="1", facts=[LabeledFact(subject="A", predicate="supports", object="B")])]
    metrics = score_predictions(samples, predictions)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_comparison_counts_removed_false_positive():
    samples = [EvaluationSample(sample_id="1", text="x", gold_facts=[LabeledFact(subject="A", predicate="invested in", object="B")])]
    baseline = [PredictionRecord(sample_id="1", facts=[LabeledFact(subject="A", predicate="acquired", object="B")])]
    verified = [PredictionRecord(sample_id="1", facts=[LabeledFact(subject="A", predicate="invested in", object="B")])]
    result = compare_systems(samples, baseline, verified)
    assert result["false_positives_removed"] == 1
    assert result["error_detection_rate"] == 1.0

