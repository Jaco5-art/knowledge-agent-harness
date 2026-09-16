import json

from knowledge_agents.evaluation import EvaluationSample, LabeledFact
from knowledge_agents.experiment import run_experiment
from knowledge_agents.models import Evidence, ExtractedFact, FactType
from knowledge_agents.rag import HashEmbeddingProvider


class ExactProvider:
    def extract(self, source_text, fact_type):
        if fact_type is not FactType.EVENT:
            return []
        return [ExtractedFact(fact_id="1", fact_type=FactType.EVENT, subject="A", predicate="invested in", object="B", evidence=[Evidence(quote=source_text)], extraction_confidence=0.9, source_agent="event_agent")]

    def correct(self, source_text, facts, results, correction_round):
        return facts


def test_experiment_writes_predictions_and_metrics(tmp_path):
    samples = [EvaluationSample(sample_id="s1", text="A invested in B.", gold_facts=[LabeledFact(subject="A", predicate="invested in", object="B")])]
    metrics = run_experiment(samples, ExactProvider(), HashEmbeddingProvider(), tmp_path)
    assert metrics["baseline"]["f1"] == 1.0
    assert metrics["verified_system"]["f1"] == 1.0
    assert json.loads((tmp_path / "metrics.json").read_text())["baseline"]["samples"] == 1

