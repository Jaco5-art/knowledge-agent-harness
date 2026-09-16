from knowledge_agents.benchmark import build_synthetic_benchmark


def test_synthetic_benchmark_has_100_unique_samples():
    samples = build_synthetic_benchmark()
    assert len(samples) == 100
    assert len({sample.sample_id for sample in samples}) == 100
    assert any(not sample.gold_facts for sample in samples)

