from pathlib import Path
import json

import pytest

from knowledge_agents.redocred import load_and_sample, write_jsonl, write_manifest


FIXTURES = Path(__file__).parent / "fixtures"


def test_redocred_adapter_maps_entities_and_relations():
    samples = load_and_sample(
        FIXTURES / "redocred_fixture.json",
        FIXTURES / "docred_relations.json",
        sample_size=2,
        seed=42,
    )
    assert len(samples) == 2
    assert samples[0].review_status == "public_benchmark"
    predicates = {fact.predicate for sample in samples for fact in sample.gold_facts}
    assert predicates == {"headquarters location", "founded by"}


def test_redocred_sampling_is_reproducible():
    first = load_and_sample(FIXTURES / "redocred_fixture.json", sample_size=1, seed=7)
    second = load_and_sample(FIXTURES / "redocred_fixture.json", sample_size=1, seed=7)
    assert first[0].model_dump() == second[0].model_dump()


def test_redocred_rejects_oversized_sample():
    with pytest.raises(ValueError):
        load_and_sample(FIXTURES / "redocred_fixture.json", sample_size=3)


def test_manifest_records_frozen_subset_and_hashes(tmp_path):
    samples = load_and_sample(
        FIXTURES / "redocred_fixture.json",
        FIXTURES / "docred_relations.json",
        sample_size=1,
        seed=7,
    )
    dataset = write_jsonl(samples, tmp_path / "subset.jsonl")
    manifest_path = write_manifest(
        samples,
        dataset,
        FIXTURES / "redocred_fixture.json",
        tmp_path / "manifest.json",
        relation_map_path=FIXTURES / "docred_relations.json",
        sample_size=1,
        seed=7,
        source_revision="test-revision",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_revision"] == "test-revision"
    assert manifest["sample_ids"] == [samples[0].sample_id]
    assert len(manifest["derived_dataset_sha256"]) == 64
