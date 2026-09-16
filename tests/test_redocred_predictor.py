import pytest
from pydantic import ValidationError

from knowledge_agents.redocred_predictor import IndexedPrediction, IndexedRelation, build_output_model, score_indexed


def test_dynamic_schema_accepts_only_catalog_relations():
    model = build_output_model(["P112", "P159"])
    parsed = model.model_validate({"relations": [{"head_index": 0, "tail_index": 1, "relation_id": "P159", "evidence_quote": "based in Paris"}]})
    assert parsed.relations[0].relation_id == "P159"
    with pytest.raises(ValidationError):
        model.model_validate({"relations": [{"head_index": 0, "tail_index": 1, "relation_id": "invented", "evidence_quote": "x"}]})


def test_indexed_scorer_avoids_wording_mismatch():
    gold = [IndexedPrediction(sample_id="s1", relations=[IndexedRelation(head_index=0, tail_index=1, relation_id="P159", evidence_quote="")])]
    predicted = [IndexedPrediction(sample_id="s1", relations=[IndexedRelation(head_index=0, tail_index=1, relation_id="P159", evidence_quote="Apex is based in Paris")])]
    metrics = score_indexed(gold, predicted)
    assert metrics["f1"] == 1.0

