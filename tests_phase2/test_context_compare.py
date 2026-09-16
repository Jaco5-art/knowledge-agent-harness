from knowledge_agents.harness.context_compare import run_comparison, assert_lossless
import pytest


def test_paired_offline_comparison():
    report = run_comparison()
    rows = {r['case']:r['modes'] for r in report['cases']}
    assert report['provider_calls'] == 0
    for name in ('high_overlap','nested_overlap'):
        assert rows[name]['compact']['compacted']
        assert rows[name]['compact']['payload_bytes'] < rows[name]['budget']['payload_bytes']
    for name in ('short_single','no_overlap'):
        assert not rows[name]['compact']['compacted']
    for mode in ('budget','compact'):
        assert rows['over_budget'][mode]['status'] == 'context_budget_exceeded'


def test_lossless_checker_detects_changes():
    with pytest.raises(AssertionError):
        assert_lossless('{"candidate":1,"evidence":[]}', '{"candidate":2,"evidence":[]}')
