import asyncio
import pytest
from knowledge_agents.harness import Harness,StateStore
from knowledge_agents.harness.adapters import FixtureAdapter,registry_for,DEMO_SOURCE
from knowledge_agents.harness.semantic_workflow import SemanticWorkflow,SemanticGate,fixture_catalog,demo
from knowledge_agents.harness.tools import PolicyDenied,ToolSpec


@pytest.fixture
def setup(tmp_path):
    store=StateStore(tmp_path/'state.sqlite')
    harness=Harness(store,registry_for(FixtureAdapter(False)))
    state=harness.create(DEMO_SOURCE,'extract entities and relationships','tenant')
    yield harness,state
    store.close()


def complete(setup):
    h,s=setup
    return asyncio.run(h.run(s.session_id,'tenant'))


def test_end_to_end_fixture():
    result=asyncio.run(demo())
    assert result['status']=='completed'
    assert len(result['ready_facts'])==2 and len(result['trace'])==6
    assert any(f['predicate']=='invested_in' for f in result['ready_facts'])


def test_upstream_incomplete_blocks_gate(setup):
    h,s=setup
    result=asyncio.run(SemanticWorkflow(h,fixture_catalog(s)).run(s.session_id,'tenant',max_steps=1))
    assert result['status']=='upstream_incomplete' and not result['ready_facts'] and not result['trace']


def test_scope_denied_before_upstream(setup):
    h,s=setup
    catalog=fixture_catalog(s)
    catalog.scope='other'
    with pytest.raises(PolicyDenied):
        asyncio.run(SemanticWorkflow(h,catalog).run(s.session_id,'tenant'))
    assert h.store.load(s.session_id,'tenant').call_count==0


@pytest.mark.parametrize('mode,reason',[('missing_rule','unmapped_relation'),('missing_type','missing_entity_type'),
    ('wrong_type','type_mismatch'),('ambiguous','entity_unresolved_or_ambiguous')])
def test_fail_closed(setup,mode,reason):
    state=complete(setup)
    catalog=fixture_catalog(state)
    if mode=='missing_rule': catalog.relations={}
    if mode=='missing_type': catalog.entities[1].entity_type=None
    if mode=='wrong_type': catalog.entities[1].entity_type='Person'
    if mode=='ambiguous': catalog.entities[0].aliases=['OpenAI']
    result=asyncio.run(SemanticGate(catalog).apply(state))
    assert result['status']=='needs_review' and not result['ready_facts']
    assert all(r['reason']==reason for r in result['needs_review'])


def test_preserves_upstream_and_recomputes(setup):
    h,_=setup
    state=complete(setup)
    before=state.model_dump_json()
    gate=SemanticGate(fixture_catalog(state))
    a=asyncio.run(gate.apply(state)); b=asyncio.run(gate.apply(state))
    assert a['ready_facts']==b['ready_facts']
    assert state.model_dump_json()==before
    assert h.store.load(state.session_id,'tenant').model_dump_json()==before


def test_bad_evidence_fails(setup):
    state=complete(setup)
    state.evidence[state.accepted[0]][0].text='altered'
    with pytest.raises(ValueError):
        asyncio.run(SemanticGate(fixture_catalog(state)).apply(state))


def test_tool_failure_quarantined(setup):
    state=complete(setup)
    gate=SemanticGate(fixture_catalog(state))
    from knowledge_agents.harness.tools import ToolRegistry
    registry=ToolRegistry()
    original=gate.registry.get('lookup_entity_alias','semantic')
    async def broken(request): raise RuntimeError('private detail')
    registry.register(ToolSpec(name=original.name,version=original.version,input_model=original.input_model,
        output_model=original.output_model,handler=broken,allowed_stages=original.allowed_stages))
    gate.registry=registry
    result=asyncio.run(gate.apply(state))
    assert not result['ready_facts'] and len(result['needs_review'])==2
    assert 'private detail' not in str(result)
