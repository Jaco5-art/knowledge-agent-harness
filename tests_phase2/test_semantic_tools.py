import asyncio
import pytest
from knowledge_agents.harness.semantic_tools import SemanticTools, demo_catalog, invoke, Entity
from knowledge_agents.harness.tools import ToolRegistry, PolicyDenied


def setup():
    catalog=demo_catalog()
    registry=ToolRegistry()
    tools=SemanticTools(catalog)
    tools.register(registry)
    return catalog,registry,tools


def call(registry,name,payload,**scope):
    return asyncio.run(invoke(registry,name,payload,tenant=scope.get('tenant','demo'),scope=scope.get('scope','synthetic-business')))


def test_alias_resolved():
    _,r,_=setup()
    out=call(r,'lookup_entity_alias',{'mention':'ASTER'})
    assert out.status=='resolved' and out.matches[0].entity_id=='a'


@pytest.mark.parametrize('mention,status',[('Springfield','ambiguous'),('aster','unknown'),('missing','unknown')])
def test_no_guess(mention,status):
    _,r,_=setup()
    assert call(r,'lookup_entity_alias',{'mention':mention}).status==status


@pytest.mark.parametrize('scope',[{'tenant':'other'},{'scope':'other'}])
def test_scope_denied(scope):
    _,r,_=setup()
    with pytest.raises(PolicyDenied):
        call(r,'lookup_entity_alias',{'mention':'ASTER'},**scope)


def test_payload_cannot_choose_scope():
    _,r,_=setup()
    with pytest.raises(PolicyDenied):
        call(r,'lookup_entity_alias',{'mention':'ASTER','tenant':'demo'})


def test_namespace_status():
    _,r,_=setup()
    assert call(r,'normalize_status',{'namespace':'ticket-system','raw_status':'CLOSED'}).normalized_status=='closed'
    assert call(r,'normalize_status',{'namespace':'job-system','raw_status':'CLOSED'}).normalized_status=='completed'


def test_unknown_status_preserved():
    _,r,_=setup()
    out=call(r,'normalize_status',{'namespace':'missing','raw_status':'CLOSED'})
    assert out.status=='unknown' and out.raw_status=='CLOSED' and out.normalized_status is None


@pytest.mark.parametrize('subject,obj,pred,status',[('a','b','acquired_by','valid'),('c','b','acquired_by','invalid'),
    ('a','c','acquired_by','invalid'),('missing','b','acquired_by','unknown'),('a','b','unknown','unknown')])
def test_relation_constraints(subject,obj,pred,status):
    _,r,_=setup()
    out=call(r,'validate_relation_schema',{'subject_id':subject,'object_id':obj,'predicate':pred})
    assert out.status==status and out.evidence_verified is False


def test_config_snapshot_and_output_copy():
    c,r,t=setup()
    c.entities[0].name='changed'
    out=call(r,'lookup_entity_alias',{'mention':'ASTER'})
    out.matches[0].name='changed-again'
    assert call(r,'lookup_entity_alias',{'mention':'ASTER'}).matches[0].name=='Aster Ltd'
    assert t.version!=SemanticTools(c).version


def test_missing_type_not_passed():
    c=demo_catalog()
    c.entities[0].entity_type=None
    r=ToolRegistry()
    SemanticTools(c).register(r)
    out=call(r,'validate_relation_schema',{'subject_id':'a','object_id':'b','predicate':'acquired_by'})
    assert out.status=='unknown' and out.reason=='missing_entity_type'


def test_stage_access_guard():
    _,r,_=setup()
    with pytest.raises(PolicyDenied):
        r.get('lookup_entity_alias','extract')


def test_duplicate_entity_ids():
    c=demo_catalog().model_dump()
    c['entities'].append(c['entities'][0])
    from knowledge_agents.harness.semantic_tools import SemanticCatalog
    with pytest.raises(ValueError):
        SemanticCatalog.model_validate(c)
