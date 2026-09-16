import copy
import pytest
from knowledge_agents.harness.semantic import project, demo, read_session
from knowledge_agents.harness.contracts import Candidate, Fact
from knowledge_agents.harness.semantic import from_session


def candidate(i='1',**updates):
    values=dict(subject='A',predicate='acquired',object='B',quote='A acquired B.')
    values.update(updates)
    return Candidate(candidate_id=i,fact=Fact(**values))


def test_demo_split_and_merge():
    r=demo()
    assert [len(r[k]) for k in ('business_facts','entity_mentions','document_statements','needs_review')]==[1,1,1,1]
    assert r['merged_records']==1
    assert len(r['original_candidates'])==5


def test_original_not_mutated():
    inputs=[candidate(),candidate('2',predicate='acquired',fact_type='event')]
    before=copy.deepcopy(inputs)
    r=project(inputs,scope='doc')
    assert inputs==before
    assert r['business_facts'][0]['candidate_ids']==['1','2']


@pytest.mark.parametrize('change',[{'time':'2023'},{'location':'London'},
    {'subject':'B','object':'A'},{'predicate':'acquired_by'},{'subject':'a'}])
def test_qualifiers_direction_case_not_merged(change):
    assert len(project([candidate(),candidate('2',**change)],scope='doc')['business_facts'])==2


@pytest.mark.parametrize('predicate',['did not acquire','may acquire','plans to acquire','not_acquired'])
def test_negation_modality_not_positive(predicate):
    r=project([candidate(predicate=predicate)],scope='doc')
    assert not r['business_facts'] and len(r['needs_review'])==1


def test_ambiguous_alias():
    r=project([candidate()],scope='doc',aliases={'A':['Company A1','Company A2']})
    assert not r['business_facts']
    assert r['needs_review'][0]['reason']=='ambiguous_or_missing_entity'


def test_explicit_alias_preserves_original():
    r=project([candidate()],scope='doc',aliases={'A':['Company A']})
    assert r['business_facts'][0]['subject']=='Company A'
    assert r['original_candidates'][0]['fact']['subject']=='A'
    assert r['business_facts'][0]['alias_applied']


def test_alias_chain_rejected():
    with pytest.raises(ValueError):
        project([candidate()],scope='doc',aliases={'A':['B'],'B':['C']})


def test_unknown_predicate_preserved_for_review():
    r=project([candidate(predicate='funded')],scope='doc')
    assert r['original_candidates'][0]['fact']['predicate']=='funded'
    assert r['needs_review'][0]['reason']=='unmapped_predicate'


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        project([candidate(),candidate()],scope='doc')


def test_missing_db_not_created(tmp_path):
    path=tmp_path/'missing.sqlite'
    with pytest.raises(FileNotFoundError):
        read_session(path,'id','tenant')
    assert not path.exists()


def test_readonly_export_and_tenant_boundary(tmp_path):
    import asyncio
    from knowledge_agents.harness import Harness, StateStore
    from knowledge_agents.harness.adapters import FixtureAdapter, registry_for, DEMO_SOURCE
    path=tmp_path/'state.sqlite'
    store=StateStore(path)
    harness=Harness(store,registry_for(FixtureAdapter(False)))
    session=harness.create(DEMO_SOURCE,'extract entities and relationships','tenant')
    session=asyncio.run(harness.run(session.session_id,'tenant'))
    store.close()
    before=path.read_bytes()
    result=from_session(read_session(path,session.session_id,'tenant'))
    assert len(result['business_facts'])==2
    assert path.read_bytes()==before
    with pytest.raises(ValueError):
        read_session(path,session.session_id,'other-tenant')
    session.accepted.append('nonexistent')
    with pytest.raises(ValueError):
        from_session(session)
