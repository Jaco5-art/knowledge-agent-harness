import asyncio
import json
import pytest
from knowledge_agents.harness.workbench import main,stage_demo,ReadOnlyReviews,export_publication
from knowledge_agents.harness import Harness,StateStore
from knowledge_agents.harness.adapters import FixtureAdapter,registry_for,DEMO_SOURCE
from knowledge_agents.harness.semantic_workflow import fixture_catalog
from knowledge_agents.harness.semantic_review import ReviewStore
from knowledge_agents.harness.store import ConflictError


@pytest.fixture
def setup(tmp_path):
    db=tmp_path/'upstream.sqlite'; store=StateStore(db)
    h=Harness(store,registry_for(FixtureAdapter(False)))
    state=h.create(DEMO_SOURCE,'extract entities and relationships','t')
    state=asyncio.run(h.run(state.session_id,'t')); store.close()
    return tmp_path,db,state


def test_stage_demo():
    result=asyncio.run(stage_demo())
    assert result['status']=='passed' and result['provider_calls']==0
    assert result['published_fact_count']==2 and result['history_actions']==['create','retry']


def test_complete_cli_lifecycle(setup):
    path,db,state=setup
    catalog=fixture_catalog(state); good=catalog.model_copy(deep=True)
    catalog.entities[1].entity_type=None
    catalog_path=path/'catalog.json'; catalog_path.write_text(catalog.model_dump_json())
    prefix=['--review-db',str(path/'review.sqlite'),'--tenant','t']
    sessions=main(['--tenant','t','sessions','--db',str(db)])
    assert sessions[0]['session_id']==state.session_id
    record=main(prefix+['check','--db',str(db),'--session',state.session_id,'--catalog',str(catalog_path),'--actor','tester'])
    rid=record['review_id']
    assert record['publishable']==0
    assert len(main(prefix+['reviews']))==1
    assert main(prefix+['show','--review-id',rid])['status']=='needs_review'
    with pytest.raises(ValueError):
        main(prefix+['export','--review-id',rid,'--expected-version','0','--output',str(path/'blocked.json')])
    assert not (path/'blocked.json').exists()
    catalog_path.write_text(good.model_dump_json())
    record=main(prefix+['retry','--review-id',rid,'--expected-version','0','--actor','tester','--catalog',str(catalog_path)])
    assert record['publishable']==2
    assert len(main(prefix+['history','--review-id',rid]))==2
    main(prefix+['export','--review-id',rid,'--expected-version','1','--output',str(path/'published.json')])
    assert len(json.loads((path/'published.json').read_text())['facts'])==2


def test_export_no_overwrite_and_stale(setup):
    path,_,state=setup; store=ReviewStore(path/'r.sqlite')
    try:
        record=asyncio.run(store.create(state,fixture_catalog(state),'tester'))
        output=path/'existing'; output.write_text('keep')
        with pytest.raises(FileExistsError): export_publication(store,record['review_id'],'t',0,output)
        with pytest.raises(ConflictError): export_publication(store,record['review_id'],'t',8,path/'new')
        assert output.read_text()=='keep' and not (path/'new').exists()
    finally: store.close()


def test_read_commands_no_creation(tmp_path):
    with pytest.raises(SystemExit): main(['--review-db',str(tmp_path/'absent.sqlite'),'reviews'])
    assert not (tmp_path/'absent.sqlite').exists()


def test_readonly_and_tenant_filter(setup):
    path,_,state=setup; db=path/'r.sqlite'; store=ReviewStore(db)
    asyncio.run(store.create(state,fixture_catalog(state),'tester')); store.close()
    before=db.read_bytes()
    assert main(['--review-db',str(db),'--tenant','other','reviews'])==[]
    assert db.read_bytes()==before
    ro=ReadOnlyReviews(db)
    try:
        import sqlite3
        with pytest.raises(sqlite3.OperationalError): ro.connection.execute('DELETE FROM semantic_records')
    finally: ro.close()


@pytest.mark.parametrize('action',['reject','cancel'])
def test_terminal_cli_blocks_export(setup,action):
    path,_,state=setup; db=path/'r.sqlite'; store=ReviewStore(db)
    catalog=fixture_catalog(state); catalog.relations={}
    record=asyncio.run(store.create(state,catalog,'tester')); store.close()
    prefix=['--review-db',str(db),'--tenant','t']
    main(prefix+[action,'--review-id',record['review_id'],'--expected-version','0','--actor','tester'])
    with pytest.raises(ValueError):
        main(prefix+['export','--review-id',record['review_id'],'--expected-version','1','--output',str(path/'out.json')])


def test_wrong_catalog_no_review_db_created(setup):
    path,db,state=setup; catalog=fixture_catalog(state); catalog.scope='wrong'
    config=path/'catalog.json'; config.write_text(catalog.model_dump_json())
    from knowledge_agents.harness.tools import PolicyDenied
    with pytest.raises(PolicyDenied):
        main(['--review-db',str(path/'new.sqlite'),'--tenant','t','check','--db',str(db),'--session',state.session_id,'--catalog',str(config),'--actor','tester'])
    assert not (path/'new.sqlite').exists()
