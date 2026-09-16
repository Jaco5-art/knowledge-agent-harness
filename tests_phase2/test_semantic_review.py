import asyncio
import sqlite3
import pytest
from knowledge_agents.harness import Harness,StateStore
from knowledge_agents.harness.adapters import FixtureAdapter,registry_for,DEMO_SOURCE
from knowledge_agents.harness.semantic_workflow import fixture_catalog
from knowledge_agents.harness.semantic_review import ReviewStore
from knowledge_agents.harness.store import ConflictError


@pytest.fixture
def setup(tmp_path):
    upstream=StateStore(tmp_path/'upstream.sqlite')
    h=Harness(upstream,registry_for(FixtureAdapter(False)))
    state=h.create(DEMO_SOURCE,'extract entities and relationships','tenant')
    state=asyncio.run(h.run(state.session_id,'tenant')); upstream.close()
    store=ReviewStore(tmp_path/'reviews.sqlite')
    good=fixture_catalog(state); bad=good.model_copy(deep=True); bad.entities[1].entity_type=None
    yield store,state,good,bad,tmp_path
    store.close()


def create(setup):
    store,state,good,bad,_=setup
    return asyncio.run(store.create(state,bad,'operator'))


def test_resume_and_retry(setup):
    store,state,good,_,path=setup
    before=state.model_dump_json(); body=create(setup)
    assert store.publication(body['review_id'],'tenant')['facts']==[]
    reopened=ReviewStore(path/'reviews.sqlite')
    try:
        result=asyncio.run(reopened.resolve(body['review_id'],'tenant',0,'operator','retry',good))
        assert result['status']=='completed' and result['version']==1
        assert len(reopened.publication(body['review_id'],'tenant')['facts'])==2
        assert len(reopened.history(body['review_id'],'tenant'))==2
        assert state.model_dump_json()==before
    finally: reopened.close()


@pytest.mark.parametrize('action',['reject','cancel'])
def test_terminal_no_publication(setup,action):
    store,_,_,_,_=setup; body=create(setup)
    asyncio.run(store.resolve(body['review_id'],'tenant',0,'operator',action))
    assert not store.publication(body['review_id'],'tenant')['facts']
    with pytest.raises(ValueError):
        asyncio.run(store.resolve(body['review_id'],'tenant',1,'operator',action))


def test_stale_version(setup):
    store,_,_,_,_=setup; body=create(setup)
    asyncio.run(store.resolve(body['review_id'],'tenant',0,'operator','reject'))
    with pytest.raises(ConflictError):
        asyncio.run(store.resolve(body['review_id'],'tenant',0,'operator','cancel'))


def test_no_accept_bypass(setup):
    store,_,_,_,_=setup; body=create(setup)
    with pytest.raises(ValueError):
        asyncio.run(store.resolve(body['review_id'],'tenant',0,'operator','accept'))


def test_revision_only_no_progress(setup):
    store,_,_,bad,_=setup; body=create(setup); bad.revision='2'
    with pytest.raises(ValueError):
        asyncio.run(store.resolve(body['review_id'],'tenant',0,'operator','retry',bad))
    assert store.show(body['review_id'],'tenant')['version']==0


def test_wrong_scope_and_tenant(setup):
    store,_,good,_,_=setup; body=create(setup)
    with pytest.raises(ValueError): store.show(body['review_id'],'other')
    good.scope='other'
    from knowledge_agents.harness.tools import PolicyDenied
    with pytest.raises(PolicyDenied):
        asyncio.run(store.resolve(body['review_id'],'tenant',0,'operator','retry',good))
    assert len(store.history(body['review_id'],'tenant'))==1


def test_idempotent_create(setup):
    store,_,_,_,_=setup; a=create(setup); b=create(setup)
    assert a==b and len(store.history(a['review_id'],'tenant'))==1


def test_do_not_initialize_upstream_db(setup):
    _,_,_,_,path=setup
    data=(path/'upstream.sqlite').read_bytes()
    with pytest.raises(ValueError): ReviewStore(path/'upstream.sqlite')
    assert data==(path/'upstream.sqlite').read_bytes()


def test_atomic_rollback_on_history_failure(setup):
    store,_,_,_,_=setup; body=create(setup)
    store.connection.execute("CREATE TRIGGER block_history BEFORE INSERT ON semantic_history BEGIN SELECT RAISE(ABORT, 'injected'); END")
    store.connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(store.resolve(body['review_id'],'tenant',0,'operator','reject'))
    assert store.show(body['review_id'],'tenant')['version']==0


def test_retry_budget(setup):
    store,_,_,bad,_=setup; body=create(setup)
    for i in range(3):
        bad.entities[0].aliases.append('alias'+str(i))
        body=asyncio.run(store.resolve(body['review_id'],'tenant',i,'operator','retry',bad))
    bad.entities[0].aliases.append('another')
    with pytest.raises(ValueError):
        asyncio.run(store.resolve(body['review_id'],'tenant',3,'operator','retry',bad))
