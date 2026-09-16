import asyncio
import sqlite3
import pytest
from knowledge_agents.harness.history_query import HistoryTool,invoke_history,read_authorizer
from knowledge_agents.harness.tools import ToolRegistry,PolicyDenied


@pytest.fixture
def database(tmp_path):
    path=tmp_path/'business.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE relation_history(tenant TEXT,scope TEXT,record_id TEXT PRIMARY KEY,subject_id TEXT,predicate TEXT,object_id TEXT,occurred_at TEXT,evidence_ref TEXT)')
        conn.executemany('INSERT INTO relation_history VALUES(?,?,?,?,?,?,?,?)',[
            ('t','s','r1','a','invested_in','b','2023-01-01','news:1'),
            ('t','s','r2','a','invested_in','b','2024-01-01','news:2'),
            ('other','s','r3','a','invested_in','b','2025-01-01','private'),
            ('t','other','r4','a','invested_in','b','2026-01-01','private'),
            ('t','s','r5','b','invested_in','a','2023-01-01','reverse')])
    return path


def setup(path,**kwargs):
    tool=HistoryTool(path,tenant='t',scope='s',dataset_revision='fixture-1',**kwargs)
    registry=ToolRegistry(); tool.register(registry)
    return tool,registry


def query(registry,**updates):
    payload=dict(subject_id='a',predicate='invested_in',object_id='b'); payload.update(updates)
    return asyncio.run(invoke_history(registry,payload,tenant='t',scope='s'))


def test_scoped_sorted_readonly(database):
    before=database.read_bytes(); tool,r=setup(database)
    output=query(r)
    assert [v.record_id for v in output.records]==['r2','r1']
    assert not output.evidence_verified and not output.has_more
    assert before==database.read_bytes() and tool.trace[0]['rows']==2


def test_limit_and_has_more(database):
    _,r=setup(database); out=query(r,limit=1)
    assert len(out.records)==1 and out.has_more


@pytest.mark.parametrize('limit',[0,51,-1])
def test_invalid_limit_no_execution(database,limit):
    tool,r=setup(database)
    with pytest.raises(ValueError): query(r,limit=limit)
    assert tool.calls==0


def test_injection_literal(database):
    _,r=setup(database)
    assert query(r,subject_id="a' OR 1=1 --").status=='empty'


def test_no_sql_payload(database):
    tool,r=setup(database)
    with pytest.raises(ValueError): query(r,sql='DELETE FROM relation_history')
    assert tool.calls==0


def test_scope_override_denied(database):
    tool,r=setup(database)
    with pytest.raises(PolicyDenied): query(r,tenant='other')
    with pytest.raises(PolicyDenied):
        asyncio.run(invoke_history(r,dict(subject_id='a',predicate='invested_in',object_id='b'),tenant='other',scope='s'))
    assert tool.calls==0


def test_budget(database):
    _,r=setup(database,max_calls=1); query(r)
    with pytest.raises(PolicyDenied): query(r)


def test_reverse_not_matching(database):
    _,r=setup(database)
    assert [x.record_id for x in query(r,subject_id='b',object_id='a').records]==['r5']


def test_authorizer_denies_write_and_attach(database):
    with sqlite3.connect(database) as conn:
        conn.set_authorizer(read_authorizer)
        for sql in ('DELETE FROM relation_history',"ATTACH DATABASE ':memory:' AS other",'SELECT * FROM sqlite_master'):
            with pytest.raises(sqlite3.DatabaseError): conn.execute(sql)


def test_deadline(database, monkeypatch):
    from types import SimpleNamespace
    import knowledge_agents.harness.history_query as module

    tool, r = setup(database, timeout=2)
    ticks = iter([100.0])

    # Patch this module only; leave asyncio's real clock untouched.
    fake_time = SimpleNamespace(
        monotonic=lambda: next(ticks, 103.0),
        perf_counter=module.time.perf_counter,
    )
    monkeypatch.setattr(module, "time", fake_time)

    with pytest.raises(TimeoutError, match="Query deadline exceeded"):
        query(r)

    assert tool.calls == 1
    assert tool.trace[0]["status"] == "failed"
    assert tool.trace[0]["error_type"] == "TimeoutError"


def test_missing_file_no_creation(tmp_path):
    path=tmp_path/'absent.sqlite'
    with pytest.raises(FileNotFoundError): setup(path)
    assert not path.exists()


def test_wrong_stage(database):
    _,r=setup(database)
    with pytest.raises(PolicyDenied): r.get('query_relation_history','verify')


def test_missing_schema_error_traced(tmp_path):
    path=tmp_path/'empty.sqlite'
    sqlite3.connect(path).close()
    tool,r=setup(path)
    with pytest.raises(sqlite3.Error): query(r)
    assert tool.trace[0]['status']=='failed' and 'error_type' in tool.trace[0]
