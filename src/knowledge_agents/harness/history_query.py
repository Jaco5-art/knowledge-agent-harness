"""Scoped, bounded, parameterized history queries. Not an arbitrary SQL sandbox."""
import argparse
import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Literal
from pydantic import Field
from .contracts import Contract
from .tools import ToolRegistry,ToolSpec,ToolReply,PolicyDenied


class HistoryInput(Contract):
    tenant: str
    scope: str
    subject_id: str = Field(min_length=1,max_length=200)
    predicate: str = Field(min_length=1,max_length=200)
    object_id: str = Field(min_length=1,max_length=200)
    limit: int = Field(default=10,ge=1,le=50)


class HistoryRow(Contract):
    record_id: str = Field(max_length=200)
    subject_id: str = Field(max_length=200)
    predicate: str = Field(max_length=200)
    object_id: str = Field(max_length=200)
    occurred_at: str = Field(max_length=100)
    evidence_ref: str = Field(max_length=2000)


class HistoryOutput(Contract):
    status: Literal['found','empty']
    records: list[HistoryRow]
    has_more: bool
    evidence_verified: bool = False


COLUMNS={'tenant','scope','record_id','subject_id','predicate','object_id','occurred_at','evidence_ref'}


def read_authorizer(action,arg1,arg2,db,source):
    if action==sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action==sqlite3.SQLITE_READ and db=='main' and arg1=='relation_history' and arg2 in COLUMNS and source is None:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class HistoryTool:
    def __init__(self,path,*,tenant,scope,dataset_revision,timeout=2,max_calls=100):
        if not tenant or not scope or not dataset_revision or timeout<=0 or max_calls<1:
            raise ValueError('Explicit scope, revision and positive limits required')
        self.path=Path(path).resolve(strict=True)
        self.tenant,self.scope=tenant,scope
        self.timeout,self.max_calls=timeout,max_calls
        self.calls=0
        self.trace=[]
        # Dataset revision is a trusted operator label, not a file content hash.
        identity=json.dumps([tenant,scope,dataset_revision,timeout,max_calls])
        self.version='history-query-v1:'+hashlib.sha256(identity.encode()).hexdigest()

    def _read(self,request):
        start=time.monotonic()
        conn=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=self.timeout)
        try:
            conn.execute('PRAGMA query_only=ON')
            conn.execute('PRAGMA trusted_schema=OFF')
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,65536)
            conn.set_authorizer(read_authorizer)
            conn.set_progress_handler(lambda: int(time.monotonic()-start>=self.timeout),100)
            rows=conn.execute('''SELECT record_id,subject_id,predicate,object_id,occurred_at,evidence_ref
                FROM relation_history WHERE tenant=? AND scope=? AND subject_id=? AND predicate=? AND object_id=?
                ORDER BY occurred_at DESC,record_id ASC LIMIT ?''',
                (self.tenant,self.scope,request.subject_id,request.predicate,request.object_id,request.limit+1)).fetchall()
            if time.monotonic()-start>=self.timeout:
                raise TimeoutError('Query deadline exceeded')
            values=[HistoryRow(**dict(zip(('record_id','subject_id','predicate','object_id','occurred_at','evidence_ref'),r))) for r in rows[:request.limit]]
            return HistoryOutput(status='found' if values else 'empty',records=values,has_more=len(rows)>request.limit)
        except sqlite3.OperationalError as exc:
            if time.monotonic()-start>=self.timeout:
                raise TimeoutError('Query deadline exceeded') from exc
            raise
        finally: conn.close()

    async def query(self,request):
        if (request.tenant,request.scope)!=(self.tenant,self.scope):
            raise PolicyDenied('History scope denied')
        if self.calls>=self.max_calls:
            raise PolicyDenied('History call budget exhausted')
        self.calls+=1
        event=dict(tool='query_relation_history',call_number=self.calls,version=self.version,
                   request_hash=hashlib.sha256(request.model_dump_json().encode()).hexdigest())
        start=time.perf_counter()
        try:
            output=await asyncio.to_thread(self._read,request)
            event.update(status='succeeded',rows=len(output.records),has_more=output.has_more)
            return ToolReply(output)
        except asyncio.CancelledError:
            event.update(status='cancelled')
            raise
        except Exception as exc:
            event.update(status='failed',error_type=type(exc).__name__)
            raise
        finally:
            event['latency_ms']=(time.perf_counter()-start)*1000
            self.trace.append(event)

    def register(self,registry):
        registry.register(ToolSpec(name='query_relation_history',version=self.version,input_model=HistoryInput,
            output_model=HistoryOutput,handler=self.query,allowed_stages=frozenset({'history'})))


async def invoke_history(registry,payload,*,tenant,scope):
    if 'tenant' in payload or 'scope' in payload:
        raise PolicyDenied('Scope must come from trusted caller')
    spec=registry.get('query_relation_history','history')
    request=spec.input_model.model_validate(dict(payload,tenant=tenant,scope=scope))
    result=await spec.handler(request)
    return spec.output_model.model_validate(result.data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,required=True)
    p.add_argument('--tenant',required=True); p.add_argument('--scope',required=True)
    p.add_argument('--dataset-revision',required=True)
    p.add_argument('--subject-id',required=True); p.add_argument('--predicate',required=True)
    p.add_argument('--object-id',required=True); p.add_argument('--limit',type=int,default=10)
    args=p.parse_args()
    tool=HistoryTool(args.db,tenant=args.tenant,scope=args.scope,dataset_revision=args.dataset_revision)
    registry=ToolRegistry(); tool.register(registry)
    result=asyncio.run(invoke_history(registry,dict(subject_id=args.subject_id,predicate=args.predicate,
        object_id=args.object_id,limit=args.limit),tenant=args.tenant,scope=args.scope))
    print(json.dumps(dict(result=result.model_dump(),trace=tool.trace),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
