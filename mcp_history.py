"""Local stdio MCP server/client for operator-scoped read-only history queries."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from pydantic import BaseModel, ConfigDict, Field
from knowledge_agents.harness.history_query import HistoryTool, invoke_history
from knowledge_agents.harness.tools import ToolRegistry


class Query(BaseModel):
    model_config=ConfigDict(extra='forbid')
    subject_id:str=Field(min_length=1,max_length=200)
    predicate:str=Field(min_length=1,max_length=200)
    object_id:str=Field(min_length=1,max_length=200)
    limit:int=Field(default=10,ge=1,le=50)


def serve(args):
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations
    tool=HistoryTool(args.db,tenant=args.tenant,scope=args.scope,dataset_revision=args.revision)
    registry=ToolRegistry();tool.register(registry)
    server=MCPServer('Scoped history',log_level='ERROR')
    @server.tool(annotations=ToolAnnotations(readOnlyHint=True,destructiveHint=False))
    async def query_relation_history(request:Query)->dict:
        """Read relation history in the operator-configured tenant and document scope."""
        result=await invoke_history(registry,request.model_dump(),tenant=args.tenant,scope=args.scope)
        return result.model_dump()
    server.run(transport='stdio')


async def client(args,check=False):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params=StdioServerParameters(command=sys.executable,args=[str(Path(__file__).resolve()),'serve',
        '--db',str(args.db.resolve()),'--tenant',args.tenant,'--scope',args.scope,'--revision',args.revision])
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            listed=await session.list_tools()
            assert [t.name for t in listed.tools]==['query_relation_history']
            payload=dict(subject_id=args.subject,predicate=args.predicate,object_id=args.object,limit=10)
            async def call(value):
                return await session.call_tool('query_relation_history',arguments={'request':value})
            result=await call(payload)
            if result.is_error:
                raise RuntimeError('MCP query returned an error')
            data=result.structured_content
            if data is None:
                data=json.loads(next(c.text for c in result.content if c.type=='text'))
            if not check:
                return data
            checks={'scoped_result':[r['record_id'] for r in data['records']]==['r1'],
                    'reference_not_verified':data['evidence_verified'] is False}
            for label,update in [('tenant_override',{'tenant':'other'}),('scope_override',{'scope':'other'}),
                                 ('sql_payload',{'sql':'DELETE FROM relation_history'}),('invalid_limit',{'limit':51})]:
                rejected=await call(dict(payload,**update))
                checks[label]=bool(rejected.is_error)
            injected=await call(dict(payload,subject_id="a' OR 1=1 --"))
            value=injected.structured_content
            if value is None:
                value=json.loads(next(c.text for c in injected.content if c.type=='text'))
            checks['injection_is_literal']=not injected.is_error and value['records']==[]
            return checks


async def self_test():
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as folder:
        db=Path(folder)/'history.sqlite'
        conn=sqlite3.connect(db)
        try:
            conn.execute('CREATE TABLE relation_history(tenant TEXT,scope TEXT,record_id TEXT PRIMARY KEY,subject_id TEXT,predicate TEXT,object_id TEXT,occurred_at TEXT,evidence_ref TEXT)')
            conn.executemany('INSERT INTO relation_history VALUES(?,?,?,?,?,?,?,?)',[
                ('t','s','r1','a','invested_in','b','2023','fixture:1'),
                ('other','s','r2','a','invested_in','b','2024','private'),
                ('t','other','r3','a','invested_in','b','2025','private')])
            conn.commit()
        finally:
            conn.close()
        before=db.read_bytes()
        args=SimpleNamespace(db=db,tenant='t',scope='s',revision='fixture-v1',subject='a',predicate='invested_in',object='b')
        checks=await client(args,True)
        checks['database_unchanged']=before==db.read_bytes()
        db.rename(db.with_name('closed.sqlite'))
        checks['handles_closed']=True
    import importlib.metadata
    report=dict(passed=all(checks.values()),api_calls=0,transport='stdio',mcp_version=importlib.metadata.version('mcp'),checks=checks,
        limitations=['Local child-process transport; scope is fixed by the operator at server startup.',
                     'Not a remote multi-user authentication or HTTP deployment test.',
                     'Read-only hint is metadata; SQLite read-only mode and existing authorizer enforce restrictions.'])
    path=Path('mcp_history_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(path.resolve())
    if not report['passed']:
        raise SystemExit('MCP CHECK FAILED: '+str(checks))
    print('MCP OK: real stdio client/server; 9 checks passed; 0 API calls.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['serve','query','self-test'])
    p.add_argument('--db',type=Path);p.add_argument('--tenant');p.add_argument('--scope');p.add_argument('--revision')
    p.add_argument('--subject');p.add_argument('--predicate');p.add_argument('--object')
    args=p.parse_args()
    if args.command=='self-test':
        asyncio.run(asyncio.wait_for(self_test(),60));return
    if not all((args.db,args.tenant,args.scope,args.revision)):
        p.error('Provide --db --tenant --scope --revision')
    if args.command=='serve':
        serve(args);return
    if not all((args.subject,args.predicate,args.object)):
        p.error('Provide --subject --predicate --object')
    print(json.dumps(asyncio.run(asyncio.wait_for(client(args),60)),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
