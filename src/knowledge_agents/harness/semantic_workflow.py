"""Opt-in verified-output gate. Never changes upstream acceptance or evidence."""
import argparse
import asyncio
import copy
import json
import tempfile
import time
from pathlib import Path
from .contracts import validate_evidence
from .semantic import from_session, read_session
from .semantic_tools import SemanticCatalog, SemanticTools, invoke, Entity, RelationRule
from .tools import ToolRegistry, PolicyDenied


class SemanticGate:
    def __init__(self,catalog,*,timeout=5,history_tool=None):
        if timeout<=0:
            raise ValueError('Timeout must be positive')
        self.catalog=catalog.model_copy(deep=True)
        self.timeout=timeout
        self.registry=ToolRegistry()
        self.tools=SemanticTools(self.catalog)
        self.tools.register(self.registry)
        self.history_tool = history_tool
        self.history_registry = ToolRegistry()
        if history_tool is not None:
            if (history_tool.tenant, history_tool.scope) != (
                self.catalog.tenant, self.catalog.scope
            ):
                raise PolicyDenied('History tool scope does not match catalog')
            history_tool.register(self.history_registry)

    def authorize(self,state):
        if (state.tenant_id,state.source_hash)!=(self.catalog.tenant,self.catalog.scope):
            raise PolicyDenied('Catalog does not match tenant and document scope')

    async def apply(self,state):
        self.authorize(state)
        result=dict(version='semantic-gate-v1',session_id=state.session_id,
            state_version=state.version,scope=state.source_hash,tenant=state.tenant_id,
            catalog_version=self.tools.version,catalog=self.catalog.model_dump(),
            upstream_status=state.status,ready_facts=[],needs_review=[],trace=[],
            status='upstream_incomplete',evidence_reverified=False)
        if state.status!='completed':
            return await self.attach_history(result)
        view=from_session(state)
        # Structural grounding rechecked locally; not a second LLM entailment check.
        for cid in state.accepted:
            evidence=state.evidence.get(cid,[])
            validate_evidence(state.source,evidence)
            decisions=[d for d in state.decisions if d.candidate_id==cid]
            if not decisions or decisions[-1].verdict!='supported':
                raise ValueError('Latest decision must support accepted candidate')
            ids=decisions[-1].evidence_ids
            if not ids or len(ids)!=len(set(ids)) or not set(ids)<={e.evidence_id for e in evidence}:
                raise ValueError('Invalid accepted citations')
            if state.candidates[cid].fact.quote not in state.source:
                raise ValueError('Unbound original quote')
        result['source_view']=view
        result['needs_review']=copy.deepcopy(view['needs_review'])
        async def call(name,payload,view_id):
            started=time.perf_counter()
            trace=dict(tool=name,view_id=view_id,version=self.tools.version,input=payload)
            try:
                output=await asyncio.wait_for(invoke(self.registry,name,payload,
                    tenant=state.tenant_id,scope=state.source_hash),self.timeout)
                trace.update(status='succeeded',output=output.model_dump())
                return output
            except Exception as exc:
                trace.update(status='failed',error_type=type(exc).__name__)
                raise
            finally:
                trace['latency_ms']=(time.perf_counter()-started)*1000
                result['trace'].append(trace)
        for fact in view['business_facts']:
            vid=fact['view_id']
            reason=None
            try:
                subject=await call('lookup_entity_alias',{'mention':fact['subject']},vid)
                obj=await call('lookup_entity_alias',{'mention':fact['object']},vid)
                if subject.status!='resolved' or obj.status!='resolved':
                    reason='entity_unresolved_or_ambiguous'
                else:
                    check=await call('validate_relation_schema',dict(predicate=fact['predicate'],
                        subject_id=subject.matches[0].entity_id,object_id=obj.matches[0].entity_id),vid)
                    if check.status!='valid':
                        reason=check.reason
                    else:
                        # Enrich with IDs but do not rewrite names or generate a new claim.
                        result['ready_facts'].append(dict(fact,entity_type_validation='valid',
                            subject_entity_id=subject.matches[0].entity_id,
                            object_entity_id=obj.matches[0].entity_id,
                            identity_basis='trusted_scoped_catalog',evidence_reverified=False))
            except Exception as exc:
                reason='semantic_tool_error:'+type(exc).__name__
            if reason:
                result['needs_review'].append(dict(view_id=vid,candidate_ids=fact['candidate_ids'],reason=reason))
        result['status']='needs_review' if result['needs_review'] else 'completed'
        return await self.attach_history(result)

    # HISTORY_INTEGRATION_V1
    async def attach_history(self, result):
        from .history_query import invoke_history

        result["history_results"] = []
        if self.history_tool is None:
            result["history_status"] = "disabled"
            return result
        if result["status"] != "completed":
            result["history_status"] = "not_run"
            return result

        result["history_status"] = "completed"
        for fact in result["ready_facts"]:
            payload = dict(
                subject_id=fact["subject_entity_id"],
                predicate=fact["predicate"],
                object_id=fact["object_entity_id"],
                limit=10,
            )
            entry = dict(
                view_id=fact["view_id"],
                evidence_verified=False,
                usage="reference_only",
            )
            trace = dict(
                tool="query_relation_history",
                view_id=fact["view_id"],
                version=self.history_tool.version,
                input=payload.copy(),
            )
            started = time.perf_counter()
            try:
                output = await invoke_history(
                    self.history_registry, payload,
                    tenant=result["tenant"], scope=result["scope"],
                )
                entry.update(status="succeeded", result=output.model_dump())
                trace.update(status="succeeded", rows=len(output.records))
            except asyncio.CancelledError:
                trace.update(status="cancelled")
                raise
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__)
                trace.update(status="failed", error_type=type(exc).__name__)
                result["history_status"] = "partial_failure"
            finally:
                trace["latency_ms"] = (time.perf_counter()-started)*1000
                result["trace"].append(trace)
            result["history_results"].append(entry)
        return result


class SemanticWorkflow:
    """New explicit entry: Harness.run -> completed-state check -> semantic gate.

    Upstream durable checkpoints remain in Harness; the gate is recomputable
    read-only work, not a new checkpointed runtime stage or approval queue.
    """
    def __init__(self,harness,catalog,*,history_tool=None):
        self.harness=harness
        self.gate=SemanticGate(catalog,history_tool=history_tool)

    async def run(self,session_id,tenant,**kwargs):
        before=self.harness.store.load(session_id,tenant)
        self.gate.authorize(before)  # Validate scope before any paid upstream call.
        state=await self.harness.run(session_id,tenant,**kwargs)
        return await self.gate.apply(state)


def fixture_catalog(state):
    return SemanticCatalog(tenant=state.tenant_id,scope=state.source_hash,revision='fixture-1',
        entities=[Entity(entity_id='ms',name='Microsoft',entity_type='Company'),
                  Entity(entity_id='oa',name='OpenAI',entity_type='Company'),
                  Entity(entity_id='sf',name='San Francisco',entity_type='City')],
        relations={'invested_in':RelationRule(subject_types=['Company'],object_types=['Company']),
                   'headquartered_in':RelationRule(subject_types=['Company'],object_types=['City'])})


async def demo():
    from .runtime import Harness
    from .store import StateStore
    from .adapters import FixtureAdapter,registry_for,DEMO_SOURCE
    with tempfile.TemporaryDirectory() as temp:
        store=StateStore(Path(temp)/'demo.sqlite')
        try:
            harness=Harness(store,registry_for(FixtureAdapter(False)))
            state=harness.create(DEMO_SOURCE,'extract entities and relationships','demo')
            return await SemanticWorkflow(harness,fixture_catalog(state)).run(state.session_id,'demo')
        finally:
            store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--db',type=Path)
    parser.add_argument('--session')
    parser.add_argument('--tenant',default='local')
    parser.add_argument('--catalog',type=Path)
    parser.add_argument('--output',type=Path)
    # HISTORY_CLI_V1
    parser.add_argument('--history-db', type=Path,
                        help='Optional read-only relationship history database')
    parser.add_argument('--history-revision',
                        help='Operator-supplied history dataset revision')
    args=parser.parse_args()
    if (args.history_db is None) != (args.history_revision is None):
        parser.error('--history-db and --history-revision must be supplied together')
    if args.history_revision is not None and not args.history_revision.strip():
        parser.error('--history-revision cannot be blank')
    if args.demo and args.history_db is not None:
        parser.error('--demo cannot use an external history database')
    if args.demo:
        if args.db or args.session or args.catalog:
            parser.error('Demo cannot use external session/catalog')
        result=asyncio.run(demo())
    else:
        if not (args.db and args.session and args.catalog):
            parser.error('Provide --demo OR --db --session --catalog')
        state=read_session(args.db,args.session,args.tenant)
        catalog=SemanticCatalog.model_validate_json(args.catalog.read_text(encoding='utf-8-sig'))
        history_tool = None
        if args.history_db is not None:
            from .history_query import HistoryTool
            history_tool = HistoryTool(
                args.history_db,
                tenant=args.tenant,
                scope=state.source_hash,
                dataset_revision=args.history_revision,
            )
        result=asyncio.run(
            SemanticGate(catalog, history_tool=history_tool).apply(state)
        )
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(text)
    else:
        print(text)


if __name__=='__main__':
    main()
