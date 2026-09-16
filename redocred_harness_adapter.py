"""Index-preserving Re-DocRED adapter. Gold labels are never adapter inputs."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal
from pydantic import create_model
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from knowledge_agents.harness.adapters import OpenAIAdapter
from knowledge_agents.harness.contracts import Fact, ExtractOutput, CorrectOutput, validate_evidence
from knowledge_agents.harness.tools import ToolReply
from knowledge_agents.redocred_predictor import IndexedPrediction, IndexedRelation


class IndexedHarnessAdapter(OpenAIAdapter):
    def __init__(self,model,*,source,entities,relation_map,embedding_model='text-embedding-3-small',client=None):
        if not source or not entities or not relation_map:
            raise ValueError('Source, entity catalog and relation map are required')
        if any(not isinstance(x,str) or not x for x in entities) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in relation_map.items()):
            raise ValueError('Catalog entries must be strings')
        super().__init__(model,embedding_model,client=client)
        self.source=source
        self.entities=list(entities)
        self.relations=dict(relation_map)
        self.catalog={'entities':{f'E{i}':name for i,name in enumerate(entities)},'relations':self.relations}
        identity=json.dumps(dict(source=source,catalog=self.catalog),sort_keys=True)
        self.version+=':indexed-v1:'+hashlib.sha256(identity.encode()).hexdigest()
        entity_type=Literal.__getitem__(tuple(self.catalog['entities']))
        relation_type=Literal.__getitem__(tuple(sorted(self.relations)))
        self.fact_schema=create_model('IndexedHarnessFact',__base__=Fact,
            subject=(entity_type,...),object=(entity_type,...),predicate=(relation_type,...),
            fact_type=(Literal['relationship'],...),time=(Literal[None],None),location=(Literal[None],None))
        self.extract_schema=create_model('IndexedHarnessExtraction',__base__=ExtractOutput,facts=(list[self.fact_schema],...))
        self.correct_schema=create_model('IndexedHarnessCorrection',__base__=CorrectOutput,replacement=(self.fact_schema|None,...))

    def check_fact(self,fact):
        self.fact_schema.model_validate(fact.model_dump())
        if fact.quote not in self.source:
            raise ValueError('Quote is not bound to document')

    async def _parse(self,prompt,payload,schema):
        data=json.loads(payload)
        data['trusted_catalog']=self.catalog
        prompt+=(' Subject/object are entity IDs, predicates are relation IDs. Interpret them ONLY through trusted_catalog. '
                 'The candidate claim is its subject/predicate/object, NOT its supporting quote. '
                 'A quote denying the candidate relation does not support a positive candidate. '
                 'Catalog entries identify entities/relations; they do not establish that a relation occurred.')
        return await super()._parse(prompt,json.dumps(data,ensure_ascii=False),schema)

    async def extract(self,request):
        if request.source!=self.source:
            raise ValueError('Wrong document')
        if request.fact_type!='relationship':
            # Benchmark entities are supplied; non-relation stages have no model work.
            return ToolReply(ExtractOutput(facts=[]),0,0)
        reply=await self._parse('Extract document-supported relations using ONLY catalog IDs. Copy exact quotes. '
            'Respect direction and negation. Set fact_type=relationship and time/location=null; the benchmark scores relation triples.',
            request.model_dump_json(),self.extract_schema)
        output=ExtractOutput.model_validate(reply.data.model_dump())
        for fact in output.facts:self.check_fact(fact)
        return ToolReply(output,reply.input_tokens,reply.output_tokens,reply.context_metrics)

    async def retrieve(self,request):
        if request.source!=self.source:raise ValueError('Wrong document')
        self.check_fact(request.fact)
        f=request.fact
        translated=f.model_copy(update={'subject':self.catalog['entities'][f.subject],
            'object':self.catalog['entities'][f.object],'predicate':self.relations[f.predicate]})
        # Names improve retrieval; persisted candidates and verifier requests keep IDs.
        return await super().retrieve(request.model_copy(update={'fact':translated}))

    async def verify(self,request):
        self.check_fact(request.candidate.fact)
        validate_evidence(self.source,request.evidence)
        return await super().verify(request)

    async def correct(self,request):
        self.check_fact(request.candidate.fact)
        validate_evidence(self.source,request.evidence)
        reply=await self._parse('Correct this candidate only from supplied evidence. Keep subject/object IDs and fact_type unchanged. '
            'Use a catalog relation ID and exact quote; return replacement=null if no supported correction exists. '
            'A replacement must be reverified.',request.model_dump_json(),self.correct_schema)
        output=CorrectOutput.model_validate(reply.data.model_dump())
        if output.replacement:
            self.check_fact(output.replacement)
            if (output.replacement.subject,output.replacement.object)!=(request.candidate.fact.subject,request.candidate.fact.object):
                raise ValueError('Correction changed entity pair')
        return ToolReply(output,reply.input_tokens,reply.output_tokens,reply.context_metrics)


def export_indexed(state,adapter,sample_id):
    if state.status!='completed':raise ValueError('Incomplete session cannot be exported as completed prediction')
    if state.source!=adapter.source:raise ValueError('Wrong source')
    # Validate accepted decision consistency, without using business predicate projection.
    if set(state.accepted)&set(state.rejected) or len(set(state.accepted))!=len(state.accepted):
        raise ValueError('Inconsistent accepted set')
    relations=[]
    for cid in state.accepted:
        fact=state.candidates[cid].fact
        adapter.check_fact(fact)
        evidence=state.evidence.get(cid,[])
        validate_evidence(state.source,evidence)
        decisions=[d for d in state.decisions if d.candidate_id==cid]
        if not decisions or decisions[-1].verdict!='supported':raise ValueError('Accepted relation lacks supported decision')
        ids=decisions[-1].evidence_ids
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<={e.evidence_id for e in evidence}:
            raise ValueError('Invalid accepted citations')
        relations.append(IndexedRelation(head_index=int(fact.subject[1:]),tail_index=int(fact.object[1:]),
                                        relation_id=fact.predicate,evidence_quote=fact.quote))
    return IndexedPrediction(sample_id=sample_id,relations=relations)


async def self_test():
    import tempfile
    from types import SimpleNamespace
    from knowledge_agents.harness.adapters import registry_for
    from knowledge_agents.harness.runtime import Harness
    from knowledge_agents.harness.store import StateStore
    from knowledge_agents.harness.contracts import VerifyOutput,Decision,RetrieveOutput,EvidenceSpan
    source='A invested in B.'
    class MockAdapter(IndexedHarnessAdapter):
        async def _parse(self,prompt,payload,schema):
            data=json.loads(payload)
            if schema is self.extract_schema:
                return ToolReply(schema(facts=[self.fact_schema(subject='E0',predicate='P1',object='E1',fact_type='relationship',quote=source)]))
            if schema is self.correct_schema:
                return ToolReply(schema(replacement=self.fact_schema(subject='E0',predicate='P2',object='E1',fact_type='relationship',quote=source)))
            c=data['candidate']
            return ToolReply(VerifyOutput(decisions=[Decision(candidate_id=c['candidate_id'],
                verdict='supported' if c['fact']['predicate']=='P2' else 'needs_correction',
                reason='Controlled fixture',evidence_ids=['e'])]))
        async def retrieve(self,request):
            return ToolReply(RetrieveOutput(evidence=[EvidenceSpan(evidence_id='e',start=0,end=len(source),text=source)]))
    adapter=MockAdapter('offline',source=source,entities=['A','B'],relation_map={'P1':'acquired','P2':'invested in'},client=SimpleNamespace())
    for update in ({'subject':'E2'},{'predicate':'UNKNOWN'},{'fact_type':'entity'}):
        invalid=dict(subject='E0',predicate='P2',object='E1',fact_type='relationship',quote=source,**{})
        invalid.update(update)
        try:adapter.fact_schema.model_validate(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid index/type accepted')
    with tempfile.TemporaryDirectory() as temp:
        store=StateStore(Path(temp)/'state.sqlite')
        try:
            h=Harness(store,registry_for(adapter))
            state=h.create(source,'extract relationships','fixture')
            state=await h.run(state.session_id,'fixture')
            result=export_indexed(state,adapter,'fixture')
            assert [r.key() for r in result.relations]==[(0,1,'P2')]
            assert len(state.rejected)==1 and any(c.round==1 for c in state.candidates.values())
            state.evidence[state.accepted[0]][0].text='altered'
            try:export_indexed(state,adapter,'fixture')
            except ValueError:pass
            else:raise AssertionError('Altered evidence exported')
        finally:store.close()
    print('INDEXED ADAPTER OK: ID constraints, correction/reverification and export checked; 0 API calls.')

if __name__=='__main__':
    asyncio.run(self_test())
