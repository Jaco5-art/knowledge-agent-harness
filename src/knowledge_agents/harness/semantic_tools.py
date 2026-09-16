"""Scoped read-only semantic tools. Type validity is NOT evidence support."""
import asyncio
import hashlib
import json
from typing import Literal
from pydantic import Field, model_validator
from .contracts import Contract
from .tools import ToolRegistry, ToolSpec, ToolReply, PolicyDenied


class Entity(Contract):
    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    entity_type: str | None = None
    aliases: list[str] = Field(default_factory=list)


class RelationRule(Contract):
    subject_types: list[str] = Field(min_length=1)
    object_types: list[str] = Field(min_length=1)


class SemanticCatalog(Contract):
    tenant: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    entities: list[Entity]
    relations: dict[str, RelationRule] = Field(default_factory=dict)
    statuses: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def unique_ids(self):
        ids=[e.entity_id for e in self.entities]
        if len(ids)!=len(set(ids)):
            raise ValueError('Duplicate entity IDs')
        return self


class ScopedInput(Contract):
    tenant: str
    scope: str


class EntityInput(ScopedInput):
    mention: str


class EntityOutput(Contract):
    status: Literal['resolved','ambiguous','unknown']
    matches: list[Entity]


class StatusInput(ScopedInput):
    namespace: str
    raw_status: str


class StatusOutput(Contract):
    status: Literal['mapped','unknown']
    raw_status: str
    normalized_status: str | None = None
    namespace: str


class RelationInput(ScopedInput):
    predicate: str
    subject_id: str
    object_id: str


class RelationOutput(Contract):
    status: Literal['valid','invalid','unknown']
    reason: str
    evidence_verified: bool = False


class SemanticTools:
    def __init__(self, catalog: SemanticCatalog):
        # Snapshot configuration: later caller edits cannot change a registered version.
        self._catalog=catalog.model_copy(deep=True)
        self.version='semantic-tools-v1:'+hashlib.sha256(catalog.model_dump_json().encode()).hexdigest()

    def _authorize(self, request):
        if (request.tenant,request.scope)!=(self._catalog.tenant,self._catalog.scope):
            raise PolicyDenied('Semantic scope denied')

    async def lookup_entity_alias(self, request):
        self._authorize(request)
        matches=[e.model_copy(deep=True) for e in self._catalog.entities
                 if request.mention==e.name or request.mention in e.aliases]
        return ToolReply(EntityOutput(status='resolved' if len(matches)==1 else 'ambiguous' if matches else 'unknown', matches=matches))

    async def normalize_status(self, request):
        self._authorize(request)
        value=self._catalog.statuses.get(request.namespace,{}).get(request.raw_status)
        return ToolReply(StatusOutput(status='mapped' if value is not None else 'unknown',
            raw_status=request.raw_status,normalized_status=value,namespace=request.namespace))

    async def validate_relation_schema(self, request):
        self._authorize(request)
        entities={e.entity_id:e for e in self._catalog.entities}
        rule=self._catalog.relations.get(request.predicate)
        subject,object_=entities.get(request.subject_id),entities.get(request.object_id)
        if rule is None:
            output=RelationOutput(status='unknown',reason='unmapped_relation')
        elif subject is None or object_ is None:
            output=RelationOutput(status='unknown',reason='unresolved_entity_id')
        elif subject.entity_type is None or object_.entity_type is None:
            output=RelationOutput(status='unknown',reason='missing_entity_type')
        else:
            ok=subject.entity_type in rule.subject_types and object_.entity_type in rule.object_types
            output=RelationOutput(status='valid' if ok else 'invalid',reason='type_match' if ok else 'type_mismatch')
        return ToolReply(output)

    def register(self, registry):
        for name, ins, outs in [('lookup_entity_alias',EntityInput,EntityOutput),
                               ('normalize_status',StatusInput,StatusOutput),
                               ('validate_relation_schema',RelationInput,RelationOutput)]:
            registry.register(ToolSpec(name=name,version=self.version,input_model=ins,output_model=outs,
                handler=getattr(self,name),allowed_stages=frozenset({'semantic'})))


async def invoke(registry, name, payload, *, tenant, scope):
    """Trusted caller supplies scope; model/tool payload cannot select it."""
    if 'tenant' in payload or 'scope' in payload:
        raise PolicyDenied('Caller scope must not come from tool payload')
    spec=registry.get(name,'semantic')
    request=spec.input_model.model_validate(dict(payload,tenant=tenant,scope=scope))
    reply=await spec.handler(request)
    return spec.output_model.model_validate(reply.data)


def demo_catalog():
    return SemanticCatalog(tenant='demo',scope='synthetic-business',revision='1',entities=[
        Entity(entity_id='a',name='Aster Ltd',entity_type='Company',aliases=['ASTER']),
        Entity(entity_id='b',name='Birch Ltd',entity_type='Company',aliases=['BIRCH']),
        Entity(entity_id='c',name='Springfield North',entity_type='City',aliases=['Springfield']),
        Entity(entity_id='d',name='Springfield South',entity_type='City',aliases=['Springfield'])],
        relations={'acquired_by':RelationRule(subject_types=['Company'],object_types=['Company'])},
        statuses={'ticket-system':{'CLOSED':'closed'},'job-system':{'CLOSED':'completed','WAITING':'pending'}})


async def demo():
    registry=ToolRegistry()
    SemanticTools(demo_catalog()).register(registry)
    requests=[('lookup_entity_alias',{'mention':'ASTER'}),('lookup_entity_alias',{'mention':'Springfield'}),
        ('normalize_status',{'namespace':'ticket-system','raw_status':'CLOSED'}),
        ('normalize_status',{'namespace':'job-system','raw_status':'CLOSED'}),
        ('validate_relation_schema',{'predicate':'acquired_by','subject_id':'a','object_id':'b'}),
        ('validate_relation_schema',{'predicate':'acquired_by','subject_id':'c','object_id':'b'})]
    return [dict(tool=name,result=(await invoke(registry,name,payload,tenant='demo',scope='synthetic-business')).model_dump()) for name,payload in requests]


if __name__=='__main__':
    print(json.dumps(asyncio.run(demo()),ensure_ascii=False,indent=2))
