"""Conservative post-verification projection, NOT a new verification decision."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from .contracts import Candidate, Fact, Session

VERSION = 'semantic-view-v1:signed-partnership-v1'
PREDICATES = {
    'headquartered in': 'headquartered_in', 'is headquartered in': 'headquartered_in',
    'entered a strategic partnership with': 'strategic_partnership_with',
    'entered strategic partnership with': 'strategic_partnership_with',
    'invested in': 'invested_in', 'acquired': 'acquired', 'acquired by': 'acquired_by',
}

# SIGNED_PARTNERSHIP_MAPPING_V1
PREDICATES["signed strategic partnership with"] = "strategic_partnership_with"


def normalize_predicate(value):
    return ' '.join(value.replace('_', ' ').casefold().split())


def project(candidates, *, scope, aliases=None):
    """Caller supplies already accepted candidates from one source scope.

    Alias entries are explicit exact-name -> list of canonical names, scoped by
    the caller. No fuzzy matching, transitive resolution, or global defaults.
    Ambiguous entries never merge. A display projection does not re-verify aliases.
    """
    if not scope:
        raise ValueError('Source scope required')
    aliases = aliases or {}
    if any(not isinstance(k,str) or not isinstance(v,list) or not v or
           any(not isinstance(x,str) or not x.strip() for x in v) for k,v in aliases.items()):
        raise ValueError('Alias entries must be nonempty lists of canonical names')
    for values in aliases.values():
        for value in values:
            if value in aliases and set(aliases[value]) != {value}:
                raise ValueError('Alias chains/cycles require explicit resolution')
    raw = [c.model_dump() for c in candidates]
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate IDs')
    groups={}
    result=dict(version=VERSION,scope=scope,business_facts=[],entity_mentions=[],
                document_statements=[],needs_review=[],original_candidates=raw,
                alias_rules=aliases,semantic_verification_performed=False)
    for c in candidates:
        f=c.fact
        predicate=normalize_predicate(f.predicate)
        if predicate=='mentioned in' and f.object=='document':
            result['entity_mentions'].append(c.model_dump())
            continue
        if f.subject=='document' and predicate=='does not state acquisition of':
            result['document_statements'].append(c.model_dump())
            continue
        canonical=PREDICATES.get(predicate)
        if canonical is None:
            result['needs_review'].append(dict(candidate_id=c.candidate_id,reason='unmapped_predicate'))
            continue
        resolved=[]
        ambiguity=False
        for value in (f.subject,f.object):
            options=sorted(set(aliases.get(value,[value]))) if value is not None else []
            if len(options)!=1:
                ambiguity=True
            resolved.append(options[0] if len(options)==1 else None)
        if ambiguity:
            result['needs_review'].append(dict(candidate_id=c.candidate_id,reason='ambiguous_or_missing_entity'))
            continue
        claim=dict(subject=resolved[0],predicate=canonical,object=resolved[1],
                   time=f.time,location=f.location,polarity='positive')
        key=json.dumps([scope,claim],sort_keys=True,ensure_ascii=False)
        if key not in groups:
            groups[key]=dict(view_id=hashlib.sha256(key.encode()).hexdigest(),**claim,
                candidate_ids=[],source_agents=[],fact_types=[],quotes=[],
                entity_type_validation='not_checked',alias_applied=False)
        group=groups[key]
        group['candidate_ids'].append(c.candidate_id)
        group['alias_applied'] |= resolved != [f.subject,f.object]
        for field,value in [('source_agents',f.source_agent),('fact_types',f.fact_type),('quotes',f.quote)]:
            if value not in group[field]:
                group[field].append(value)
    result['business_facts']=list(groups.values())
    result['merged_records']=sum(len(g['candidate_ids'])-1 for g in groups.values())
    return result


def from_session(session):
    supported={d.candidate_id for d in session.decisions if d.verdict=='supported'}
    if set(session.accepted)&set(session.rejected) or not set(session.accepted)<=supported:
        raise ValueError('Accepted candidates lack consistent supported decisions')
    if len(set(session.accepted))!=len(session.accepted):
        raise ValueError('Duplicate accepted IDs')
    view=project([session.candidates[c] for c in session.accepted],scope=session.source_hash)
    view.update(session_id=session.session_id,state_version=session.version,tenant_id=session.tenant_id)
    return view


def read_session(db, session_id, tenant):
    # Read-only connection: never initialize schemas or create a missing DB.
    path=Path(db).resolve(strict=True)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
        row=connection.execute('SELECT body FROM sessions WHERE id=? AND tenant=?',(session_id,tenant)).fetchone()
    if row is None:
        raise ValueError('Session not found for tenant')
    session=Session.model_validate_json(row[0])
    if session.session_id!=session_id or session.tenant_id!=tenant:
        raise ValueError('Session identity mismatch')
    return session


def demo():
    def c(i,p,kind='relationship',subject='OpenAI',obj='San Francisco'):
        return Candidate(candidate_id=str(i),fact=Fact(subject=subject,predicate=p,object=obj,
            fact_type=kind,quote='Synthetic demo only.',source_agent=kind+'_agent'))
    return project([c(1,'headquartered in','entity'),c(2,'headquartered_in'),
        c(3,'mentioned in','entity','Microsoft','document'),
        c(4,'does_not_state_acquisition_of','relationship','document','Microsoft acquired OpenAI'),
        c(5,'might acquire','relationship','Microsoft','OpenAI')],scope='synthetic-demo')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--db',type=Path)
    parser.add_argument('--session')
    parser.add_argument('--tenant',default='local')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.demo and (args.db or args.session):
        parser.error('Choose demo OR a database session')
    if not args.demo and not (args.db and args.session):
        parser.error('Provide --demo or --db with --session')
    view=demo() if args.demo else from_session(read_session(args.db,args.session,args.tenant))
    text=json.dumps(view,ensure_ascii=False,indent=2)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(text)
    else:
        print(text)


if __name__=='__main__':
    main()
