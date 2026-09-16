"""Unified offline session/review/export CLI. No provider calls."""
import argparse
import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from .semantic import read_session
from .semantic_review import ReviewStore
from .semantic_tools import SemanticCatalog
from .semantic_workflow import fixture_catalog
from .store import ConflictError


class ReadOnlyReviews(ReviewStore):
    def __init__(self,path):
        self.connection=sqlite3.connect(Path(path).resolve(strict=True).as_uri()+'?mode=ro',uri=True)


def brief(body):
    report=body['report']
    source=report.get('source_view',{})
    return dict(review_id=body['review_id'],version=body['version'],status=body['status'],
        session_id=report['session_id'],source_state_version=report['state_version'],
        upstream_accepted=len(source.get('original_candidates',[])),
        gate_ready=len(report['ready_facts']),needs_review=report['needs_review'],
        publishable=len(report['ready_facts']) if body['status']=='completed' else 0,
        retry_count=body['retry_count'],catalog_version=report['catalog_version'])


def list_sessions(db,tenant):
    with sqlite3.connect(Path(db).resolve(strict=True).as_uri()+'?mode=ro',uri=True) as conn:
        rows=conn.execute('SELECT id,version,body FROM sessions WHERE tenant=? ORDER BY id',(tenant,)).fetchall()
    return [dict(session_id=i,version=v,status=json.loads(b)['status'],
                 accepted_count=len(json.loads(b)['accepted'])) for i,v,b in rows]


def export_publication(store,review_id,tenant,expected_version,output):
    body=store.show(review_id,tenant)
    if body['version']!=expected_version:
        raise ConflictError('Review version changed; inspect before exporting')
    if body['status']!='completed':
        raise ValueError('Publication blocked: review is not completed')
    report=body['report']
    publication=dict(format='knowledge-publication-v1',review_id=review_id,version=body['version'],
        session_id=report['session_id'],source_state_version=report['state_version'],
        scope=report['scope'],tenant=tenant,catalog_version=report['catalog_version'],
        snapshot_hash=body['snapshot_hash'],facts=report['ready_facts'],
        note='Upstream-supported and catalog-consistent; no new evidence verification.')
    # Serialize first. Exclusive creation prevents overwriting databases/results.
    text=json.dumps(publication,ensure_ascii=False,indent=2)
    with Path(output).open('x',encoding='utf-8') as stream:
        stream.write(text)
    return dict(output=str(output),fact_count=len(publication['facts']),review_id=review_id,version=body['version'])


async def stage_demo():
    from .runtime import Harness
    from .store import StateStore
    from .adapters import FixtureAdapter,registry_for,DEMO_SOURCE
    with tempfile.TemporaryDirectory(prefix='semantic-stage-demo-') as temp:
        path=Path(temp)
        upstream=StateStore(path/'upstream.sqlite')
        try:
            h=Harness(upstream,registry_for(FixtureAdapter(False)))
            state=h.create(DEMO_SOURCE,'extract entities and relationships','demo')
            state=await h.run(state.session_id,'demo')
        finally: upstream.close()
        original=(path/'upstream.sqlite').read_bytes()
        good=fixture_catalog(state)
        bad=good.model_copy(deep=True); bad.entities[1].entity_type=None
        store=ReviewStore(path/'reviews.sqlite')
        try: pending=await store.create(state,bad,'demo-operator')
        finally: store.close()
        store=ReviewStore(path/'reviews.sqlite')
        try:
            try:
                export_publication(store,pending['review_id'],'demo',0,path/'blocked.json')
            except ValueError: blocked=True
            else: blocked=False
            completed=await store.resolve(pending['review_id'],'demo',0,'demo-operator','retry',good)
            exported=export_publication(store,completed['review_id'],'demo',1,path/'publication.json')
            history=store.history(completed['review_id'],'demo')
        finally: store.close()
        unchanged=original==(path/'upstream.sqlite').read_bytes()
        if not blocked or not unchanged or exported['fact_count']!=2:
            raise AssertionError('Stage acceptance failed')
        return dict(status='passed',mode='deterministic_fixture',provider_calls=0,
            pending_publication_blocked=blocked,reopened_database=True,
            history_actions=[e['action'] for e in history],final_status=completed['status'],
            published_fact_count=exported['fact_count'],upstream_unchanged=unchanged,
            persistent_user_files_created=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-db',type=Path,default=Path('semantic_reviews.sqlite'))
    p.add_argument('--tenant',default='local')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('demo')
    sessions=sub.add_parser('sessions'); sessions.add_argument('--db',type=Path,required=True)
    sub.add_parser('reviews')
    check=sub.add_parser('check')
    check.add_argument('--db',type=Path,required=True); check.add_argument('--session',required=True)
    check.add_argument('--catalog',type=Path,required=True); check.add_argument('--actor',required=True)
    for name in ('show','history','retry','reject','cancel','export'):
        command=sub.add_parser(name); command.add_argument('--review-id',required=True)
        if name in ('retry','reject','cancel','export'):
            command.add_argument('--expected-version',type=int,required=True)
        if name in ('retry','reject','cancel'): command.add_argument('--actor',required=True)
        if name=='retry': command.add_argument('--catalog',type=Path,required=True)
        if name=='export': command.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='demo': result=asyncio.run(stage_demo())
    elif args.command=='sessions': result=list_sessions(args.db,args.tenant)
    else:
        if args.command=='check':
            if args.db.resolve()==args.review_db.resolve(): p.error('Use a separate review database')
            state=read_session(args.db,args.session,args.tenant)
            catalog=SemanticCatalog.model_validate_json(args.catalog.read_text(encoding='utf-8-sig'))
            if not args.actor.strip(): p.error('Actor required')
            from .semantic_workflow import SemanticGate
            SemanticGate(catalog).authorize(state)
            if state.status!='completed': p.error('Upstream session must be completed')
            store=ReviewStore(args.review_db)
        else:
            if not args.review_db.is_file(): p.error('Review database does not exist; use check first')
            store=ReviewStore(args.review_db) if args.command in ('retry','reject','cancel') else ReadOnlyReviews(args.review_db)
        try:
            if args.command=='check': result=brief(asyncio.run(store.create(state,catalog,args.actor)))
            elif args.command=='reviews':
                ids=[r[0] for r in store.connection.execute('SELECT id FROM semantic_records WHERE tenant=? ORDER BY id',(args.tenant,))]
                result=[brief(store.show(i,args.tenant)) for i in ids]
            elif args.command=='show': result=brief(store.show(args.review_id,args.tenant))
            elif args.command=='history': result=store.history(args.review_id,args.tenant)
            elif args.command=='export': result=export_publication(store,args.review_id,args.tenant,args.expected_version,args.output)
            else:
                catalog=SemanticCatalog.model_validate_json(args.catalog.read_text(encoding='utf-8-sig')) if args.command=='retry' else None
                result=brief(asyncio.run(store.resolve(args.review_id,args.tenant,args.expected_version,args.actor,args.command,catalog)))
        finally: store.close()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__': main()
