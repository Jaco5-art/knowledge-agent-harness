"""Durable local semantic publication review; never bypass upstream verification."""
import argparse
import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .contracts import Session
from .semantic import read_session
from .semantic_tools import SemanticCatalog
from .semantic_workflow import SemanticGate
from .store import ConflictError


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def substantive_catalog(catalog):
    return catalog.model_dump(exclude={'revision'})


class ReviewStore:
    def __init__(self,path):
        path=Path(path)
        self.connection=sqlite3.connect(path)
        try:
            names={r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if names and names!={'semantic_records','semantic_history'}:
                raise ValueError('Use a separate semantic review database, not the upstream database')
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS semantic_records (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, version INTEGER NOT NULL,
                    snapshot TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS semantic_history (
                    review_id TEXT NOT NULL, version INTEGER NOT NULL, event TEXT NOT NULL,
                    PRIMARY KEY(review_id,version));
            ''')
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def _load(self,review_id,tenant):
        row=self.connection.execute('SELECT snapshot,body FROM semantic_records WHERE id=? AND tenant=?',
                                    (review_id,tenant)).fetchone()
        if row is None:
            raise ValueError('Review not found for tenant')
        state=Session.model_validate_json(row[0])
        body=json.loads(row[1])
        if state.tenant_id!=tenant or digest(state.model_dump())!=body['snapshot_hash']:
            raise ValueError('Stored snapshot integrity mismatch')
        return state,body

    def show(self,review_id,tenant):
        return self._load(review_id,tenant)[1]

    def history(self,review_id,tenant):
        self._load(review_id,tenant)
        return [json.loads(r[0]) for r in self.connection.execute(
            'SELECT event FROM semantic_history WHERE review_id=? ORDER BY version',(review_id,))]

    def _commit(self,state,body,expected,actor,action):
        event=dict(review_id=body['review_id'],version=body['version'],actor=actor,action=action,
                   timestamp=datetime.now(timezone.utc).isoformat(),report=body['report'],status=body['status'])
        try:
            self.connection.execute('BEGIN IMMEDIATE')
            if expected is None:
                self.connection.execute('INSERT INTO semantic_records VALUES (?,?,?,?,?)',
                    (body['review_id'],state.tenant_id,body['version'],state.model_dump_json(),json.dumps(body)))
            else:
                cursor=self.connection.execute('UPDATE semantic_records SET version=?,body=? WHERE id=? AND tenant=? AND version=?',
                    (body['version'],json.dumps(body),body['review_id'],state.tenant_id,expected))
                if cursor.rowcount!=1:
                    raise ConflictError('Review version changed')
            self.connection.execute('INSERT INTO semantic_history VALUES (?,?,?)',
                (body['review_id'],body['version'],json.dumps(event)))
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    async def create(self,state,catalog,actor):
        if not actor.strip(): raise ValueError('Actor is required')
        if state.status!='completed': raise ValueError('Upstream must be completed')
        # Immutable snapshot: no source changes during an awaited gate call.
        state=Session.model_validate_json(state.model_dump_json())
        catalog=catalog.model_copy(deep=True)
        SemanticGate(catalog).authorize(state)
        snapshot_hash=digest(state.model_dump())
        rid=digest([snapshot_hash,catalog.model_dump()])
        existing=self.connection.execute('SELECT id FROM semantic_records WHERE id=? AND tenant=?',(rid,state.tenant_id)).fetchone()
        if existing: return self.show(rid,state.tenant_id)
        report=await SemanticGate(catalog).apply(state)
        body=dict(review_id=rid,version=0,snapshot_hash=snapshot_hash,status=report['status'],
                  retry_count=0,max_retries=3,report=report)
        try:
            self._commit(state,body,None,actor,'create')
        except sqlite3.IntegrityError:
            return self.show(rid,state.tenant_id)
        return body

    async def resolve(self,review_id,tenant,expected_version,actor,action,catalog=None):
        if not actor.strip(): raise ValueError('Actor is required')
        if action not in {'retry','reject','cancel'}:
            raise ValueError('Only retry/reject/cancel; no accept bypass')
        state,body=self._load(review_id,tenant)
        if body['version']!=expected_version: raise ConflictError('Review version changed')
        if body['status']!='needs_review': raise ValueError('Only pending reviews can be resolved')
        if action=='retry':
            if catalog is None: raise ValueError('Retry requires a revised catalog')
            catalog=catalog.model_copy(deep=True)
            old=SemanticCatalog.model_validate(body['report']['catalog'])
            if substantive_catalog(old)==substantive_catalog(catalog):
                raise ValueError('No-progress retry: rules unchanged')
            if body['retry_count']>=body['max_retries']: raise ValueError('Semantic retry budget exhausted')
            report=await SemanticGate(catalog).apply(state)
            body.update(report=report,status=report['status'],retry_count=body['retry_count']+1)
        else:
            if catalog is not None: raise ValueError('Catalog only allowed for retry')
            body['status']='rejected' if action=='reject' else 'cancelled'
        body['version']+=1
        self._commit(state,body,expected_version,actor,action)
        return body

    def publication(self,review_id,tenant):
        body=self.show(review_id,tenant)
        # Whole-run fail-closed: no partial export while anything needs review.
        return dict(review_id=review_id,version=body['version'],status=body['status'],
                    facts=body['report']['ready_facts'] if body['status']=='completed' else [])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-db',type=Path,required=True)
    parser.add_argument('--tenant',default='local')
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create')
    create.add_argument('--db',type=Path,required=True)
    create.add_argument('--session',required=True)
    create.add_argument('--catalog',type=Path,required=True)
    create.add_argument('--actor',required=True)
    show=sub.add_parser('show'); show.add_argument('--review-id',required=True)
    resolve=sub.add_parser('resolve')
    resolve.add_argument('--review-id',required=True)
    resolve.add_argument('--expected-version',type=int,required=True)
    resolve.add_argument('--actor',required=True)
    resolve.add_argument('--action',choices=['retry','reject','cancel'],required=True)
    resolve.add_argument('--catalog',type=Path)
    args=parser.parse_args()
    if args.command=='create' and args.review_db.resolve()==args.db.resolve():
        parser.error('Review database must differ from upstream database')
    if args.command!='create' and not args.review_db.is_file():
        parser.error('Review database does not exist')
    store=ReviewStore(args.review_db)
    try:
        if args.command=='create':
            state=read_session(args.db,args.session,args.tenant)
            catalog=SemanticCatalog.model_validate_json(args.catalog.read_text(encoding='utf-8-sig'))
            body=asyncio.run(store.create(state,catalog,args.actor))
        elif args.command=='resolve':
            catalog=SemanticCatalog.model_validate_json(args.catalog.read_text(encoding='utf-8-sig')) if args.catalog else None
            body=asyncio.run(store.resolve(args.review_id,args.tenant,args.expected_version,args.actor,args.action,catalog))
        else:
            body=store.show(args.review_id,args.tenant)
        print(json.dumps(dict(review=body,publication=store.publication(body['review_id'],args.tenant)),ensure_ascii=False,indent=2))
    finally:
        store.close()


if __name__=='__main__': main()
