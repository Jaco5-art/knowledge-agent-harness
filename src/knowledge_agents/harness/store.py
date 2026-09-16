"""SQLite snapshots and events committed together; optimistic concurrency guard."""
from __future__ import annotations

import json
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contracts import Session


class ConflictError(RuntimeError):
    pass


class StateStore:
    def __init__(self, path: str | Path):
        self.lock_directory = Path(str(path) + ".locks")
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, tenant TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                version INTEGER NOT NULL, body TEXT NOT NULL
            );
        """)

    def close(self):
        self.connection.close()

    @contextmanager
    def session_lock(self, session_id: str):
        """Local OS lock, released by the OS on process death; not a cluster lock."""
        import os
        name = hashlib.sha256(session_id.encode()).hexdigest()
        with (self.lock_directory / name).open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ConflictError("Session is already being processed") from exc
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def create(self, state: Session):
        with self.connection:
            self.connection.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (state.session_id, state.tenant_id, state.version, state.model_dump_json()))
            self._event(state, {"event": "session_created"})

    def load(self, session_id: str, tenant_id: str) -> Session:
        row = self.connection.execute("SELECT body FROM sessions WHERE id=? AND tenant=?",
                                      (session_id, tenant_id)).fetchone()
        if row is None:
            raise KeyError("Session not found in tenant scope")
        return Session.model_validate_json(row[0])

    def _event(self, state: Session, event: dict):
        body = {"timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": state.session_id, "state_version": state.version, **event}
        self.connection.execute("INSERT INTO events(session_id, version, body) VALUES (?, ?, ?)",
                                (state.session_id, state.version, json.dumps(body, ensure_ascii=False)))

    def save(self, state: Session, event: dict):
        old_version = state.version
        state.version += 1
        try:
            with self.connection:
                updated = self.connection.execute(
                    "UPDATE sessions SET version=?, body=? WHERE id=? AND tenant=? AND version=?",
                    (state.version, state.model_dump_json(), state.session_id, state.tenant_id, old_version))
                if updated.rowcount != 1:
                    raise ConflictError("Stale state; reload before proceeding")
                self._event(state, event)
        except BaseException:
            state.version = old_version
            raise

    def events(self, session_id: str, tenant_id: str) -> list[dict]:
        self.load(session_id, tenant_id)
        rows = self.connection.execute("SELECT body FROM events WHERE session_id=? ORDER BY id",
                                       (session_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def export_trace(self, session_id: str, tenant_id: str, path: str | Path):
        rows = self.events(session_id, tenant_id)
        Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
