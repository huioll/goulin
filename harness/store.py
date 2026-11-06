# -*- coding: utf-8 -*-
"""SQLite 存储层：短期记忆 / 任务状态 / 长期记忆 / trace 四张主表。

用 SQLite 而不是外部服务，是为了让整套记忆体系"零依赖可跑"；
向量部分见 embedding.py + long_term.py（SQLite 存 BLOB，暴力余弦检索）。
如果需要换 Chroma/Milvus/pgvector，只要替换 LongTermMemory 里的 retrieve 实现即可。
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS short_memory (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_id   TEXT,
    turn      INTEGER NOT NULL DEFAULT 0,
    step      INTEGER NOT NULL DEFAULT 0,
    kind      TEXT NOT NULL,              -- user/thought/action/observation/answer/summary/note
    content   TEXT NOT NULL,
    protected INTEGER NOT NULL DEFAULT 0, -- 1 = 关键决策/证据，降级时优先保留
    meta      TEXT,                       -- 结构化附加信息（工具名、成功与否、证据 id...）
    tokens    INTEGER NOT NULL DEFAULT 0,
    ts        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stm_session ON short_memory(session_id, id);

CREATE TABLE IF NOT EXISTS task_state (
    task_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    goal       TEXT NOT NULL,
    status     TEXT NOT NULL,             -- running/done/needs_recheck
    version    INTEGER NOT NULL DEFAULT 1,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_state_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    ts      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tsh_task ON task_state_history(task_id, version);

CREATE TABLE IF NOT EXISTS long_memory (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    text          TEXT NOT NULL,
    kind          TEXT NOT NULL,          -- preference/fact/conclusion/schema
    source        TEXT NOT NULL,          -- user/agent/tool/seed
    task_id       TEXT,
    session_id    TEXT,
    tags          TEXT,                   -- JSON 数组
    confidence    REAL NOT NULL DEFAULT 0.8,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    valid_until   TEXT,                   -- NULL = 永久
    superseded_by INTEGER,                -- 被哪条新记忆取代（矛盾抑制）
    content_hash  TEXT,
    dim           INTEGER NOT NULL DEFAULT 0,
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS idx_ltm_kind ON long_memory(kind);
CREATE INDEX IF NOT EXISTS idx_ltm_hash ON long_memory(content_hash);

CREATE TABLE IF NOT EXISTS trace (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    session_id TEXT,
    task_id   TEXT,
    step      INTEGER,
    event     TEXT NOT NULL,
    detail    TEXT,
    duration_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_trace_task ON trace(task_id, id);
CREATE INDEX IF NOT EXISTS idx_trace_event ON trace(event);
"""


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def iso_after_days(days: int) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


class Store:
    def __init__(self, path: str = None):
        self.path = path or config.DB_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self.conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ 短期记忆
    def stm_add(self, session_id, task_id, turn, step, kind, content,
                protected=False, meta=None, tokens=0) -> int:
        with self.conn() as conn:
            cur = conn.execute(
                "INSERT INTO short_memory (session_id, task_id, turn, step, kind, content,"
                " protected, meta, tokens, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (session_id, task_id, turn, step, kind, content, 1 if protected else 0,
                 json.dumps(meta or {}, ensure_ascii=False), tokens, now_iso()),
            )
            return int(cur.lastrowid)

    def stm_list(self, session_id: str, limit: int = None):
        sql = "SELECT * FROM short_memory WHERE session_id=? ORDER BY id ASC"
        args = [session_id]
        if limit:
            sql = ("SELECT * FROM (SELECT * FROM short_memory WHERE session_id=?"
                   " ORDER BY id DESC LIMIT ?) ORDER BY id ASC")
            args = [session_id, limit]
        with self.conn() as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

    def stm_clear(self, session_id: str) -> int:
        with self.conn() as conn:
            cur = conn.execute("DELETE FROM short_memory WHERE session_id=?", (session_id,))
            return cur.rowcount

    def stm_count(self, session_id: str) -> int:
        with self.conn() as conn:
            row = conn.execute("SELECT COUNT(*) c FROM short_memory WHERE session_id=?",
                               (session_id,)).fetchone()
            return int(row["c"])

    # ------------------------------------------------------------------ 任务状态
    def state_save(self, task_id, session_id, goal, status, version, payload: str):
        ts = now_iso()
        with self.conn() as conn:
            exists = conn.execute("SELECT 1 FROM task_state WHERE task_id=?", (task_id,)).fetchone()
            if exists:
                conn.execute(
                    "UPDATE task_state SET session_id=?, goal=?, status=?, version=?, payload=?,"
                    " updated_at=? WHERE task_id=?",
                    (session_id, goal, status, version, payload, ts, task_id),
                )
            else:
                conn.execute(
                    "INSERT INTO task_state (task_id, session_id, goal, status, version, payload,"
                    " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (task_id, session_id, goal, status, version, payload, ts, ts),
                )
            conn.execute(
                "INSERT INTO task_state_history (task_id, version, payload, ts) VALUES (?,?,?,?)",
                (task_id, version, payload, ts),
            )

    def state_load(self, task_id: str):
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM task_state WHERE task_id=?", (task_id,)).fetchone()
            return dict(row) if row else None

    def state_list(self, limit: int = 20):
        with self.conn() as conn:
            rows = conn.execute(
                "SELECT task_id, status, version, updated_at, goal FROM task_state"
                " ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def state_history(self, task_id: str, limit: int = 20):
        with self.conn() as conn:
            rows = conn.execute(
                "SELECT version, ts FROM task_state_history WHERE task_id=?"
                " ORDER BY version DESC LIMIT ?", (task_id, limit)).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 长期记忆
    def ltm_insert(self, text, kind, source, task_id, session_id, tags, confidence,
                   valid_until, content_hash, dim, embedding: bytes) -> int:
        ts = now_iso()
        with self.conn() as conn:
            cur = conn.execute(
                "INSERT INTO long_memory (text, kind, source, task_id, session_id, tags,"
                " confidence, created_at, updated_at, valid_until, content_hash, dim, embedding)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (text, kind, source, task_id, session_id,
                 json.dumps(tags or [], ensure_ascii=False), confidence,
                 ts, ts, valid_until, content_hash, dim, embedding),
            )
            return int(cur.lastrowid)

    def ltm_all(self, dim: int = None, include_dead: bool = False):
        sql = "SELECT * FROM long_memory"
        where, args = [], []
        if dim:
            where.append("dim=?")
            args.append(dim)
        if not include_dead:
            where.append("superseded_by IS NULL")
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self.conn() as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

    def ltm_get(self, mem_id: int):
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM long_memory WHERE id=?", (mem_id,)).fetchone()
            return dict(row) if row else None

    def ltm_supersede(self, old_id: int, new_id: int):
        with self.conn() as conn:
            conn.execute("UPDATE long_memory SET superseded_by=?, updated_at=? WHERE id=?",
                         (new_id, now_iso(), old_id))

    def ltm_touch(self, mem_id: int):
        with self.conn() as conn:
            conn.execute("UPDATE long_memory SET updated_at=? WHERE id=?", (now_iso(), mem_id))

    def ltm_delete(self, mem_id: int) -> bool:
        with self.conn() as conn:
            return conn.execute("DELETE FROM long_memory WHERE id=?", (mem_id,)).rowcount > 0

    def ltm_count(self) -> int:
        with self.conn() as conn:
            return int(conn.execute("SELECT COUNT(*) c FROM long_memory").fetchone()["c"])

    # ------------------------------------------------------------------ trace
    def trace_add(self, session_id, task_id, step, event, detail, duration_ms=None) -> int:
        with self.conn() as conn:
            cur = conn.execute(
                "INSERT INTO trace (ts, session_id, task_id, step, event, detail, duration_ms)"
                " VALUES (?,?,?,?,?,?,?)",
                (now_iso(), session_id, task_id, step, event,
                 json.dumps(detail, ensure_ascii=False, default=str)[:4000], duration_ms),
            )
            return int(cur.lastrowid)

    def trace_query(self, task_id: str = None, limit: int = 80, event_like: str = None):
        sql, args = "SELECT * FROM trace", []
        where = []
        if task_id:
            where.append("task_id=?")
            args.append(task_id)
        if event_like:
            where.append("event LIKE ?")
            args.append(f"%{event_like}%")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.conn() as conn:
            rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
        return list(reversed(rows))

    def trace_count(self, task_id: str = None) -> int:
        sql, args = "SELECT COUNT(*) c FROM trace", []
        if task_id:
            sql += " WHERE task_id=?"
            args.append(task_id)
        with self.conn() as conn:
            return int(conn.execute(sql, args).fetchone()["c"])
