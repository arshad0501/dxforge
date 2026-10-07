"""Lightweight SQLite audit log (pilot). PostgreSQL-ready schema shape.

Records every generated artefact and every human approval/rejection, so the
system can show an audit trail — one of the PDF's governance requirements.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dxforge_audit.db")


def _conn():
    con = sqlite3.connect(_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            project TEXT NOT NULL,
            user TEXT NOT NULL,
            action TEXT NOT NULL,
            intent TEXT,
            request_hex TEXT,
            outcome TEXT,
            status TEXT
        )
    """)
    return con


def log(project: str, user: str, action: str, intent: str = "",
        request_hex: str = "", outcome: str = "", status: str = "generated") -> None:
    con = _conn()
    con.execute(
        "INSERT INTO audit (ts, project, user, action, intent, request_hex, outcome, status)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), project, user,
         action, intent, request_hex, outcome, status),
    )
    con.commit()
    con.close()


def recent(project: str | None = None, limit: int = 100) -> list[dict]:
    con = _conn()
    con.row_factory = sqlite3.Row
    if project:
        rows = con.execute("SELECT * FROM audit WHERE project=? ORDER BY id DESC LIMIT ?",
                           (project, limit)).fetchall()
    else:
        rows = con.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]
