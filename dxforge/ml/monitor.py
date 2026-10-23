"""Prediction logging and drift monitoring.

Logs every ML prediction (timestamp, text, predicted service, confidence)
to an SQLite table. Surfaces low-confidence rows and confidence distribution
summaries for the ML Lab page.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).parent.parent.parent / "data" / "ml_monitor.db"

_CREATE = """
CREATE TABLE IF NOT EXISTS ml_predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    service     TEXT,
    confidence  REAL,
    abstained   INTEGER DEFAULT 0,
    method      TEXT,
    engine_outcome TEXT
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.execute(_CREATE)
    c.commit()
    return c


def log_prediction(text: str, service: Optional[str], confidence: float,
                   abstained: bool = False, method: str = "ml",
                   engine_outcome: Optional[str] = None) -> None:
    """Persist one prediction row."""
    with _conn() as c:
        c.execute(
            "INSERT INTO ml_predictions (ts, text, service, confidence, "
            "abstained, method, engine_outcome) VALUES (?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), text, service, confidence,
             int(abstained), method, engine_outcome))


def recent_predictions(limit: int = 200) -> pd.DataFrame:
    try:
        with _conn() as c:
            return pd.read_sql(
                f"SELECT * FROM ml_predictions ORDER BY id DESC LIMIT {limit}", c)
    except Exception:
        return pd.DataFrame()


def low_confidence_count(threshold: float = 0.50) -> int:
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM ml_predictions WHERE confidence < ? AND abstained = 0",
                (threshold,)).fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def confidence_summary() -> dict:
    try:
        df = recent_predictions(500)
        if df.empty:
            return {}
        c = df["confidence"].dropna()
        return {
            "mean": float(c.mean()), "median": float(c.median()),
            "p10": float(c.quantile(0.10)), "p90": float(c.quantile(0.90)),
            "low_conf_pct": float((c < 0.50).mean() * 100),
        }
    except Exception:
        return {}
