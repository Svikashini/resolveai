"""
SQLite persistence for resolution strategies ("episodes").

Single responsibility: own the single `episodes` table - schema creation,
one read, one write. Used in exactly two places: the Planner reads the most
recent successful strategy for an intent before choosing tools, and the
Memory node writes an episode on PASS (and, per spec, on retry-exhausted
FAIL so nothing is lost). No graph logic, no LLM.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT,
    intent      TEXT,
    strategy    TEXT,
    resolution  TEXT,
    score       REAL,
    succeeded   INTEGER
)
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path or config.DB_PATH))


def init_db(db_path: Path | None = None) -> None:
    """Create the episodes table if absent. Idempotent."""
    conn = _connect(db_path)
    try:
        conn.execute(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    logger.info("memory: db ready at %s", db_path or config.DB_PATH)


def get_successful_strategy(intent: str, db_path: Path | None = None) -> dict | None:
    """Most recent succeeded=1 row for this intent, decoded.

    Returns the episode `id` alongside the decoded strategy so the Planner can
    name the exact episode it is reusing in its trace.
    """
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, strategy, resolution, score FROM episodes "
            "WHERE intent = ? AND succeeded = 1 ORDER BY id DESC LIMIT 1",
            (intent,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "strategy": json.loads(row[1]),
        "resolution": json.loads(row[2]),
        "score": row[3],
    }


def store_episode(
    intent: str,
    strategy: list,
    resolution: dict,
    score: float,
    succeeded: bool,
    db_path: Path | None = None,
) -> int:
    """Insert one episode row, commit to disk, return its autoincrement id."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO episodes "
            "(created_at, intent, strategy, resolution, score, succeeded) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                intent,
                json.dumps(strategy),
                json.dumps(resolution),
                float(score),
                1 if succeeded else 0,
            ),
        )
        episode_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    logger.info(
        "memory: stored episode id=%s intent=%s succeeded=%s score=%.2f",
        episode_id,
        intent,
        succeeded,
        score,
    )
    return episode_id
