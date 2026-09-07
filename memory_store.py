"""
SQLite persistence for resolution strategies ("episodes") and their rolled-up
confidence ("strategies").

Two tables, both owned here:

    episodes    one row per resolution attempt - the raw history.
    strategies  one row per (intent, tool-order-as-string), carrying the
                running counts the Planner uses to pick a strategy by
                *confidence* (success_count / times_used) rather than by
                recency.

Used in exactly two places: the Planner calls best_strategy_for(intent) to
reuse the highest-confidence tool order before choosing tools, and the Memory
node writes an episode on PASS (and, per spec, on retry-exhausted FAIL so
nothing is lost) - every episode write also upserts the matching strategies
row. No graph logic, no LLM.
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
);

CREATE TABLE IF NOT EXISTS strategies (
    intent        TEXT    NOT NULL,
    tool_order    TEXT    NOT NULL,
    times_used    INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    total_score   REAL    NOT NULL DEFAULT 0.0,
    last_used_at  TEXT,
    PRIMARY KEY (intent, tool_order)
);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path or config.DB_PATH))


def init_db(db_path: Path | None = None) -> None:
    """Create the episodes and strategies tables if absent. Idempotent."""
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    logger.info("memory: db ready at %s", db_path or config.DB_PATH)


def _strategy_key(strategy: list) -> str:
    """Canonical string form of a tool order - the strategies table's key."""
    return json.dumps(list(strategy))


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


def best_strategy_for(intent: str, db_path: Path | None = None) -> dict | None:
    """Highest-*confidence* strategy on record for this intent, or None.

    Confidence is success_count / times_used. Ties are broken by the higher
    average score (total_score / times_used), then by the most recent
    last_used_at (ISO-8601, so a plain string compare orders by time). This
    is a genuine aggregate over every episode with this intent - NOT simply
    the newest successful episode.

    Returned dict:
        strategy       decoded tool order (list[str])
        times_used, success_count, total_score, last_used_at   raw columns
        confidence     success_count / times_used   (0.0 .. 1.0)
        avg_score      total_score / times_used
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT tool_order, times_used, success_count, total_score, last_used_at "
            "FROM strategies WHERE intent = ? AND times_used > 0",
            (intent,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None

    def _rank(r: tuple) -> tuple:
        _order, times_used, success_count, total_score, last_used_at = r
        return (
            success_count / times_used,
            total_score / times_used,
            last_used_at or "",
        )

    order, times_used, success_count, total_score, last_used_at = max(rows, key=_rank)
    return {
        "intent": intent,
        "strategy": json.loads(order),
        "times_used": times_used,
        "success_count": success_count,
        "total_score": total_score,
        "last_used_at": last_used_at,
        "confidence": success_count / times_used,
        "avg_score": total_score / times_used,
    }


def _bump_strategy(
    conn: sqlite3.Connection,
    intent: str,
    strategy: list,
    score: float,
    succeeded: bool,
) -> None:
    """Upsert the strategies row for (intent, tool order) on `conn`.

    Increments times_used, adds `succeeded` (0/1) to success_count, adds
    `score` to total_score and stamps last_used_at. Does NOT commit - the
    caller (store_episode) commits the episode insert and this bump together
    so an episode and its aggregate never drift apart.
    """
    key = _strategy_key(strategy)
    now = datetime.now(timezone.utc).isoformat()
    won = 1 if succeeded else 0
    row = conn.execute(
        "SELECT times_used, success_count, total_score FROM strategies "
        "WHERE intent = ? AND tool_order = ?",
        (intent, key),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO strategies "
            "(intent, tool_order, times_used, success_count, total_score, last_used_at) "
            "VALUES (?, ?, 1, ?, ?, ?)",
            (intent, key, won, float(score), now),
        )
    else:
        conn.execute(
            "UPDATE strategies SET times_used = ?, success_count = ?, "
            "total_score = ?, last_used_at = ? WHERE intent = ? AND tool_order = ?",
            (row[0] + 1, row[1] + won, row[2] + float(score), now, intent, key),
        )


def store_episode(
    intent: str,
    strategy: list,
    resolution: dict,
    score: float,
    succeeded: bool,
    db_path: Path | None = None,
) -> int:
    """Insert one episode row, upsert its strategies aggregate, commit both to
    disk in one transaction, and return the episode's autoincrement id."""
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
        _bump_strategy(conn, intent, strategy, float(score), bool(succeeded))
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
