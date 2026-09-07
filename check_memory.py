"""
Verification script: confirm memory_store.py actually persisted the demo
episodes to SQLite on disk (not just held them in process state).

Flow:
    python main.py            # runs both scenarios, writes 2 episode rows
    python check_memory.py    # this script - reads them back from the .db file

What it does:
    1. opens the SAME SQLite file the pipeline writes to (config.DB_PATH)
    2. prints the on-disk file size and the real CREATE TABLE statement
       (read from sqlite_master, i.e. straight from the database)
    3. queries `episodes` for intent = 'payment_order_mismatch'
    4. prints the stored strategy (tool order), resolution, critic score and
       success flag for BOTH episodes (Rahul / C1001 and Ananya / C1002)
    5. rolls the episodes up via get_strategy_stats(INTENT) - the aggregate
       (intent, tool order) confidence the Planner actually decides on
"""

from __future__ import annotations

import json
import sqlite3

import config

INTENT = "payment_order_mismatch"
NAMES = {"C1001": "Rahul Sharma", "C1002": "Ananya Rao", "C1003": "Karthik Iyer"}


def get_strategy_stats(intent: str) -> list[dict]:
    """Aggregate confidence per distinct strategy (tool order) for an intent.

    Reads the `strategies` table that memory_store upserts on every episode
    write and returns one dict per tool order, best-first by the SAME ranking
    memory_store.best_strategy_for() uses: confidence (success_count /
    times_used), then average score, then most recent use. Complements the
    raw per-episode list with the rolled-up view the Planner reuses from.

    Each dict: strategy, times_used, success_count, confidence, avg_score,
    last_used_at. Returns [] if the db or the strategies table is absent.
    """
    db = config.DB_PATH
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='strategies'"
        ).fetchone()
        if not has_table:
            return []
        rows = conn.execute(
            "SELECT tool_order, times_used, success_count, total_score, last_used_at "
            "FROM strategies WHERE intent = ?",
            (intent,),
        ).fetchall()
    finally:
        conn.close()

    stats: list[dict] = []
    for r in rows:
        used = r["times_used"] or 1
        stats.append(
            {
                "strategy": json.loads(r["tool_order"]),
                "times_used": r["times_used"],
                "success_count": r["success_count"],
                "confidence": r["success_count"] / used,
                "avg_score": r["total_score"] / used,
                "last_used_at": r["last_used_at"],
            }
        )
    stats.sort(
        key=lambda s: (s["confidence"], s["avg_score"], s["last_used_at"] or ""),
        reverse=True,
    )
    return stats


def main() -> None:
    db = config.DB_PATH
    print(f"SQLite file : {db}")
    exists = db.exists()
    size = db.stat().st_size if exists else 0
    print(f"exists      : {exists}")
    print(f"size on disk: {size} bytes")
    if not exists:
        print("\nNo database yet - run `python main.py` first.")
        return

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        schema_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='episodes'"
        ).fetchone()
        print("\n--- CREATE TABLE (from sqlite_master) ---")
        print(schema_row["sql"] if schema_row else "(no 'episodes' table found)")

        total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        rows = conn.execute(
            "SELECT id, created_at, intent, strategy, resolution, score, succeeded "
            "FROM episodes WHERE intent = ? ORDER BY id",
            (INTENT,),
        ).fetchall()
    finally:
        conn.close()

    print(f"\ntotal rows in episodes         : {total}")
    print(f"rows where intent = {INTENT!r} : {len(rows)}")
    print("-" * 74)

    for r in rows:
        strategy = json.loads(r["strategy"])
        resolution = json.loads(r["resolution"])
        cust = (resolution.get("args") or {}).get("customer_id")
        print(f"\nepisode #{r['id']}   created_at = {r['created_at']}")
        print(f"  customer      : {cust}  ({NAMES.get(cust, '?')})")
        print(f"  intent        : {r['intent']}")
        print(f"  strategy      : {strategy}   <- tool order")
        print(f"  resolution    : {resolution.get('action')}   args = {resolution.get('args')}")
        print(f"  critic score  : {r['score']}")
        print(f"  success flag  : {r['succeeded']}  ({'PASS' if r['succeeded'] else 'FAIL'})")

    print("\n" + "-" * 74)
    print(f"aggregate strategy confidence  (intent = {INTENT!r})")
    print("-" * 74)
    stats = get_strategy_stats(INTENT)
    if not stats:
        print("  (no 'strategies' table yet - run `python main.py` first)")
    else:
        for i, s in enumerate(stats):
            pct = round(s["confidence"] * 100)
            tag = "   <- Planner reuses this" if i == 0 else ""
            print(f"\n  tool order    : {s['strategy']}")
            print(
                f"  confidence    : {pct}% "
                f"({s['success_count']}/{s['times_used']} successful){tag}"
            )
            print(f"  avg score     : {s['avg_score']:.2f}")
            print(f"  last used at  : {s['last_used_at']}")

    print("\n" + "-" * 74)
    if len(rows) >= 3:
        print("OK - all three episodes were read back from the SQLite file on disk.")
    elif len(rows) >= 2:
        print("OK - episodes were read back from the SQLite file on disk.")
    else:
        print("Expected 3 rows (Rahul + Ananya + Karthik). Run `python main.py` with no --scenario first.")


if __name__ == "__main__":
    main()
