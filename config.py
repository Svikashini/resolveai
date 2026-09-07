"""
Environment-driven configuration and process-wide logging setup.

Single responsibility: read settings from the environment once, expose them
as plain module-level constants, and configure the root logger. No business
logic; the only I/O is loading a local .env file.

Business thresholds deliberately live NOT here but next to the code that
applies them (`policy.REFUND_AUTO_APPROVE_LIMIT`, `critic.PASS_THRESHOLD`),
so this module stays purely infrastructural.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:  # python-dotenv is optional; the mock demo runs fine without a .env
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - best effort only
    pass


_BASE_DIR = Path(__file__).resolve().parent


LLM_MODE: str = os.getenv("RESOLVEAI_LLM_MODE", "mock").strip().lower()
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

DATA_DIR: Path = Path(os.getenv("RESOLVEAI_DATA_DIR", str(_BASE_DIR / "data")))
DB_PATH: Path = Path(os.getenv("RESOLVEAI_DB_PATH", str(_BASE_DIR / "resolveai_memory.db")))

# Graph-topology knob for the replan loop (see graph.route_after_critic).
MAX_REPLAN_RETRIES: int = 2


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once. Called from main.py at startup."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s",
        datefmt="%H:%M:%S",
    )
