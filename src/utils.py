# src/utils.py
"""Utility module shared across the sommar‑projekt.
Provides:
- `get_db_connection()` – returns a SQLite connection to the finance DB.
- `load_report()` – loads the JSON evaluation report.
- `log(msg)` – simple timestamped logger (writes to stdout and optional logfile).
- `read_env(key, default=None)` – convenient wrapper around `os.getenv`.
"""

import os
import json
import sqlite3
import sys
from datetime import datetime
from typing import Optional, Dict, List

# ---------------------------------------------------------------------------
# Paths come from the single project-wide config module (see config.py)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DB_PATH, REPORT_PATH, LOG_FILE


def log(message: str) -> None:
    """Write a timestamped log line to stdout and to the log file.

    The function is deliberately lightweight – no external logging framework
    is required for this lightweight project.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {message}\n"
    # stdout (visible in the terminal)
    sys.stdout.write(line)
    sys.stdout.flush()
    # also append to a persistent log file
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        # In a sandboxed environment we ignore file‑write errors – the stdout
        # output is still useful for debugging.
        pass


def get_db_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the finance database.

    The caller is responsible for closing the connection when done.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    return conn


def load_report() -> Optional[Dict]:
    """Load the evaluation report JSON if it exists.

    Returns ``None`` when the file is missing or cannot be parsed.
    """
    if not os.path.exists(REPORT_PATH):
        log(f"Report not found at {REPORT_PATH}")
        return None
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to load report: {e}")
        return None


def fetch_all_tickers() -> List[str]:
    """Return a sorted list of distinct tickers from the SQLite DB.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT ticker FROM historical_prices ORDER BY ticker ASC")
        tickers = [row[0] for row in cur.fetchall()]
        conn.close()
        return tickers
    except Exception as e:
        log(f"Error fetching tickers: {e}")
        return []


def read_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Wrapper around ``os.getenv`` that also logs missing keys.
    """
    value = os.getenv(key, default)
    if value is None:
        log(f"Environment variable '{key}' not set; using default.")
    return value

# Exported symbols for convenience
__all__ = [
    "log",
    "get_db_connection",
    "load_report",
    "fetch_all_tickers",
    "read_env",
]
