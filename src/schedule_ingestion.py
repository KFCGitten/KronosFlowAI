# src/schedule_ingestion.py
"""Simple scheduler that runs the ingestion script once per day at 02:00 UTC.

It uses the ``schedule`` library (already in ``requirements.txt``) and the
shared ``utils`` module for logging.

Run it in the background, e.g. with ``nohup`` or as a systemd service:
    nohup venv/bin/python src/schedule_ingestion.py &
"""

import os
import sys
import subprocess
import time
import schedule
import signal
from typing import Optional
from .utils import log

# Resolve paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import INGESTION_TIME, FETCH_INTERVAL

# Use the interpreter that is currently running this process (works on Linux, macOS, Windows)
PYTHON_EXECUTABLE = sys.executable
INGESTION_SCRIPT = os.path.join(PROJECT_ROOT, "src", "ingestion.py")
INTERVAL_MINUTES: int = FETCH_INTERVAL  # from config

# Interval can be overridden via env, default 02:00 UTC
# INGESTION_TIME is now imported from config

def job():
    """Execute the ingestion script and log the result."""
    log(f"[scheduler] Starting daily ingestion at {INGESTION_TIME}")
    try:
        result = subprocess.run([PYTHON_EXECUTABLE, INGESTION_SCRIPT], capture_output=True, text=True)
        if result.returncode == 0:
            log("[scheduler] Ingestion completed successfully.")
            if result.stdout:
                log(f"[scheduler] stdout: {result.stdout.strip()}")
        else:
            log(f"[scheduler] Ingestion failed (code {result.returncode}). Stderr: {result.stderr.strip()}")
    except Exception as e:
        log(f"[scheduler] Unexpected error: {e}")

# Schedule the job
schedule.every().day.at(INGESTION_TIME).do(job)
log(f"[scheduler] Scheduler initialized – will run daily at {INGESTION_TIME} UTC.")

while True:
    schedule.run_pending()
    time.sleep(60)  # check every minute
