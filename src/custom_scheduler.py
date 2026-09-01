# src/custom_scheduler.py
"""Custom scheduler for sommar‑projekt – lightweight replacement for Penga_Patte12's scheduler.

Features:
- Register arbitrary callables (functions) with a name and interval (minutes).
- All tasks share the same logging via ``utils.log``.
- Configuration (intervals, enabled flags) is read from ``config.py`` – you can override via env vars.
- Simple ``run()`` loop that can be launched as a background process (e.g. ``nohup``).
"""

import os
import sys
import time
import schedule
from typing import Callable, Dict

# Project‑local utilities
from .utils import log

# ---------------------------------------------------------------------------
# Helper: task registry
# ---------------------------------------------------------------------------
TaskRegistry: Dict[str, Callable[[], None]] = {}
TaskIntervals: Dict[str, int] = {}

def register_task(name: str, func: Callable[[], None], interval_minutes: int) -> None:
    """Register *func* to be executed every *interval_minutes* minutes.
    Overwrites an existing entry with the same name.
    """
    TaskRegistry[name] = func
    TaskIntervals[name] = interval_minutes
    log(f"[custom_scheduler] Registered task '{name}' – every {interval_minutes} min")

# ---------------------------------------------------------------------------
# Example task implementations – replace with real logic as needed
# ---------------------------------------------------------------------------

def fetch_data() -> None:
    """Placeholder for a data‑fetch routine.
    In a real project you would call ``src/ingestion.py`` or an external API.
    """
    log("[custom_scheduler] Running fetch_data – invoking ingestion script")
    python_exec = sys.executable
    ingestion_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ingestion.py"))
    try:
        import subprocess
        result = subprocess.run([python_exec, ingestion_path], capture_output=True, text=True)
        if result.returncode == 0:
            log("[custom_scheduler] fetch_data succeeded")
            if result.stdout:
                log(f"[custom_scheduler] stdout: {result.stdout.strip()}")
        else:
            log(f"[custom_scheduler] fetch_data failed (code {result.returncode}) – {result.stderr.strip()}")
    except Exception as e:
        log(f"[custom_scheduler] fetch_data exception: {e}")


def update_models() -> None:
    """Placeholder for a model‑training routine.
    Calls ``src/models.py`` which contains the training logic.
    """
    log("[custom_scheduler] Running update_models – training models")
    python_exec = sys.executable
    models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models.py"))
    try:
        import subprocess
        result = subprocess.run([python_exec, models_path], capture_output=True, text=True)
        if result.returncode == 0:
            log("[custom_scheduler] update_models succeeded")
        else:
            log(f"[custom_scheduler] update_models failed (code {result.returncode}) – {result.stderr.strip()}")
    except Exception as e:
        log(f"[custom_scheduler] update_models exception: {e}")


def generate_report() -> None:
    """Placeholder for a reporting routine.
    Calls ``src/evaluate.py`` which writes ``evaluation_report.json``.
    """
    log("[custom_scheduler] Running generate_report – evaluating models")
    python_exec = sys.executable
    eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluate.py"))
    try:
        import subprocess
        result = subprocess.run([python_exec, eval_path], capture_output=True, text=True)
        if result.returncode == 0:
            log("[custom_scheduler] generate_report succeeded")
        else:
            log(f"[custom_scheduler] generate_report failed (code {result.returncode}) – {result.stderr.strip()}")
    except Exception as e:
        log(f"[custom_scheduler] generate_report exception: {e}")

# ---------------------------------------------------------------------------
# Register default tasks – intervals can be overridden via env vars
# ---------------------------------------------------------------------------
fetch_interval = int(os.getenv("FETCH_INTERVAL", "30"))      # minutes
model_interval = int(os.getenv("MODEL_INTERVAL", "180"))    # minutes (3 h)
report_interval = int(os.getenv("REPORT_INTERVAL", "1440"))  # minutes (daily)

register_task("fetch_data", fetch_data, fetch_interval)
register_task("update_models", update_models, model_interval)
register_task("generate_report", generate_report, report_interval)

# ---------------------------------------------------------------------------
# Translate registry into schedule jobs
# ---------------------------------------------------------------------------
for name, func in TaskRegistry.items():
    minutes = TaskIntervals[name]
    schedule.every(minutes).minutes.do(func)
    log(f"[custom_scheduler] Scheduled '{name}' every {minutes} minutes")

# ---------------------------------------------------------------------------
# Main loop – runs forever (or until SIGINT/SIGTERM)
# ---------------------------------------------------------------------------
def run() -> None:
    log("[custom_scheduler] Starting custom scheduler loop")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # check twice per minute
    except KeyboardInterrupt:
        log("[custom_scheduler] Received KeyboardInterrupt – exiting")
    except Exception as exc:
        log(f"[custom_scheduler] Unexpected error in main loop: {exc}")

if __name__ == "__main__":
    run()
