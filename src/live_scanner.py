# src/live_scanner.py
"""Live scanner – orchestrates scheduled tasks defined in custom_scheduler.

It imports the task registry from `custom_scheduler` and schedules each
registered callable according to its configured interval (in minutes).
The scheduler runs indefinitely until a termination signal is received.
"""

import signal
import time
import schedule
from . import custom_scheduler

    """Entry point that simply forwards to ``custom_scheduler.run``.

    ``custom_scheduler`` already registers its default tasks (fetch_data,
    update_models, generate_report) and any additional tasks the user may add.
    The scheduler uses the ``schedule`` library to execute each task at the
    configured interval.  This wrapper exists so that the historic ``live_scanner``
    entry point remains usable without modification of external launch scripts.
    """
    # Ensure the interpreter matches the project's virtual environment
    python_exec = sys.executable
    if not python_exec:
        raise RuntimeError("Unable to determine Python executable for live_scanner")

    # Forward to the custom scheduler's run loop
    custom_scheduler.run()

if __name__ == "__main__":
    main()
