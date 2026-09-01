# config.py - centralized project settings
import os

SEED = 42

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "finance.db")
REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_report.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "project.log")

# Ensure folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Scheduler settings (overridable via environment variables)
FETCH_INTERVAL: int = int(os.getenv("FETCH_INTERVAL", "30"))      # live scanner interval (minutes)
MODEL_INTERVAL: int = int(os.getenv("MODEL_INTERVAL", "180"))     # model-retrain interval (minutes)
REPORT_INTERVAL: int = int(os.getenv("REPORT_INTERVAL", "1440"))  # daily report interval (minutes)
INGESTION_TIME: str = os.getenv("INGESTION_TIME", "02:00")        # UTC time for daily ingestion
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
