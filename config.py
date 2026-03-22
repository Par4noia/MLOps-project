from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "insurance.csv"
RAW_FILE = RAW_DIR / "insurance_raw.csv"
BATCH_SIZE = 500
