from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
REPORT_DIR = BASE / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "insurance.csv"
RAW_FILE = RAW_DIR / "insurance_raw.csv"
PROCESSED_FILE = PROCESSED_DIR / "insurance_processed.csv"
QUALITY_REPORT = REPORT_DIR / "quality_report.csv"
VALIDATION_REPORT = REPORT_DIR / "validation_report.json"
BATCH_SIZE = 1000