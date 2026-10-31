from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
DOCUMENTS_DIR = ROOT / "documents"
CHUNKS_PATH = PROCESSED_DIR / "passenger_rights_chunks.jsonl"
FLIGHTS_PATH = PROCESSED_DIR / "flights.parquet"
MODEL_PATH = MODEL_DIR / "flight_delay_model.joblib"
BTS_BASE_URL = "https://transtats.bts.gov/PREZIP/"
DOT_AIR_CONSUMER_URL = "https://www.transportation.gov/airconsumer"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "flight_delay_model")
FEATURES = ["OP_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME", "MONTH", "DAY_OF_WEEK", "DISTANCE"]
TARGET = "DELAY_15"

for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, DOCUMENTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
