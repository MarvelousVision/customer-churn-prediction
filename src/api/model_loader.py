import sys
import json
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))  # must be BEFORE joblib.load()

MODEL_PATH = ROOT_DIR / "artifacts" / "churn_model.joblib"
METADATA_PATH = ROOT_DIR / "artifacts" / "model_info.json"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    model_info = json.load(f)
THRESHOLD = model_info["threshold"] 
