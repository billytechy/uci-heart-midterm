"""Paths, constants and the random seed shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
EXPECTED_DIR = ROOT / "expected"

SEED = 42
TEST_SIZE = 0.20
MAX_MISSING_SHARE = 0.30  # drop a feature if more than 30% of all patients lack it

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
SITE_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Clinically impossible zeros that stand for "not measured".
ZERO_MEANS_MISSING = ["chol", "trestbps"]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
