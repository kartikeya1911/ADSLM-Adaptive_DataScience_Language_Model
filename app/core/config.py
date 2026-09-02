"""
app/core/config.py
==================
Central configuration for the ADSLM application.
Uses environment variables with sensible defaults for industrial deployments.
"""

import os
from pathlib import Path

# Silence joblib/loky CPU count detection warning on Windows (wmic subprocess call)
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── Directory Paths ───────────────────────────────────────────────────────────
DATASETS_DIR    = BASE_DIR / "datasets"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR     = BASE_DIR / "reports"

# Create directories if they do not exist
for _dir in [DATASETS_DIR, SAVED_MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── API Settings ──────────────────────────────────────────────────────────────
API_TITLE       = "ADSLM API"
API_VERSION     = "1.0.0"
API_DESCRIPTION = (
    "Adaptive Data Science Language Model — An industrial-grade AutoML + AI Copilot "
    "system designed for ABB innovation evaluation."
)

# ── ML Defaults ───────────────────────────────────────────────────────────────
DEFAULT_TEST_SIZE       = 0.20   # 80/20 train-test split
DEFAULT_RANDOM_STATE    = 42
LARGE_DATASET_THRESHOLD = 10_000  # rows — determines model recommendations
N_KMEANS_CLUSTERS       = 3       # default for KMeans

# ── Expertise Levels ──────────────────────────────────────────────────────────
EXPERTISE_LEVELS = ["beginner", "intermediate", "expert"]
