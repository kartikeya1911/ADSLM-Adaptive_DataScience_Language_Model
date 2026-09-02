"""
app/services/task_detection.py
================================
Module  : Task Detection Engine
Purpose : Automatically determines the ML problem type from dataset properties.

Decision Rules:
    1. datetime column present AND numeric target → Time-Series
    2. No target column specified           → Clustering
    3. Target is categorical                → Classification
    4. Target is numerical                  → Regression (with smart binary override)

Industrial Relevance (ABB):
    - Predicts whether incoming sensor stream is forecasting, anomaly, or classification
"""

from typing import Any, Dict, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TaskDetector:
    """
    Detects the ML task type from dataset analysis results.

    Usage:
        detector = TaskDetector(analysis_stats, target_column="Fault")
        task_type = detector.detect_task()   # → "Classification"
    """

    SUPPORTED_TASKS = ["Regression", "Classification", "Clustering", "Time-Series"]

    def __init__(self, dataset_analysis: Dict[str, Any], target_column: Optional[str] = None):
        """
        Args:
            dataset_analysis : Output dict from DatasetAnalyzer.analyze()
            target_column    : Column name selected by the user as the prediction target.
        """
        self.analysis = dataset_analysis
        self.target   = target_column

    def detect_task(self) -> str:
        """
        Runs the task detection logic and returns a task label.

        Returns:
            One of: "Regression", "Classification", "Clustering", "Time-Series", "Unknown"
        """
        task = self._apply_rules()
        logger.info(f"Task detected → {task} (target='{self.target}')")
        return task

    def get_detection_reason(self) -> str:
        """Returns a human-readable explanation of why this task was chosen."""
        task = self.detect_task()
        reasons = {
            "Clustering":     "No target column was specified. The system will discover natural groupings in the data.",
            "Time-Series":    f"A datetime column was found alongside a numerical target '{self.target}', indicating temporal forecasting.",
            "Classification": f"Target column '{self.target}' contains categorical (discrete) values — predicting classes.",
            "Regression":     f"Target column '{self.target}' contains continuous numerical values — predicting quantities.",
            "Unknown":        "Could not determine the task type. Please verify the target column.",
        }
        return reasons.get(task, "Unknown task.")

    # ── Private Rule Engine ───────────────────────────────────────────────────

    def _apply_rules(self) -> str:
        col_types    = self.analysis.get("column_types", {})
        numerical    = col_types.get("numerical", [])
        categorical  = col_types.get("categorical", [])
        datetime_cols= col_types.get("datetime", [])

        # Rule 1: No target → Clustering (unsupervised)
        if not self.target:
            return "Clustering"

        # Rule 2: Time-Series — datetime present AND numerical target
        if datetime_cols and self.target in numerical:
            return "Time-Series"

        # Rule 3: Categorical target → Classification
        if self.target in categorical:
            return "Classification"

        # Rule 4: Numerical target → Regression by default
        #   Smart override: if 2 or fewer unique values (e.g. 0/1 binary target), treat as Classification
        if self.target in numerical:
            summary = self.analysis.get("summary_statistics", {}).get("numerical", {})
            unique_count = summary.get(self.target, {}).get("nunique", 999)
            if unique_count <= 2:
                logger.info(f"Target '{self.target}' is numerical with {unique_count} unique values → override to Classification")
                return "Classification"
            return "Regression"

        return "Unknown"
