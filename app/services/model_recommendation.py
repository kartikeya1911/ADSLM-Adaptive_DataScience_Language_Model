"""
app/services/model_recommendation.py
======================================
Module  : Model Recommendation Engine
Purpose : Recommends suitable ML algorithms based on dataset characteristics and task type.
          Each recommendation includes a detailed industrial-grade rationale.

Selection Criteria:
    - Dataset size (small vs large)
    - Feature type mix (numerical vs categorical)
    - Problem type
    - Presence of missing data / imbalance

Industrial Relevance (Enterprise Industrial):
    - Guides engineers to the right algorithm for predictive maintenance
    - Avoids overpowered models for small sensor datasets
"""

from typing import Any, Dict, List

from app.core.config import LARGE_DATASET_THRESHOLD
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRecommendationEngine:
    """
    Produces an ordered list of model recommendations with rationale.

    Usage:
        engine = ModelRecommendationEngine(task_type="Classification", dataset_stats=stats)
        recs = engine.recommend()
    """

    def __init__(self, task_type: str, dataset_stats: Dict[str, Any]):
        self.task_type = task_type
        self.stats     = dataset_stats

    def recommend(self) -> List[Dict[str, str]]:
        """
        Returns a list of recommended models, each as:
            {"model": <name>, "reason": <rationale>, "priority": <1-based rank>}
        """
        recs = []
        basic_info     = self.stats.get("basic_info", {})
        is_large       = basic_info.get("row_count", 0) > LARGE_DATASET_THRESHOLD
        has_missing    = bool(self.stats.get("missing_values", {}).get("_summary", {}).get("total_missing_cells", 0))
        cat_count      = len(self.stats.get("column_types", {}).get("categorical", []))
        has_mixed_types= cat_count > 0

        if self.task_type == "Regression":
            recs = self._recommend_regression(is_large, has_mixed_types)
        elif self.task_type == "Classification":
            recs = self._recommend_classification(is_large, has_mixed_types)
        elif self.task_type == "Clustering":
            recs = self._recommend_clustering(is_large)
        elif self.task_type == "Time-Series":
            recs = self._recommend_timeseries()
        else:
            logger.warning(f"No recommendation available for task type: {self.task_type}")

        # Tag with priority rank
        for i, rec in enumerate(recs, start=1):
            rec["priority"] = i

        logger.info(f"Recommended {len(recs)} models for '{self.task_type}' task")
        return recs

    # ── Task-Specific Recommendation Logic ───────────────────────────────────

    def _recommend_regression(self, is_large: bool, has_mixed: bool) -> List[Dict]:
        recs = [
            {
                "model": "Linear Regression",
                "reason": (
                    "Serves as a transparent, highly-interpretable baseline. "
                    "Ideal for datasets with approximately linear feature-target relationships. "
                    "Widely used in industrial applications for process parameter prediction."
                ),
            },
            {
                "model": "Random Forest Regressor",
                "reason": (
                    "Ensemble of decision trees that captures non-linear interactions and "
                    "feature dependencies. Robust to outliers and handles mixed feature types "
                    "well — excellent for complex sensor data regression."
                ),
            },
        ]
        if is_large or has_mixed:
            recs.append({
                "model": "XGBoost Regressor",
                "reason": (
                    "Gradient-boosted tree model; state-of-the-art on structured tabular data. "
                    "Scales efficiently with large datasets and includes built-in regularization "
                    "to prevent overfitting on noisy industrial data."
                ),
            })
        return recs

    def _recommend_classification(self, is_large: bool, has_mixed: bool) -> List[Dict]:
        recs = [
            {
                "model": "Logistic Regression",
                "reason": (
                    "Linear classifier providing high interpretability and fast training. "
                    "Optimal as a baseline for fault/no-fault binary decisions in Enterprise Industrial systems."
                ),
            },
            {
                "model": "Random Forest",
                "reason": (
                    "Ensemble classifier that handles class imbalance gracefully. "
                    "Provides feature importance — critical for understanding which machine "
                    "parameters drive failure predictions."
                ),
            },
        ]
        if is_large:
            recs.append({
                "model": "XGBoost",
                "reason": (
                    "Gold-standard gradient boosting for large-scale classification. "
                    "Handles mixed features and imbalanced classes via class weights. "
                    "Used extensively in industrial anomaly detection."
                ),
            })
        else:
            recs.append({
                "model": "SVM",
                "reason": (
                    "Support Vector Machine is highly effective in high-dimensional feature "
                    "spaces. Excellent for datasets where classes are nearly separable, such "
                    "as quality pass/fail classification on well-controlled production lines."
                ),
            })
        return recs

    def _recommend_clustering(self, is_large: bool) -> List[Dict]:
        return [
            {
                "model": "KMeans",
                "reason": (
                    "Fast, scalable centroid-based clustering. Ideal for partitioning "
                    "machine operational modes (e.g., normal, degraded, critical) from "
                    "unlabeled sensor telemetry."
                ),
            },
            {
                "model": "DBSCAN",
                "reason": (
                    "Density-based algorithm that discovers clusters of arbitrary shape and "
                    "automatically flags outliers as noise. Perfect for anomaly detection in "
                    "manufacturing process data where defects are rare but critical."
                ),
            },
        ]

    def _recommend_timeseries(self) -> List[Dict]:
        return [
            {
                "model": "ARIMA",
                "reason": (
                    "AutoRegressive Integrated Moving Average — classical statistical model "
                    "for univariate time series with trends and seasonality. Widely applied "
                    "to energy consumption forecasting and equipment life prediction."
                ),
            },
            {
                "model": "Prophet",
                "reason": (
                    "Facebook Prophet handles missing data, multiple seasonalities, and "
                    "holiday effects robustly. Excellent for forecasting KPIs like production "
                    "throughput or downtime frequency over industrial shift schedules."
                ),
            },
        ]
