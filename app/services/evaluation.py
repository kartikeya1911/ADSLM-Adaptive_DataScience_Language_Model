"""
app/services/evaluation.py
===========================
Module  : Evaluation Engine
Purpose : Computes task-specific performance metrics for trained models.

Supported:
    Classification → Accuracy, Precision, Recall, F1-score, Confusion Matrix
    Regression     → RMSE, MAE, R²
    Clustering     → Silhouette Score
    Time-Series    → RMSE, MAE

Industrial Relevance (Enterprise Industrial):
    - RMSE used in vibration prediction and energy consumption forecasting
    - F1-score critical where false negatives (missed faults) are costly
    - Silhouette Score assesses operational mode clustering quality
"""

import numpy as np
from typing import Any, Dict, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score, silhouette_score,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationEngine:
    """
    Static evaluation utility. All methods are @staticmethod for easy access.

    Usage:
        metrics = EvaluationEngine.evaluate("Classification", y_true=y_test, y_pred=preds)
    """

    @staticmethod
    def evaluate(
        task_type: str,
        y_true=None,
        y_pred=None,
        X=None,
    ) -> Dict[str, Any]:
        """
        Routes evaluation to the correct method based on task type.

        Args:
            task_type : One of Classification / Regression / Clustering / Time-Series
            y_true    : Ground truth labels / values
            y_pred    : Predicted labels / values
            X         : Feature matrix (required for Clustering silhouette score)

        Returns:
            dict of metric names → values
        """
        try:
            if task_type == "Classification":
                return EvaluationEngine._classify(y_true, y_pred)
            elif task_type == "Regression":
                return EvaluationEngine._regress(y_true, y_pred)
            elif task_type == "Clustering":
                return EvaluationEngine._cluster(X, y_pred)
            elif task_type == "Time-Series":
                return EvaluationEngine._timeseries(y_true, y_pred)
            else:
                return {"error": f"Unsupported task type: {task_type}"}
        except Exception as e:
            logger.error(f"Evaluation error for {task_type}: {e}")
            return {"error": str(e)}

    # ── Classification ────────────────────────────────────────────────────────

    @staticmethod
    def _classify(y_true, y_pred) -> Dict[str, Any]:
        unique_vals = set(np.unique(y_true)).union(set(np.unique(y_pred)))
        avg = "binary" if (len(unique_vals) == 2 and unique_vals == {0, 1}) else "weighted"
        return {
            "Accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
            "Precision":         round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "Recall":            round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "F1-score":          round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "Confusion Matrix":  confusion_matrix(y_true, y_pred).tolist(),
        }

    # ── Regression ────────────────────────────────────────────────────────────

    @staticmethod
    def _regress(y_true, y_pred) -> Dict[str, Any]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "RMSE": round(float(np.sqrt(mse)), 4),
            "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
            "R2":   round(float(r2_score(y_true, y_pred)), 4),
        }

    # ── Clustering ────────────────────────────────────────────────────────────

    @staticmethod
    def _cluster(X, labels) -> Dict[str, Any]:
        unique = np.unique(labels)
        # Silhouette requires at least 2 non-noise clusters
        valid_labels = labels[labels != -1]  # DBSCAN uses -1 for noise
        valid_X = X[labels != -1] if hasattr(X, "__len__") else X

        if len(np.unique(valid_labels)) > 1:
            score = silhouette_score(valid_X, valid_labels)
        else:
            score = -1.0  # Cannot compute with a single cluster

        n_clusters = int(len([l for l in np.unique(labels) if l != -1]))
        n_noise    = int((labels == -1).sum()) if -1 in labels else 0

        return {
            "Silhouette Score": round(float(score), 4),
            "N Clusters":       n_clusters,
            "N Noise Points":   n_noise,
        }

    # ── Time-Series ───────────────────────────────────────────────────────────

    @staticmethod
    def _timeseries(y_true, y_pred) -> Dict[str, Any]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "RMSE": round(float(np.sqrt(mse)), 4),
            "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        }
