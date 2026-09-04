"""
app/services/training.py
=========================
Module  : Automated Training Engine
Purpose : Trains multiple models, compares performance, selects and saves the best.

Architecture:
    - Model Factory (registry dict) → clean, extensible design
    - EvaluationEngine integration for metric computation
    - joblib for serialisation of best model

Industrial Relevance (Enterprise Industrial):
    - Automates model selection for predictive maintenance pipelines
    - Modular design allows plugging in custom estimators (e.g., LSTM, AutoML)
"""

import os
import joblib
from typing import Any, Dict, List, Optional

# ── Classifiers ───────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN

# ── Optional XGBoost ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from app.services.evaluation import EvaluationEngine
from app.core.config import SAVED_MODELS_DIR, DEFAULT_RANDOM_STATE, N_KMEANS_CLUSTERS
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingEngine:
    """
    Trains all recommended models for the detected task type,
    evaluates each, and saves the best-performing model to disk.

    Usage:
        trainer = TrainingEngine("Classification", X_train, y_train, X_test, y_test, model_names)
        results = trainer.run()
    """

    def __init__(
        self,
        task_type:     str,
        X_train,
        y_train,
        X_test,
        y_test,
        models_to_run: List[str],
    ):
        self.task_type     = task_type
        self.X_train       = X_train
        self.y_train       = y_train
        self.X_test        = X_test
        self.y_test        = y_test
        self.models_to_run = models_to_run

        # ── Model Factory / Registry ─────────────────────────────────────────
        self.model_registry: Dict[str, Any] = {
            # Classification
            "Logistic Regression":          LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE),
            "Random Forest":                RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=DEFAULT_RANDOM_STATE),
            "SVM":                          SVC(probability=True, max_iter=2000, random_state=DEFAULT_RANDOM_STATE),
            # Regression
            "Linear Regression":            LinearRegression(n_jobs=-1),
            "Random Forest Regressor":      RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=DEFAULT_RANDOM_STATE),
            # Clustering
            "KMeans":                       KMeans(n_clusters=N_KMEANS_CLUSTERS, random_state=DEFAULT_RANDOM_STATE, n_init=5),
            "DBSCAN":                       DBSCAN(eps=0.5, min_samples=5, n_jobs=-1),
            # Time-Series (Regression models fit on temporal tabular features)
            "ARIMA":                        LinearRegression(n_jobs=-1),
            "Prophet":                      RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=DEFAULT_RANDOM_STATE),
        }

        if HAS_XGB:
            self.model_registry["XGBoost"]          = XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=DEFAULT_RANDOM_STATE, verbosity=0, n_jobs=-1
            )
            self.model_registry["XGBoost Regressor"]= XGBRegressor(
                random_state=DEFAULT_RANDOM_STATE, verbosity=0, n_jobs=-1
            )
            logger.info("XGBoost models registered successfully.")
        else:
            logger.warning("XGBoost not installed — XGB models skipped.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Trains and evaluates all specified models.

        Returns dict with:
            all_results       : {model_name: metrics_dict}
            best_model        : name of best model
            best_score_metric : numeric score of best model
            saved_model_path  : file path where best model was saved
        """
        all_results: Dict[str, Dict] = {}
        best_model_name: Optional[str] = None
        best_model_obj:  Optional[Any] = None
        best_score:      Optional[float] = None

        for model_name in self.models_to_run:
            if model_name not in self.model_registry:
                logger.warning(f"Model '{model_name}' not found in registry — skipping.")
                continue

            model = self.model_registry[model_name]
            logger.info(f"Training: {model_name} …")

            try:
                metrics, score = self._train_and_evaluate(model, model_name)
                all_results[model_name] = metrics

                # Determine whether this is the new best model
                if self.task_type in ("Regression", "Time-Series"):
                    # Lower RMSE is better
                    if best_score is None or score < best_score:
                        best_score, best_model_name, best_model_obj = score, model_name, model
                else:
                    # Higher score is better (F1 or Silhouette)
                    if best_score is None or score > best_score:
                        best_score, best_model_name, best_model_obj = score, model_name, model

                logger.info(f"{model_name} → score={score:.4f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}

        # Save best model
        saved_path = self._save_model(best_model_obj) if best_model_obj else None

        return {
            "all_results":      all_results,
            "best_model":       best_model_name,
            "best_score_metric": float(best_score) if best_score is not None else None,
            "saved_model_path": str(saved_path) if saved_path else None,
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _train_and_evaluate(self, model, model_name: str):
        """Fits the model and returns (metrics_dict, primary_score)."""

        if self.task_type == "Clustering":
            # For massive datasets, sample up to 5,000 points for clustering evaluation to prevent memory freeze
            X_fit = self.X_train.iloc[:5000] if len(self.X_train) > 5000 else self.X_train
            model.fit(X_fit)
            labels  = model.labels_
            metrics = EvaluationEngine.evaluate(
                self.task_type, X=X_fit, y_pred=labels
            )
            score = metrics.get("Silhouette Score", -1.0)

        else:
            # Subsample for SVM if dataset is large (>3000 rows) to prevent O(N^3) hang
            if model_name == "SVM" and len(self.X_train) > 3000:
                X_tr_fit = self.X_train.iloc[:3000]
                y_tr_fit = self.y_train.iloc[:3000]
            else:
                X_tr_fit = self.X_train
                y_tr_fit = self.y_train

            model.fit(X_tr_fit, y_tr_fit)
            preds   = model.predict(self.X_test)
            metrics = EvaluationEngine.evaluate(
                self.task_type, y_true=self.y_test, y_pred=preds
            )
            if self.task_type == "Classification":
                score = metrics.get("F1-score", 0.0)
            else:  # Regression / Time-Series
                score = metrics.get("RMSE", float("inf"))

        return metrics, score

    def _save_model(self, model) -> str:
        """Saves the best model using joblib."""
        try:
            save_path = SAVED_MODELS_DIR / f"best_{self.task_type.lower()}_model.pkl"
            joblib.dump(model, save_path)
            logger.info(f"Best model saved → {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Could not save model: {e}")
            return ""
