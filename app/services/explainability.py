"""
app/services/explainability.py
================================
Module  : Explainable AI (XAI) Module
Purpose : Extracts model reasoning — feature importances, coefficient weights,
          model strengths and limitations. SHAP-ready architecture.

Why XAI matters for ABB:
    - Engineers must understand WHY the model flags a fault
    - Regulatory compliance requires decision traceability
    - Enables engineers to trust and validate AI recommendations
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityEngine:
    """
    Extracts and formats model explanations for any supported estimator.

    Usage:
        xai = ExplainabilityEngine(model, feature_names=["temp", "vibration", "rpm"])
        importances = xai.get_feature_importance()
        summary     = xai.get_model_summary()
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model         = model
        self.feature_names = feature_names

    # ── Feature Importance ────────────────────────────────────────────────────

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extracts feature importances from the trained model.

        Supports:
            - Tree-based models (feature_importances_): RF, XGBoost, etc.
            - Linear models (coef_): Logistic Regression, Linear Regression

        Returns:
            Sorted dict {feature_name: importance_score}
        """
        importances: Dict[str, float] = {}

        try:
            # Tree-based models
            if hasattr(self.model, "feature_importances_"):
                vals = self.model.feature_importances_
                importances = dict(zip(self.feature_names, [float(v) for v in vals]))

            # Linear models
            elif hasattr(self.model, "coef_"):
                coefs = self.model.coef_
                # Multi-class → average absolute coefficients per feature
                if coefs.ndim > 1:
                    coefs = np.mean(np.abs(coefs), axis=0)
                else:
                    coefs = np.abs(coefs)
                importances = dict(zip(self.feature_names, [float(v) for v in coefs]))

            else:
                logger.warning("Model does not expose feature importances natively.")
                return {"_note": "Feature importance not available for this model type."}

        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {"error": str(e)}

        # Sort descending by importance
        return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    def get_top_features(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Returns top-N features as a list of dicts for easy display.

        Returns:
            [{"rank": 1, "feature": "vibration", "importance": 0.32}, …]
        """
        fi = self.get_feature_importance()
        if "error" in fi or "_note" in fi:
            return []
        return [
            {"rank": i + 1, "feature": feat, "importance": round(score, 6)}
            for i, (feat, score) in enumerate(list(fi.items())[:n])
        ]

    # ── Model Profile ─────────────────────────────────────────────────────────

    def get_model_summary(self) -> Dict[str, str]:
        """
        Returns a textual summary of the model's type, strengths, and limitations.
        """
        model_class = type(self.model).__name__

        profiles: Dict[str, Dict[str, str]] = {
            "RandomForestClassifier": {
                "type":       "Ensemble — Bagging",
                "strengths":  "Handles non-linearity, robust to outliers, built-in feature importance.",
                "limitations":"Less interpretable than linear models; slower prediction on very large datasets.",
            },
            "RandomForestRegressor": {
                "type":       "Ensemble — Bagging",
                "strengths":  "Robust to noise, captures interaction effects, good on mixed data.",
                "limitations":"Slow inference at scale; model size can be large.",
            },
            "LogisticRegression": {
                "type":       "Linear Classifier",
                "strengths":  "Highly interpretable, fast, probabilistic output.",
                "limitations":"Assumes linear decision boundary; struggles with complex patterns.",
            },
            "LinearRegression": {
                "type":       "Linear Regressor",
                "strengths":  "Maximum interpretability, fast, no hyperparameters.",
                "limitations":"Fails when relationship is non-linear or features are collinear.",
            },
            "SVC": {
                "type":       "Kernel-based Classifier",
                "strengths":  "Works well in high-dimensional spaces; effective with small datasets.",
                "limitations":"Computationally expensive on large datasets; no native probability without calibration.",
            },
            "XGBClassifier": {
                "type":       "Ensemble — Gradient Boosting",
                "strengths":  "State-of-the-art accuracy on tabular data, handles missing values natively.",
                "limitations":"Requires careful hyperparameter tuning; can overfit small datasets.",
            },
            "XGBRegressor": {
                "type":       "Ensemble — Gradient Boosting",
                "strengths":  "Best-in-class for structured regression; built-in regularization.",
                "limitations":"Black-box nature reduces direct interpretability without XAI tools.",
            },
            "KMeans": {
                "type":       "Centroid-based Clustering",
                "strengths":  "Fast, scalable, easy to interpret cluster centers.",
                "limitations":"Requires pre-specifying K; sensitive to outliers and non-spherical clusters.",
            },
            "DBSCAN": {
                "type":       "Density-based Clustering",
                "strengths":  "Automatically detects number of clusters; robust outlier identification.",
                "limitations":"Sensitive to eps/min_samples hyperparameters; struggles with varying densities.",
            },
        }

        profile = profiles.get(model_class, {
            "type":       model_class,
            "strengths":  "General-purpose ML model.",
            "limitations":"See model documentation for details.",
        })

        return {"model_class": model_class, **profile}

    # ── SHAP Placeholder (Future) ─────────────────────────────────────────────

    @staticmethod
    def generate_shap_explanation(model: Any, X_sample: pd.DataFrame) -> str:
        """
        Future hook for SHAP-based explanations.
        Install shap and replace this stub with:
            import shap
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)
        """
        return (
            "SHAP (SHapley Additive exPlanations) is supported in the extended version. "
            "It provides per-instance feature contribution explanations — ideal for ABB "
            "predictive maintenance use cases where individual asset decisions must be justified."
        )
