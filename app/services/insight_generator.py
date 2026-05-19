"""
app/services/insight_generator.py
===================================
Module  : Natural Language Insight Generator
Purpose : Translates raw statistics, task types, and model metrics into
          human-readable, AI-style narrative insights.

Output Examples:
    "Random Forest performed best because the dataset contains nonlinear
     relationships and mixed feature types."

    "12.3% of records had missing values, which were automatically imputed
     to preserve data integrity."

Industrial Relevance (ABB):
    - Provides non-expert engineers with plain-English summaries
    - Reduces the gap between data science outputs and operational decisions
"""

from typing import Any, Dict, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


class InsightGenerator:
    """
    Generates structured, AI-style natural language insights.

    All methods are @staticmethod for convenience.

    Usage:
        text = InsightGenerator.generate_data_profile_insights(analysis_stats)
        text = InsightGenerator.generate_task_insight("Classification", "Fault")
        text = InsightGenerator.generate_model_insight("Random Forest", "Classification", metrics)
        text = InsightGenerator.generate_feature_insight(importances)
        text = InsightGenerator.generate_preprocessing_insight(prep_summary)
    """

    # ── Dataset Profile ───────────────────────────────────────────────────────

    @staticmethod
    def generate_data_profile_insights(analysis: Dict[str, Any]) -> str:
        """Generates a narrative about dataset size, health, and structure."""
        rows   = analysis["basic_info"]["row_count"]
        cols   = analysis["basic_info"]["column_count"]
        dupes  = analysis["basic_info"]["duplicate_rows"]
        miss   = analysis.get("missing_values", {}).get("_summary", {})
        num_cols= len(analysis["column_types"].get("numerical", []))
        cat_cols= len(analysis["column_types"].get("categorical", []))
        out_info= analysis.get("outlier_info", {}).get("_summary", {})

        parts = [
            f"The dataset contains {rows:,} records across {cols} features "
            f"({num_cols} numerical, {cat_cols} categorical)."
        ]

        if miss.get("overall_missing_pct", 0) > 0:
            parts.append(
                f"Missing values were found in {miss['total_missing_cells']:,} cells "
                f"({miss['overall_missing_pct']}% of the dataset), automatically imputed "
                f"to preserve analytical integrity."
            )
        else:
            parts.append("The dataset is complete — no missing values detected.")

        if dupes > 0:
            parts.append(f"{dupes} duplicate rows were identified and removed before training.")

        n_out_cols = out_info.get("columns_with_outliers", 0)
        if n_out_cols > 0:
            parts.append(
                f"Outliers were detected in {n_out_cols} column(s) using the IQR method — "
                f"these were retained for training but may warrant domain investigation."
            )

        return " ".join(parts)

    # ── Task Detection ────────────────────────────────────────────────────────

    @staticmethod
    def generate_task_insight(task_type: str, target: str) -> str:
        """Explains the detected task type in business terms."""
        messages = {
            "Regression": (
                f"A Regression problem was detected because the target column '{target}' "
                f"holds continuous numerical values. The model will predict quantities — "
                f"such as energy consumption, machine temperature, or throughput rate."
            ),
            "Classification": (
                f"A Classification problem was detected because the target column '{target}' "
                f"contains discrete categories (e.g., Fault / No-Fault, Pass / Fail). "
                f"The model will assign each sample to a class."
            ),
            "Clustering": (
                "No target column was specified. The system applies Unsupervised Learning — "
                "discovering natural groupings in the data without predefined labels. "
                "Useful for operational mode detection and anomaly profiling."
            ),
            "Time-Series": (
                f"A Time-Series structure was inferred because a datetime column is present "
                f"alongside the numerical target '{target}'. The model will learn temporal "
                f"patterns to forecast future values."
            ),
        }
        return messages.get(task_type, "Task type could not be determined from the dataset.")

    # ── Model Performance ─────────────────────────────────────────────────────

    @staticmethod
    def generate_model_insight(best_model: str, task_type: str, metrics: Dict[str, Any]) -> str:
        """Explains why the winning model was selected and its key performance metric."""
        parts = [f"After training and comparing all recommended models, {best_model} achieved the best performance."]

        if task_type == "Classification":
            f1 = metrics.get("F1-score", 0)
            acc = metrics.get("Accuracy", 0)
            parts.append(
                f"It achieved an F1-score of {f1:.4f} and an accuracy of {acc:.2%}, "
                f"indicating a strong balance between correctly identifying true faults "
                f"and avoiding false alarms."
            )
        elif task_type == "Regression":
            rmse = metrics.get("RMSE", 0)
            r2   = metrics.get("R2", 0)
            parts.append(
                f"It minimised the Root Mean Squared Error (RMSE) to {rmse:.4f} "
                f"with an R² of {r2:.4f}, explaining {r2*100:.1f}% of the variance "
                f"in the target variable."
            )
        elif task_type == "Clustering":
            sil = metrics.get("Silhouette Score", 0)
            n_c = metrics.get("N Clusters", "N/A")
            parts.append(
                f"It formed {n_c} coherent clusters with a Silhouette Score of {sil:.4f} "
                f"(scores near +1 indicate well-separated, compact clusters)."
            )
        elif task_type == "Time-Series":
            rmse = metrics.get("RMSE", 0)
            parts.append(f"It achieved a forecasting RMSE of {rmse:.4f} on the held-out test window.")

        return " ".join(parts)

    # ── Feature Importance ────────────────────────────────────────────────────

    @staticmethod
    def generate_feature_insight(importances: Dict[str, float]) -> str:
        """Narrates which features are most influential in the model's decisions."""
        if not importances or "_note" in importances or "error" in importances:
            return (
                "Feature importance scores are not directly extractable for this model type. "
                "Consider using SHAP for model-agnostic explanations."
            )

        top = list(importances.items())[:3]
        names = [f"'{k}'" for k, _ in top]

        return (
            f"The top predictive features are {', '.join(names)}. "
            f"These variables carry the highest statistical weight in the model's decision process. "
            f"In an industrial context, focusing data quality efforts on these columns will "
            f"yield the greatest improvement in prediction accuracy."
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def generate_preprocessing_insight(prep_summary: Dict[str, Any]) -> str:
        """Summarises what the preprocessing pipeline changed."""
        parts = []
        if prep_summary.get("duplicates_removed", 0) > 0:
            parts.append(f"{prep_summary['duplicates_removed']} duplicate rows removed.")

        imp = prep_summary.get("imputation", {})
        if imp.get("numerical"):
            parts.append("Numerical missing values filled using median imputation.")
        if imp.get("categorical"):
            parts.append("Categorical gaps filled using most-frequent (mode) strategy.")

        enc = prep_summary.get("encoding", {})
        if enc.get("columns_encoded"):
            parts.append(
                f"Categorical columns {enc['columns_encoded']} were one-hot encoded."
            )

        parts.append(f"All features standardised to zero mean and unit variance (StandardScaler).")
        fc = prep_summary.get("final_feature_count")
        if fc:
            parts.append(f"Final feature matrix contains {fc} columns after encoding.")

        return " ".join(parts) if parts else "Standard preprocessing pipeline applied."

    # ── Recommendations ───────────────────────────────────────────────────────

    @staticmethod
    def generate_actionable_recommendations(
        task_type: str, metrics: Dict[str, Any], prep_summary: Dict[str, Any]
    ) -> List[str]:
        """Returns a list of actionable next-step recommendations."""
        recs = []

        if task_type == "Classification":
            f1 = metrics.get("F1-score", 1.0)
            if f1 < 0.75:
                recs.append("Consider collecting more labelled samples — F1 below 0.75 suggests insufficient training data.")
            recs.append("Run cross-validation (5-fold) to validate model stability across data splits.")
            recs.append("Investigate class imbalance — use SMOTE or class weighting if fault class is rare.")

        elif task_type == "Regression":
            r2 = metrics.get("R2", 1.0)
            if r2 < 0.6:
                recs.append("R² is low — consider engineering domain-specific features or using polynomial feature expansion.")
            recs.append("Plot residuals to check for heteroscedasticity (non-constant error variance).")

        elif task_type == "Clustering":
            recs.append("Experiment with different K values using the Elbow Method to find the optimal cluster count.")
            recs.append("Visualise clusters using PCA or t-SNE for interpretability.")

        elif task_type == "Time-Series":
            recs.append("Evaluate stationarity using the Augmented Dickey-Fuller test before applying ARIMA.")
            recs.append("Consider adding lagged features and rolling statistics for richer temporal context.")

        recs.append("Deploy the saved model via the FastAPI /predict endpoint for real-time inference.")
        recs.append("Schedule periodic retraining as new data arrives to prevent model drift.")

        return recs
