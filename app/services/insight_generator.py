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

Industrial Relevance (Enterprise Industrial):
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
        if not best_model or str(best_model).lower() in ("none", "n/a"):
            return "Model training was completed, but no single winning model could be selected."

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
        task_type: str,
        metrics: Dict[str, Any],
        prep_summary: Dict[str, Any],
        analysis: Dict[str, Any] = None,
        top_features: List[Dict[str, Any]] = None,
        regulatory_audit: Dict[str, Any] = None,
        big_data_profile: Dict[str, Any] = None,
    ) -> List[str]:
        """Returns a dynamic, dataset-specific list of actionable next-step recommendations."""
        recs = []

        # 1. Feature-specific recommendation
        if top_features and len(top_features) > 0:
            top_names = [f"'{f['feature']}'" for f in top_features[:2]]
            recs.append(f"Focus sensor calibration and quality assurance on top predictive features: {', '.join(top_names)}.")

        # 2. Data Health & Preprocessing recommendations
        if analysis:
            mv = {k: v for k, v in analysis.get("missing_values", {}).items() if k != "_summary"}
            if mv:
                top_miss = list(mv.keys())[:2]
                recs.append(f"Address missing data in columns {top_miss} using automated sensor telemetry interpolation.")

            oi = {k: v for k, v in analysis.get("outlier_info", {}).items() if k != "_summary"}
            if oi:
                top_out = list(oi.keys())[:2]
                recs.append(f"Investigate extreme sensor spikes detected in columns {top_out} via IQR bounds.")

        if prep_summary and prep_summary.get("dropped_id_columns"):
            recs.append(f"High-cardinality ID columns {prep_summary['dropped_id_columns']} were safely excluded from feature scaling.")

        # 3. Task & Performance recommendations
        if task_type == "Classification":
            f1 = metrics.get("F1-score", 1.0)
            acc = metrics.get("Accuracy", 1.0)
            if f1 >= 0.85:
                recs.append(f"Model achieved strong performance (F1-score: {f1:.4f}, Accuracy: {acc:.2%}); set up model drift monitoring in production.")
            else:
                recs.append(f"F1-score ({f1:.4f}) indicates room for improvement; apply SMOTE oversampling or hyperparameter tuning with Optuna.")

        elif task_type == "Regression":
            r2 = metrics.get("R2", 0.0)
            rmse = metrics.get("RMSE", 0.0)
            if r2 >= 0.7:
                recs.append(f"Regression model explains {r2*100:.1f}% of variance (RMSE: {rmse:.4f}); ready for load forecasting.")
            else:
                recs.append(f"R² ({r2:.4f}) is low; consider engineering non-linear feature interactions.")

        elif task_type == "Clustering":
            sil = metrics.get("Silhouette Score", 0.0)
            recs.append(f"Evaluate cluster cohesion (Silhouette Score: {sil:.4f}) using PCA 2D projections.")

        elif task_type == "Time-Series":
            rmse = metrics.get("RMSE", 0.0)
            recs.append(f"Validate time-series forecasting (RMSE: {rmse:.4f}) against rolling shift schedules.")

        # 4. Regulatory & Governance recommendations
        if regulatory_audit:
            gdpr = regulatory_audit.get("gdpr_audit", {})
            if gdpr.get("pii_detected"):
                pii_cols = [f["column"] for f in gdpr.get("flagged_pii_columns", [])]
                recs.append(f"GDPR Action Required: Anonymize or hash PII in columns {pii_cols} before deployment.")

            eu_ai = regulatory_audit.get("eu_ai_act", {})
            if eu_ai.get("risk_badge") == "HIGH_RISK":
                recs.append("EU AI Act High-Risk System: Implement human-in-the-loop approval workflows for automated asset control.")

        # 5. Big Data Recommendation
        if big_data_profile:
            part = big_data_profile.get("partitioning_strategy", {}).get("primary_partition_column")
            if part:
                recs.append(f"Partition Delta Lake tables on column '{part}' for sub-second distributed telemetry queries.")

        # 6. Production Deployment
        recs.append("Deploy the saved model binary via the FastAPI /orchestrate endpoint for production inference.")

        return recs
