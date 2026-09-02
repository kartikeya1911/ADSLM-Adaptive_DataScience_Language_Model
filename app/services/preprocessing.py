"""
app/services/preprocessing.py
===============================
Module  : Preprocessing Engine
Purpose : Full automated preprocessing pipeline — imputation, encoding, scaling, splitting.

Steps:
    1. Remove duplicates
    2. Separate features (X) and target (y)
    3. Impute missing values (median for numerical, mode for categorical)
    4. Encode categorical features (One-Hot Encoding via pd.get_dummies)
    5. Encode target if Classification (LabelEncoder)
    6. Scale features (StandardScaler)
    7. Train / Test Split (80/20, stratified for classification)

Industrial Relevance (ABB):
    - Sensor readings often have NaN gaps during downtime → auto-imputed
    - Mixed numeric/categorical machine parameter data → auto-encoded
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Tuple

from app.core.config import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingEngine:
    """
    End-to-end preprocessing pipeline for any ML task.

    Usage:
        engine = PreprocessingEngine(df, target_column="Fault", task_type="Classification")
        X_train, X_test, y_train, y_test = engine.preprocess()
        summary = engine.get_summary()
    """

    def __init__(self, data: pd.DataFrame, target_column: Optional[str], task_type: str):
        """
        Args:
            data          : Raw pandas DataFrame.
            target_column : Column to predict (None for Clustering).
            task_type     : One of Regression / Classification / Clustering / Time-Series.
        """
        self.data        = data.copy()
        self.target      = target_column
        self.task_type   = task_type
        self._summary    = {}   # Populated during preprocess()

    # ── Public API ────────────────────────────────────────────────────────────

    def preprocess(self) -> Tuple:
        """
        Runs the full preprocessing pipeline.

        Returns:
            (X_train, X_test, y_train, y_test) for supervised tasks,
            (X, None, None, None) for Clustering.
        """
        logger.info(f"Starting preprocessing for task: {self.task_type}")
        original_shape = self.data.shape

        # Step 1: Remove duplicates
        self.data = self.data.drop_duplicates()
        dupes_removed = original_shape[0] - self.data.shape[0]
        logger.info(f"Duplicates removed: {dupes_removed}")

        # Step 2: Separate X and y
        X, y = self._split_features_target()

        # Extract numerical date features from datetime columns
        X = self._extract_datetime_features(X)

        # Identify column groups
        numerical_cols  = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        categorical_cols= X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Step 3: Impute missing values
        X, impute_summary = self._impute(X, numerical_cols, categorical_cols)

        # Step 4: Encode categorical features
        X, encode_summary = self._encode_features(X, categorical_cols)

        # Step 5: Encode target for Classification
        y, label_classes = self._encode_target(y)

        # Step 6: Scale features
        X, feature_names = self._scale(X)

        # Build summary for the UI
        self._summary = {
            "original_shape":     list(original_shape),
            "after_dedup_rows":   int(self.data.shape[0]),
            "duplicates_removed": int(dupes_removed),
            "numerical_features": numerical_cols,
            "categorical_features": categorical_cols,
            "imputation":         impute_summary,
            "encoding":           encode_summary,
            "scaling":            "StandardScaler (zero mean, unit variance)",
            "label_classes":      label_classes,
            "final_feature_count": int(X.shape[1]),
        }

        # Step 7: Split
        if y is not None:
            stratify = None
            if self.task_type == "Classification":
                val_counts = y.value_counts()
                if len(y) >= 5 and (val_counts.min() >= 2):
                    stratify = y
                else:
                    logger.warning("Class counts too small for stratified split — using unstratified split.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size   = DEFAULT_TEST_SIZE,
                random_state= DEFAULT_RANDOM_STATE,
                stratify    = stratify,
            )
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            return X_train, X_test, y_train, y_test
        else:
            logger.info("Clustering task — returning full X without split.")
            return X, None, None, None

    def get_summary(self) -> dict:
        """Returns a human-readable summary of what preprocessing did."""
        return self._summary

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _split_features_target(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.target and self.target in self.data.columns:
            y = self.data[self.target].reset_index(drop=True)
            X = self.data.drop(columns=[self.target]).reset_index(drop=True)
        else:
            y = None
            X = self.data.reset_index(drop=True)
        return X, y

    def _extract_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extracts numeric year/month/day/hour features from datetime columns."""
        for col in X.columns:
            # Check if column is already datetime or looks like date strings
            dt_series = None
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                dt_series = X[col]
            elif X[col].dtype == "object":
                try:
                    sample = X[col].dropna().head(20).astype(str)
                    if sample.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}").mean() > 0.5:
                        dt_series = pd.to_datetime(X[col], errors="coerce")
                except Exception:
                    pass

            if dt_series is not None:
                X[f"{col}_year"]  = dt_series.dt.year.fillna(-1).astype(int)
                X[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
                X[f"{col}_day"]   = dt_series.dt.day.fillna(-1).astype(int)
                X[f"{col}_hour"]  = dt_series.dt.hour.fillna(-1).astype(int)
                X = X.drop(columns=[col])
        return X

    def _impute(self, X: pd.DataFrame, num_cols: list, cat_cols: list) -> Tuple[pd.DataFrame, dict]:
        summary = {}
        # Re-check columns currently in X
        num_cols = [c for c in num_cols if c in X.columns]
        cat_cols = [c for c in cat_cols if c in X.columns]
        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
            summary["numerical"] = "Median imputation applied"
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols].astype(str))
            summary["categorical"] = "Mode (most frequent) imputation applied"
        return X, summary

    def _encode_features(self, X: pd.DataFrame, cat_cols: list) -> Tuple[pd.DataFrame, dict]:
        summary = {}
        cat_cols = [c for c in cat_cols if c in X.columns]
        low_card_cols = []
        high_card_cols = []
        dropped_cols = []

        max_ohe_categories = 50
        n_rows = len(X)

        for col in cat_cols:
            n_unique = X[col].nunique()
            col_lower = str(col).lower()

            if n_unique > max_ohe_categories:
                high_card_cols.append(col)
            else:
                low_card_cols.append(col)

        # Handle high cardinality columns via Label Encoding or Drop (if ID/Hash like)
        for col in high_card_cols:
            n_unique = X[col].nunique()
            col_lower = str(col).lower()
            if n_unique > 500 or n_unique > (n_rows * 0.5) or any(k in col_lower for k in ["id", "guid", "uuid", "hash", "serial"]):
                X = X.drop(columns=[col])
                dropped_cols.append(col)
                logger.info(f"Dropped high-cardinality/ID column '{col}' ({n_unique} unique values)")
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                logger.info(f"Label encoded high-cardinality column '{col}' ({n_unique} unique values)")

        # One-Hot Encode low-cardinality categorical columns (<= 50 unique values)
        if low_card_cols:
            X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)
            bool_cols = X.select_dtypes(include=["bool"]).columns
            X[bool_cols] = X[bool_cols].astype(int)

        summary["method"] = "Hybrid Encoding (One-Hot for <=50 categories, Label/Drop for high-cardinality)"
        summary["columns_encoded"] = low_card_cols
        if high_card_cols:
            summary["high_cardinality_columns"] = high_card_cols
        if dropped_cols:
            summary["dropped_id_columns"] = dropped_cols

        return X, summary

    def _encode_target(self, y: Optional[pd.Series]) -> Tuple[Optional[pd.Series], list]:
        if y is None or self.task_type != "Classification":
            return y, []
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
        return y_encoded, list(le.classes_)

    def _scale(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        # Ensure all columns are converted to numbers and boolean dummies are ints
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        feature_names = X.columns.tolist()
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns = feature_names,
            index   = X.index,
        )
        return X_scaled, feature_names
