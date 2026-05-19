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
            stratify = y if self.task_type == "Classification" else None
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

    def _impute(self, X: pd.DataFrame, num_cols: list, cat_cols: list) -> Tuple[pd.DataFrame, dict]:
        summary = {}
        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
            summary["numerical"] = "Median imputation applied"
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
            summary["categorical"] = "Mode (most frequent) imputation applied"
        return X, summary

    def _encode_features(self, X: pd.DataFrame, cat_cols: list) -> Tuple[pd.DataFrame, dict]:
        summary = {}
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            # Convert bool columns produced by get_dummies to int
            bool_cols = X.select_dtypes(include=["bool"]).columns
            X[bool_cols] = X[bool_cols].astype(int)
            summary["method"] = "One-Hot Encoding (pd.get_dummies)"
            summary["columns_encoded"] = cat_cols
        return X, summary

    def _encode_target(self, y: Optional[pd.Series]) -> Tuple[Optional[pd.Series], list]:
        if y is None or self.task_type != "Classification":
            return y, []
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
        return y_encoded, list(le.classes_)

    def _scale(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        feature_names = X.columns.tolist()
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns = feature_names,
            index   = X.index,
        )
        return X_scaled, feature_names
