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

Industrial Relevance (Enterprise Industrial):
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
        Runs the full preprocessing pipeline with zero train-to-test data leakage:
        1. Deduplicates data
        2. Separates features (X) and target (y)
        3. Extracts datetime components
        4. Performs Train/Test split BEFORE fitting transformers (for supervised tasks)
        5. Fits Imputer, Encoders, and Scaler exclusively on training data
        6. Transforms test data using the fitted transformers
        7. Returns (X_train, X_test, y_train, y_test) or (X, None, None, None)

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

        # Step 3: Extract numerical date features from datetime columns
        X = self._extract_datetime_features(X)

        # Step 4: Encode target for Classification
        y, label_classes = self._encode_target(y)

        # Step 5: Supervised vs Unsupervised Branching
        if y is not None:
            # ── Train / Test Split FIRST to prevent data leakage ─────────────
            stratify = None
            if self.task_type == "Classification":
                val_counts = y.value_counts()
                test_count = int(np.ceil(len(y) * DEFAULT_TEST_SIZE))
                if len(y) >= 10 and (val_counts.min() >= 2) and (test_count >= len(val_counts)):
                    stratify = y
                else:
                    logger.warning("Class counts or sample size too small for stratified split — using unstratified split.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size   = DEFAULT_TEST_SIZE,
                random_state= DEFAULT_RANDOM_STATE,
                stratify    = stratify,
            )

            # Identify column groups on training data
            numerical_cols   = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            # Impute (fit on train, transform train and test)
            X_train, X_test, impute_summary = self._impute_train_test(X_train, X_test, numerical_cols, categorical_cols)

            # Encode categorical features (fit on train, align test)
            X_train, X_test, encode_summary = self._encode_train_test(X_train, X_test, categorical_cols)

            # Scale features (fit on train, transform train and test)
            X_train, X_test, feature_names = self._scale_train_test(X_train, X_test)

            self._summary = {
                "original_shape":      list(original_shape),
                "after_dedup_rows":    int(self.data.shape[0]),
                "duplicates_removed":  int(dupes_removed),
                "numerical_features":  numerical_cols,
                "categorical_features":categorical_cols,
                "imputation":          impute_summary,
                "encoding":            encode_summary,
                "scaling":             "StandardScaler fitted strictly on training data",
                "label_classes":       label_classes,
                "final_feature_count": int(X_train.shape[1]),
                "leakage_protection":  "Enabled — Train/Test split prior to fitting transformers",
            }

            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Features: {len(feature_names)}")
            return X_train, X_test, y_train, y_test

        else:
            # ── Clustering (Unsupervised) ────────────────────────────────────
            logger.info("Clustering task — fitting transformers on full dataset without split.")
            numerical_cols   = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            X, impute_summary = self._impute_single(X, numerical_cols, categorical_cols)
            X, encode_summary = self._encode_single(X, categorical_cols)
            X, feature_names  = self._scale_single(X)

            self._summary = {
                "original_shape":      list(original_shape),
                "after_dedup_rows":    int(self.data.shape[0]),
                "duplicates_removed":  int(dupes_removed),
                "numerical_features":  numerical_cols,
                "categorical_features":categorical_cols,
                "imputation":          impute_summary,
                "encoding":            encode_summary,
                "scaling":             "StandardScaler (zero mean, unit variance)",
                "label_classes":       [],
                "final_feature_count": int(X.shape[1]),
            }
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
        X = X.copy()
        for col in X.columns:
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

    def _impute_train_test(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, num_cols: list, cat_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        summary = {}
        X_train = X_train.copy()
        X_test = X_test.copy()

        num_cols = [c for c in num_cols if c in X_train.columns]
        cat_cols = [c for c in cat_cols if c in X_train.columns]

        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            X_test[num_cols]  = num_imputer.transform(X_test[num_cols])
            summary["numerical"] = "Median imputation fitted on train, applied to train & test"

        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols].astype(str))
            X_test[cat_cols]  = cat_imputer.transform(X_test[cat_cols].astype(str))
            summary["categorical"] = "Mode imputation fitted on train, applied to train & test"

        return X_train, X_test, summary

    def _impute_single(self, X: pd.DataFrame, num_cols: list, cat_cols: list) -> Tuple[pd.DataFrame, dict]:
        summary = {}
        X = X.copy()
        num_cols = [c for c in num_cols if c in X.columns]
        cat_cols = [c for c in cat_cols if c in X.columns]
        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
            summary["numerical"] = "Median imputation applied"
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols].astype(str))
            summary["categorical"] = "Mode imputation applied"
        return X, summary

    def _encode_train_test(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        summary = {}
        X_train = X_train.copy()
        X_test = X_test.copy()

        cat_cols = [c for c in cat_cols if c in X_train.columns]
        low_card_cols = []
        high_card_cols = []
        dropped_cols = []

        max_ohe_categories = 50
        n_rows = len(X_train)

        for col in cat_cols:
            n_unique = X_train[col].nunique()
            if n_unique > max_ohe_categories:
                high_card_cols.append(col)
            else:
                low_card_cols.append(col)

        # High cardinality / ID columns
        for col in high_card_cols:
            n_unique = X_train[col].nunique()
            col_lower = str(col).lower()
            if n_unique > 500 or n_unique > (n_rows * 0.5) or any(k in col_lower for k in ["id", "guid", "uuid", "hash", "serial"]):
                X_train = X_train.drop(columns=[col])
                X_test  = X_test.drop(columns=[col], errors="ignore")
                dropped_cols.append(col)
                logger.info(f"Dropped high-cardinality/ID column '{col}' ({n_unique} unique values)")
            else:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                # Map unseen test labels to -1
                test_vals = X_test[col].astype(str)
                known_classes = set(le.classes_)
                X_test[col] = test_vals.apply(lambda v: le.transform([v])[0] if v in known_classes else -1)
                logger.info(f"Label encoded column '{col}' ({n_unique} unique values)")

        # One-Hot Encode low-cardinality columns
        if low_card_cols:
            X_train = pd.get_dummies(X_train, columns=low_card_cols, drop_first=True)
            X_test  = pd.get_dummies(X_test, columns=low_card_cols, drop_first=True)

            # Align X_test columns to match X_train exactly
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            bool_cols_tr = X_train.select_dtypes(include=["bool"]).columns
            X_train[bool_cols_tr] = X_train[bool_cols_tr].astype(int)

            bool_cols_te = X_test.select_dtypes(include=["bool"]).columns
            X_test[bool_cols_te] = X_test[bool_cols_te].astype(int)

        summary["method"] = "Hybrid Encoding (One-Hot for <=50 categories, Label/Drop for high-cardinality)"
        summary["columns_encoded"] = low_card_cols
        if high_card_cols:
            summary["high_cardinality_columns"] = high_card_cols
        if dropped_cols:
            summary["dropped_id_columns"] = dropped_cols

        return X_train, X_test, summary

    def _encode_single(self, X: pd.DataFrame, cat_cols: list) -> Tuple[pd.DataFrame, dict]:
        summary = {}
        X = X.copy()
        cat_cols = [c for c in cat_cols if c in X.columns]
        low_card_cols = []
        high_card_cols = []
        dropped_cols = []

        max_ohe_categories = 50
        n_rows = len(X)

        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique > max_ohe_categories:
                high_card_cols.append(col)
            else:
                low_card_cols.append(col)

        for col in high_card_cols:
            n_unique = X[col].nunique()
            col_lower = str(col).lower()
            if n_unique > 500 or n_unique > (n_rows * 0.5) or any(k in col_lower for k in ["id", "guid", "uuid", "hash", "serial"]):
                X = X.drop(columns=[col])
                dropped_cols.append(col)
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

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

    def _scale_train_test(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
        feature_names = X_train.columns.tolist()

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns = feature_names,
            index   = X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns = feature_names,
            index   = X_test.index,
        )
        return X_train_scaled, X_test_scaled, feature_names

    def _scale_single(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        feature_names = X.columns.tolist()
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns = feature_names,
            index   = X.index,
        )
        return X_scaled, feature_names

