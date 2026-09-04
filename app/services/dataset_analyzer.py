"""
app/services/dataset_analyzer.py
=================================
Module  : Dataset Analyzer
Purpose : Automatically reads and profiles any uploaded CSV dataset.
          Detects shape, dtypes, nulls, outliers, duplicates, and potential targets.

Industrial Relevance (Enterprise Industrial):
    - Sensor data ingestion profiling
    - Predictive maintenance dataset health checks
    - Manufacturing quality inspection data auditing
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List

from app.utils.logger import get_logger
from app.utils.helpers import sanitize_for_json

logger = get_logger(__name__)


class DatasetAnalyzer:
    """
    Profiles a dataset end-to-end and returns a comprehensive analysis dictionary.

    Usage:
        analyzer = DatasetAnalyzer(file_buffer)
        stats = analyzer.analyze()
    """

    def __init__(self, file_path_or_buffer):
        """
        Args:
            file_path_or_buffer: A file path string OR a file-like buffer (e.g., io.StringIO).
        """
        try:
            self.data: pd.DataFrame = pd.read_csv(file_path_or_buffer)
            logger.info(f"Dataset loaded: {self.data.shape[0]} rows × {self.data.shape[1]} cols")
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise ValueError(f"Could not read dataset. Ensure it is a valid CSV file. Detail: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline.

        Returns:
            dict with keys:
                basic_info, missing_values, column_types,
                summary_statistics, outlier_info, potential_targets
        """
        logger.info("Running full dataset analysis …")
        result = {
            "basic_info":          self._get_basic_info(),
            "missing_values":      self._get_missing_values(),
            "column_types":        self._get_column_types(),
            "summary_statistics":  self._get_summary_statistics(),
            "outlier_info":        self._detect_outliers(),
            "potential_targets":   self._identify_potential_targets(),
        }
        return sanitize_for_json(result)

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _get_basic_info(self) -> Dict[str, Any]:
        """Returns shape, column names, dtypes, and duplicate count."""
        return {
            "row_count":      int(self.data.shape[0]),
            "column_count":   int(self.data.shape[1]),
            "columns":        self.data.columns.tolist(),
            "dtypes":         {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "duplicate_rows": int(self.data.duplicated().sum()),
            "memory_mb":      round(self.data.memory_usage(deep=True).sum() / 1e6, 3),
        }

    def _get_missing_values(self) -> Dict[str, Any]:
        """Returns per-column missing value counts and overall missing percentage."""
        missing_counts = self.data.isnull().sum()
        missing_pct    = (missing_counts / len(self.data) * 100).round(2)

        result = {
            col: {"count": int(missing_counts[col]), "percent": float(missing_pct[col])}
            for col in missing_counts.index
            if missing_counts[col] > 0
        }

        total_cells = self.data.size
        total_missing = int(self.data.isnull().sum().sum())
        result["_summary"] = {
            "total_missing_cells": total_missing,
            "overall_missing_pct": round(total_missing / total_cells * 100, 2) if total_cells else 0,
        }
        return result

    def _get_column_types(self) -> Dict[str, List[str]]:
        """
        Classifies columns into numerical, categorical, and datetime.
        Includes a heuristic for string columns that look like dates.
        """
        numerical_cols  = self.data.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols   = self.data.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Heuristic: detect date-like string columns
        for col in self.data.select_dtypes(include=["object"]).columns:
            try:
                sample = self.data[col].dropna().head(50).astype(str)
                if sample.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}").mean() > 0.5:
                    datetime_cols.append(col)
            except Exception:
                pass

        categorical_cols = [
            col for col in self.data.columns
            if col not in numerical_cols and col not in datetime_cols
        ]

        return {
            "numerical":   numerical_cols,
            "categorical": categorical_cols,
            "datetime":    datetime_cols,
        }

    def _get_summary_statistics(self) -> Dict[str, Any]:
        """Returns describe() statistics for both numerical and categorical columns."""
        num_stats = {}
        cat_stats = {}

        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            num_stats = self.data[numerical_cols].describe().round(4).to_dict()
            for col in numerical_cols:
                num_stats[col]["nunique"] = int(self.data[col].nunique())

        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            cat_stats = {
                col: {
                    "unique":    int(self.data[col].nunique()),
                    "top_value": str(self.data[col].mode()[0]) if not self.data[col].mode().empty else "N/A",
                    "frequency": int(self.data[col].value_counts().iloc[0]) if not self.data[col].value_counts().empty else 0,
                }
                for col in cat_cols
            }

        return {"numerical": num_stats, "categorical": cat_stats}

    def _detect_outliers(self) -> Dict[str, Any]:
        """
        Uses the IQR method to detect outliers in numerical columns.
        Industrial use: sensor spike detection, anomalous machine readings.
        """
        outlier_report = {}
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            series = self.data[col].dropna()
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR    = Q3 - Q1
            lower  = Q1 - 1.5 * IQR
            upper  = Q3 + 1.5 * IQR
            n_out  = int(((series < lower) | (series > upper)).sum())

            if n_out > 0:
                outlier_report[col] = {
                    "outlier_count": n_out,
                    "lower_bound":   round(float(lower), 4),
                    "upper_bound":   round(float(upper), 4),
                }

        outlier_report["_summary"] = {"columns_with_outliers": len(outlier_report) - 1}
        return outlier_report

    def _identify_potential_targets(self) -> List[str]:
        """
        Heuristic to suggest likely target/label columns.
        Checks for common naming patterns, then falls back to the last column.
        """
        target_hints = ["target", "label", "status", "class", "output", "y", "fault",
                        "result", "failure", "defect", "anomaly", "prediction"]
        lower_cols   = self.data.columns.str.lower()

        potentials = [
            self.data.columns[i]
            for i, col in enumerate(lower_cols)
            if any(hint in col for hint in target_hints)
        ]

        # Fallback: suggest the last column (common in datasets)
        if not potentials and len(self.data.columns) > 0:
            potentials.append(self.data.columns[-1])

        return potentials
