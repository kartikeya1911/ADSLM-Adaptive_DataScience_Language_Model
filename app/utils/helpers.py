"""
app/utils/helpers.py
====================
General-purpose helper functions shared across ADSLM modules.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict


import math
from pathlib import Path
from typing import Any, Dict, List


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively converts any data structure containing NumPy, Pandas,
    NaN, Infinity, or custom objects into strictly valid, standard JSON-compliant Python types.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return sanitize_for_json(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, pd.Timedelta, Path)):
        return str(obj)
    try:
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return float(obj)
    except (ValueError, TypeError):
        pass
    return str(obj)


def ensure_dir(path: str | Path) -> Path:
    """Creates directory (and parents) if not present. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
