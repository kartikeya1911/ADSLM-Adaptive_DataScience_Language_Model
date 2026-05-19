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


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types.
    FastAPI / json.dumps cannot serialize np.float32, np.int64, etc. by default.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def sanitize_for_json(data: Any) -> Any:
    """
    Recursively converts a nested dict/list containing NumPy or Pandas
    scalars into plain Python types so FastAPI can serialise it.
    """
    return json.loads(json.dumps(data, cls=NumpyEncoder))


def ensure_dir(path: str | Path) -> Path:
    """Creates directory (and parents) if not present. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
