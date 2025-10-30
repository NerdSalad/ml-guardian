# core/persistence.py
import joblib
import os
from typing import Any

def save_pipeline(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(obj, path)
    return path

def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline not found: {path}")
    return joblib.load(path)