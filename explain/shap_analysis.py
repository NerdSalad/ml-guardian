# explain/shap_analysis.py
import os, argparse, logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from core.config import cfg
from core.persistence import load_pipeline
import shap
import matplotlib.pyplot as plt

def main(model_path=None, features_path=None, out_dir=None):
    model_path = model_path or cfg["paths"]["xgb_model"]
    features_path = features_path or cfg["paths"]["processed_features"]
    out_dir = out_dir or "mlenv/plots"
    os.makedirs(out_dir, exist_ok=True)
    logging.info("[shap] Loading pipeline and features...")
    pipeline = load_pipeline(model_path)
    model = pipeline["model"]
    feature_names = pipeline.get("feature_names", None)
    df = pd.read_parquet(features_path)
    X = df.drop(columns=["label"])
    # Use TreeExplainer for xgboost classifier
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For multiclass shap_values is list-like per class
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "shap_summary.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"[shap] Saved summary plot to {out_path}")
    except Exception as e:
        logging.error(f"[shap] TreeExplainer failed: {e}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    main(args.model, args.features, args.out)