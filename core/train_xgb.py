# Add parent dir to sys.path for module-safe imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# core/train_xgb.py
import os, argparse, logging
import numpy as np
import pandas as pd
from core.config import cfg
from core.labels import LABEL_MAP
from core.persistence import save_pipeline
from core.metrics import classification_metrics
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from collections import Counter

def set_seed(seed):
    import random; random.seed(seed)
    np.random.seed(seed)

def main(features_path=None, out_path=None):
    features_path = features_path or cfg["paths"]["processed_features"]
    out_path = out_path or cfg["paths"]["xgb_model"]
    logging.info(f"[train_xgb] Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    if "label" not in df:
        raise ValueError("Need 'label' column in features parquet")
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values

    set_seed(cfg["seed"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    params = dict(
        n_estimators=cfg["xgb"]["n_estimators"],
        max_depth=cfg["xgb"]["max_depth"],
        learning_rate=cfg["xgb"]["learning_rate"],
        subsample=cfg["xgb"]["subsample"],
        colsample_bytree=cfg["xgb"]["colsample_bytree"],
        reg_lambda=cfg["xgb"].get("reg_lambda", 1.0)
    )

    # OOF CV
    logging.info("[train_xgb] Running 5-fold OOF CV...")
    oof_preds = np.zeros_like(y)
    val_probas = np.zeros((len(y), len(LABEL_MAP)))
    fold = 0
    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        logging.info(f"[train_xgb] Fold {fold}")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric=cfg["xgb"]["eval_metric"], n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  verbose=False)
        preds = model.predict(X_va)
        oof_preds[val_idx] = preds
        try:
            val_probas[val_idx] = model.predict_proba(X_va)
        except Exception:
            # fallback: one-hot
            for i, p in enumerate(preds):
                val_probas[val_idx[i], p] = 1.0

    metrics = classification_metrics(y, oof_preds, labels=list(LABEL_MAP.keys()))
    logging.info(f"[train_xgb] CV metrics: {metrics}")

    # Final model trained on all data. Use parameters and n_estimators from CV: use num_boost_round ~ best_iteration
    logging.info("[train_xgb] Training final model on full data...")
    final_model = XGBClassifier(**params, use_label_encoder=False, eval_metric=cfg["xgb"]["eval_metric"], n_jobs=-1)
    final_model.fit(X, y, verbose=False)
    # Save pipeline: include feature names & label map
    pipeline = {
        "model": final_model,
        "feature_names": list(df.drop(columns=["label"]).columns),
        "label_map": LABEL_MAP,
        "config": cfg
    }
    save_pipeline(pipeline, out_path)
    logging.info(f"[train_xgb] Saved pipeline to {out_path}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    main(args.features, args.out)