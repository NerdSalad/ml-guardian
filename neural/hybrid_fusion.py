import os, logging, argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from core.config import cfg
from core.persistence import save_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main(features_path=None, xgb_pipeline_path=None, mlp_wrapper_path=None, transformer_preds_path=None, out_path=None):
    features_path = features_path or cfg["paths"]["processed_features"]
    xgb_pipeline_path = xgb_pipeline_path or cfg["paths"]["xgb_model"]
    mlp_wrapper_path = mlp_wrapper_path or cfg["paths"]["mlp_model"]+".joblib"
    transformer_preds_path = transformer_preds_path or cfg["paths"]["transformer_preds"]
    out_path = out_path or cfg["paths"]["fusion_model"]
    logging.info("[hybrid_fusion] Loading base outputs and features...")
    import joblib
    xgb_pipe = joblib.load(xgb_pipeline_path)
    df = pd.read_parquet(features_path)
    y = df["label"].astype(int).values
    X_features = df.drop(columns=["label"]).values

    # get XGBoost predictions (probabilities)
    xgb_model = xgb_pipe["model"]
    xgb_probs = xgb_model.predict_proba(X_features)

    # get MLP predictions if available
    mlp_wrapper = None
    if os.path.exists(mlp_wrapper_path):
        mlp_wrapper = joblib.load(mlp_wrapper_path)
        import tensorflow as tf
        scaler = mlp_wrapper["scaler"]
        model_path = mlp_wrapper["model_path"]
        use_emb = mlp_wrapper.get("use_embeddings", True)
        # load keras
        from tensorflow.keras.models import load_model
        keras_model = load_model(model_path)
        X_scaled = scaler.transform(X_features)
        if use_emb:
            claim_emb = np.load(cfg["paths"]["claim_emb"])
            evid_emb = np.load(cfg["paths"]["evid_emb"])
            emb = np.concatenate([claim_emb, evid_emb], axis=1)
            preds_mlp = keras_model.predict([X_scaled, emb], batch_size=cfg["mlp"]["batch_size"])
        else:
            preds_mlp = keras_model.predict(X_scaled, batch_size=cfg["mlp"]["batch_size"])
    else:
        preds_mlp = np.zeros((len(X_features), len(xgb_probs[0])))

    # load transformer probabilities if available (validation probs align)
    if os.path.exists(transformer_preds_path):
        transformer_probs = np.load(transformer_preds_path)
        # Ensure shape matches; else tile or truncate
        if transformer_probs.shape[0] != len(X_features):
            # transformer preds are likely from validation only; we will create stacking on whole dataset by splitting
            logging.warning("[hybrid_fusion] transformer preds length differs from features; proceeding with available preds for stacking training only.")
            # We'll stack only where transformer preds are available: assume last N rows correspond to validation, but safest is to build simple stacker on XGB+MLP using train_test_split
            transformer_probs = None
    else:
        transformer_probs = None

    # Build meta-features: concat xgb_probs + mlp_probs (+ transformer if available)
    meta = np.concatenate([xgb_probs, preds_mlp], axis=1)
    if transformer_probs is not None:
        if transformer_probs.shape[0] == meta.shape[0]:
            meta = np.concatenate([meta, transformer_probs], axis=1)

    # Train a simple logistic regression on a holdout split
    X_tr, X_va, y_tr, y_va = train_test_split(meta, y, test_size=0.15, stratify=y, random_state=cfg["seed"])
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(X_tr_s, y_tr)
    preds = stacker.predict(X_va_s)
    acc = accuracy_score(y_va, preds)
    logging.info(f"[hybrid_fusion] Validation accuracy of stacker: {acc:.4f}")

    # save final stacker + scaler + base model refs
    pipeline = {
        "stacker": stacker,
        "scaler": scaler,
        "xgb_pipeline": xgb_pipeline_path,
        "mlp_wrapper": mlp_wrapper_path,
        "transformer_preds": transformer_preds_path
    }
    save_pipeline(pipeline, out_path)
    logging.info(f"[hybrid_fusion] Saved fusion pipeline to {out_path}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None)
    parser.add_argument("--xgb", default=None)
    parser.add_argument("--mlp", default=None)
    parser.add_argument("--transformer_preds", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.features, args.xgb, args.mlp, args.transformer_preds, args.out)