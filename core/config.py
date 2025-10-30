# core/config.py
"""
Configuration settings for ML_Guardian project
"""

cfg = {
    "seed": 42,
    "paths": {
        "raw_data": "data/raw/fever_data/train.jsonl",
        "processed_features": "data/processed/features.parquet",
        "xgb_model": "models/xgb_pipeline.joblib",
        "mlp_model": "models/mlp_checkpoint",
        "transformer_model": "models/transformer_checkpoint",
        "transformer_dir": "models/transformer_checkpoint",
        "transformer_preds": "models/transformer_preds.npy",
        "claim_emb": "data/processed/features_claim_emb.npy",
        "evid_emb": "data/processed/features_evid_emb.npy",
        "fusion_model": "models/ensemble_meta.joblib",
        "ensemble_model": "models/ensemble_meta.joblib",
        "plots": "mlvenv/plots"
    },
    "xgb": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "eval_metric": "mlogloss",
        "early_stopping_rounds": 50
    },
    "mlp": {
        "hidden_sizes": [512, 256, 128],
        "dropout": 0.3,
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10
    },
    "transformer": {
        "model_name": "microsoft/deberta-base",
        "max_length": 512,
        "lr": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "warmup_steps": 500
    },
    "feature_engineering": {
        "use_embeddings": True,
        "embedding_model": "all-MiniLM-L6-v2",
        "max_evidence_length": 2000,
        "text_features": True,
        "statistical_features": True,
        "semantic_features": True
    }
}

