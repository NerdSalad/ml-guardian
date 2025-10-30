# scripts/tuner_optuna.py
import os, yaml, optuna, logging
import numpy as np
import pandas as pd
from core.config import load_config
from core.metrics import classification_metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
cfg = load_config()

SEED = cfg["seed"]
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === helper ===
def update_yaml(best_params, section):
    with open("configs/default.yaml", "r") as f:
        y = yaml.safe_load(f)
    y[section].update(best_params)
    with open("configs/default.yaml", "w") as f:
        yaml.dump(y, f, sort_keys=False)
    logging.info(f"[optuna] Updated best params in configs/default.yaml under [{section}]")

# === XGBoost objective ===
def objective_xgb(trial):
    df = pd.read_parquet(cfg["paths"]["processed_features"])
    X = df.drop(columns=["label"]).values
    y = df["label"].astype(int).values
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "n_jobs": -1,
        "random_state": SEED,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    f1_scores = []
    for tr, va in skf.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                  early_stopping_rounds=40, verbose=False)
        preds = model.predict(X[va])
        m = classification_metrics(y[va], preds)
        f1_scores.append(m["f1_macro"])
    return np.mean(f1_scores)

# === MLP objective ===
def build_mlp(input_dim, hidden_sizes, dropout, lr, num_classes):
    model = Sequential()
    model.add(Dense(hidden_sizes[0], activation="relu", input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    if len(hidden_sizes) > 1:
        for h in hidden_sizes[1:]:
            model.add(Dense(h, activation="relu"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def objective_mlp(trial):
    df = pd.read_parquet(cfg["paths"]["processed_features"])
    X = df.drop(columns=["label"]).values
    y = df["label"].astype(int).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15,
                                              stratify=y, random_state=SEED)
    hidden_layers = trial.suggest_categorical("hidden_layers", [(512,256), (256,128), (256,), (512,)])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64,128,256])
    epochs = 15

    model = build_mlp(X_tr.shape[1], hidden_layers, dropout, lr, len(np.unique(y)))
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch_size, verbose=0)
    preds = np.argmax(model.predict(X_va, verbose=0), axis=1)
    m = classification_metrics(y_va, preds)
    return m["f1_macro"]

# === main ===
def main():
    logging.info("=== Optuna Tuner ===")
    logging.info("1️⃣  Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objective_xgb, n_trials=25, timeout=None)
    logging.info(f"[xgb] Best value: {study_xgb.best_value:.4f}")
    logging.info(f"[xgb] Best params: {study_xgb.best_params}")
    update_yaml(study_xgb.best_params, "xgb")

    logging.info("2️⃣  Tuning MLP...")
    study_mlp = optuna.create_study(direction="maximize")
    study_mlp.optimize(objective_mlp, n_trials=25, timeout=None)
    logging.info(f"[mlp] Best value: {study_mlp.best_value:.4f}")
    logging.info(f"[mlp] Best params: {study_mlp.best_params}")
    update_yaml(study_mlp.best_params, "mlp")

    logging.info("✅ Tuning complete. Best parameters saved to configs/default.yaml")

if __name__ == "__main__":
    main()