"""
ML_Guardian Orchestrator
Runs the full pipeline: prepare → train_xgb → train_mlp → train_transformer → hybrid → explain → dashboard
"""

import argparse
import subprocess
import sys
import os

def run_cmd(label, cmd):
    """Helper to run subprocess with clean logs and error handling."""
    print(f"\n{'='*70}\n>>> {label}\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"[SUCCESS] {label} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {label} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

# ----------------------------------------------------------

def run_prepare(args):
    run_cmd("STEP 1: Feature Extraction",
            [sys.executable, "-m", "core.create_features", "--raw", args.raw, "--out", args.out])

def run_train_xgb(args):
    cmd = [sys.executable, "-m", "core.train_xgb", "--features", args.features, "--out", args.xgb_out]
    if args.no_calibrate:
        cmd.append("--no_calibrate")
    run_cmd("STEP 2: Train XGBoost Model", cmd)

def run_train_mlp(args):
    cmd = [sys.executable, "-m", "neural.train_mlp", "--features", args.features, "--out", args.mlp_out]
    if args.use_embeddings:
        cmd.append("--use_embeddings")
    run_cmd("STEP 3: Train MLP Model", cmd)

def run_train_transformer(args):
    cmd = [
        sys.executable, "-m", "neural.train_transformer",
        "--features", args.features,
        "--out", args.transformer_out
    ]
    run_cmd("STEP 4: Train Transformer Model", cmd)

def run_hybrid(args):
    cmd = [
        sys.executable, "-m", "neural.hybrid_fusion",
        "--features", args.features,
        "--xgb", args.xgb_pipeline,
        "--mlp", args.mlp_pipeline,
        "--transformer", args.transformer_dir,
        "--out", args.out
    ]
    run_cmd("STEP 5: Hybrid Fusion Ensemble", cmd)

def run_shap(args):
    cmd = [
        sys.executable, "-m", "explain.shap_analysis",
        "--pipeline", args.pipeline,
        "--features", args.features,
        "--out", args.out
    ]
    run_cmd("STEP 6: SHAP Explainability", cmd)

def run_dashboard(args):
    cmd = ["streamlit", "run", "explain/dashboard_app.py"]
    run_cmd("STEP 7: Launch Streamlit Dashboard", cmd)

# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ML_Guardian Orchestrator CLI")
    sub = parser.add_subparsers(dest="cmd")

    # ----- Prepare -----
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--raw", default="data/raw/fever_data/train.jsonl")
    p_prep.add_argument("--out", default="data/processed/features.parquet")

    # ----- XGBoost -----
    p_xgb = sub.add_parser("train_xgb")
    p_xgb.add_argument("--features", default="data/processed/features.parquet")
    p_xgb.add_argument("--xgb_out", default="models/xgb_pipeline.joblib")
    p_xgb.add_argument("--no_calibrate", action="store_true")

    # ----- MLP -----
    p_mlp = sub.add_parser("train_mlp")
    p_mlp.add_argument("--features", default="data/processed/features.parquet")
    p_mlp.add_argument("--mlp_out", default="models/mlp_checkpoint")
    p_mlp.add_argument("--use_embeddings", action="store_true")

    # ----- Transformer -----
    p_tr = sub.add_parser("train_transformer")
    p_tr.add_argument("--model_name", default="microsoft/deberta-base")
    p_tr.add_argument("--features", default="data/processed/features.parquet")
    p_tr.add_argument("--transformer_out", default="models/transformer_checkpoint")

    # ----- Hybrid -----
    p_h = sub.add_parser("hybrid")
    p_h.add_argument("--features", default="data/processed/features.parquet")
    p_h.add_argument("--xgb_pipeline", default="models/xgb_pipeline.joblib")
    p_h.add_argument("--mlp_pipeline", default="models/mlp_checkpoint.pipeline.joblib")
    p_h.add_argument("--transformer_dir", default="models/transformer_checkpoint")
    p_h.add_argument("--out", default="models/ensemble_meta.joblib")

    # ----- Explain -----
    p_sh = sub.add_parser("explain")
    p_sh.add_argument("--pipeline", default="models/xgb_pipeline.joblib")
    p_sh.add_argument("--features", default="data/processed/features.parquet")
    p_sh.add_argument("--out", default="mlenv/plots")

    # ----- Serve (Dashboard) -----
    sub.add_parser("serve")

    # ----- Run all sequentially -----
    sub.add_parser("all")

    args = parser.parse_args()

    if args.cmd == "prepare":
        run_prepare(args)
    elif args.cmd == "train_xgb":
        run_train_xgb(args)
    elif args.cmd == "train_mlp":
        run_train_mlp(args)
    elif args.cmd == "train_transformer":
        run_train_transformer(args)
    elif args.cmd == "hybrid":
        run_hybrid(args)
    elif args.cmd == "explain":
        run_shap(args)
    elif args.cmd == "serve":
        run_dashboard(args)
    elif args.cmd == "all":
        # full end-to-end pipeline
        print("\n[STARTING] Full ML_Guardian Pipeline...\n")
        
        # Create args object with default values for all steps
        class AllArgs:
            def __init__(self):
                # Prepare args - use smaller dataset by default to avoid disk space issues
                self.raw = "data/raw/fever_data/train_tiny.jsonl"
                self.out = "data/processed/features.parquet"
                
                # XGBoost args
                self.features = "data/processed/features.parquet"
                self.xgb_out = "models/xgb_pipeline.joblib"
                self.no_calibrate = False
                
                # MLP args
                self.mlp_out = "models/mlp_checkpoint"
                self.use_embeddings = False
                
                # Transformer args
                self.model_name = "microsoft/deberta-base"
                self.transformer_out = "models/transformer_checkpoint"
                
                # Hybrid args
                self.xgb_pipeline = "models/xgb_pipeline.joblib"
                self.mlp_pipeline = "models/mlp_checkpoint.pipeline.joblib"
                self.transformer_dir = "models/transformer_checkpoint"
                
                # Explain args
                self.pipeline = "models/xgb_pipeline.joblib"
        
        all_args = AllArgs()
        run_prepare(all_args)
        run_train_xgb(all_args)
        run_train_mlp(all_args)
        run_train_transformer(all_args)
        run_hybrid(all_args)
        run_shap(all_args)
        print("[SUCCESS] All steps completed successfully. To visualize: run 'python main.py serve'")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()