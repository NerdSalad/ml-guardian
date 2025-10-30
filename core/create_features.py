# core/create_features.py
"""
Create feature parquet from raw FEVER-style jsonl.
Saves to data/processed/features.parquet
"""
import numpy as np
import os
import json
import argparse
import pandas as pd
from pathlib import Path

from core.feature_engineer_optimized import OptimizedFeatureEngineer

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def prepare_dataframe(raw_rows):
    # FEVER format varies: ensure claim and evidence exist
    records = []
    for r in raw_rows:
        claim = r.get("claim") or r.get("sentence") or ""
        label = r.get("label", None)
        # evidence may be list-of-lists in FEVER; flatten text if present
        evidence_text = ""
        ev = r.get("evidence") or r.get("evidence_text") or r.get("evidence_sentences")
        if isinstance(ev, list):
            # if list of lists, flatten
            if ev and isinstance(ev[0], list):
                ev_texts = []
                for part in ev:
                    try:
                        # sometimes (doc_id, sent_id, text)
                        for p in part:
                            if isinstance(p, str):
                                ev_texts.append(p)
                    except Exception:
                        pass
                evidence_text = " ".join(ev_texts)
            else:
                # list of strings
                evidence_text = " ".join([str(x) for x in ev if x])
        else:
            evidence_text = ev or ""

        records.append({
            "claim": claim,
            "evidence": evidence_text,
            "label": label
        })
    return pd.DataFrame(records)

def main(raw_path, out_path, batch_size=16):
    print(f"[create_features] Loading raw: {raw_path}")
    rows = load_jsonl(raw_path)
    df = prepare_dataframe(rows)
    print(f"[create_features] Loaded {len(df)} rows; extracting features (this may take time)...")

    fe = OptimizedFeatureEngineer()
   # Extract features and embeddings safely
    feats_df, claim_emb, evid_emb = fe.batch_extract_with_embeddings(df, batch_size=batch_size)

    # Ensure label preserved and convert to integers
    if "label" in df.columns:
        # Convert string labels to integers using the label mapping
        from core.labels import REVERSE_LABEL_MAP
        feats_df["label"] = df["label"].map(REVERSE_LABEL_MAP).fillna(2).astype(int)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save the main features dataframe as parquet
    feats_df.to_parquet(out_path, index=False)
    
    # Save embeddings separately with error handling
    try:
        claim_emb_path = out_path.replace(".parquet", "_claim_emb.npy")
        evid_emb_path = out_path.replace(".parquet", "_evid_emb.npy")
        
        # Check available disk space
        import shutil
        free_space = shutil.disk_usage(os.path.dirname(out_path)).free
        required_space = claim_emb.nbytes + evid_emb.nbytes
        
        if required_space > free_space:
            print(f"[WARNING] Insufficient disk space. Required: {required_space/1024/1024:.1f}MB, Available: {free_space/1024/1024:.1f}MB")
            print("[INFO] Skipping embedding save due to disk space constraints")
        else:
            np.save(claim_emb_path, claim_emb)
            np.save(evid_emb_path, evid_emb)
            print(f"[create_features] Saved embeddings to: {claim_emb_path}, {evid_emb_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save embeddings: {e}")
        print("[INFO] Continuing without embedding files")
    
    print(f"[create_features] Saved features to: {out_path}")
    return feats_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw/fever_data/train.jsonl")
    parser.add_argument("--out", type=str, default="data/processed/features.parquet")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding extraction")
    args = parser.parse_args()
    main(args.raw, args.out, args.batch_size)