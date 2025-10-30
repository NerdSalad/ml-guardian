# neural/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
from core.config import cfg
import logging

def get_sbert_model(name=None):
    name = name or "all-MiniLM-L6-v2"
    try:
        model = SentenceTransformer(name)
        return model
    except Exception as e:
        logging.error(f"SentenceTransformer load failed: {e}")
        raise

def compute_and_save_embeddings(df, fe, out_claim_path, out_evid_path, batch_size=256):
    """
    fe: OptimizedFeatureEngineer instance with method to return embeddings or we call SBERT directly.
    But here we compute SBERT embeddings from raw text (df should have 'claim' and 'evidence').
    """
    model = get_sbert_model(cfg.get("sbert_model", None))
    claims = df["claim"].astype(str).tolist()
    evids = df["evidence"].astype(str).tolist()
    logging.info("[embeddings] computing claim embeddings...")
    claim_emb = model.encode(claims, batch_size=batch_size, show_progress_bar=True)
    logging.info("[embeddings] computing evidence embeddings...")
    evid_emb = model.encode(evids, batch_size=batch_size, show_progress_bar=True)
    claim_emb = np.asarray(claim_emb, dtype="float32")
    evid_emb = np.asarray(evid_emb, dtype="float32")
    np.save(out_claim_path, claim_emb)
    np.save(out_evid_path, evid_emb)
    return claim_emb, evid_emb