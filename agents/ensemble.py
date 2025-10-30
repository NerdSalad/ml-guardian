# ml_guardian/agents/ensemble.py
"""
EnsembleGuardian — original agent-level meta-classifier ensemble.
Also includes HallucinationPipelineHelper — lightweight wrapper for the
advanced hallucination detection pipeline (feature-engineered stacking model).
"""

from collections import OrderedDict
import numpy as np
import joblib
import os
from typing import Optional
import pandas as pd

# ==========================================================
# ORIGINAL AGENT-BASED ENSEMBLE (your previous code, unchanged)
# ==========================================================
class EnsembleGuardian:
    """
    Combine agent outputs. Two modes:
    - soft voting: weighted sum of agent scores -> pick label with highest vote
    - meta classifier: train a sklearn classifier on per-agent scores (see main.py)
    """

    def __init__(self, agents, weights=None, meta_path: str = "mlenv/meta.joblib"):
        self.agents = agents
        self.weights = weights or {a.__class__.__name__: 1.0 for a in agents}
        self.meta_path = meta_path
        self.meta = None
        self.label_order = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]

    def detect(self, claim: str, evidence: str) -> dict:
        # run agents
        results = [a.detect(claim, evidence) for a in self.agents]
        votes = {lab: 0.0 for lab in self.label_order}
        agent_results = []
        feat = []
        for res, agent in zip(results, self.agents):
            name = agent.__class__.__name__
            w = float(self.weights.get(name, 1.0))
            label = res.get("label", "NOT ENOUGH INFO")
            score = float(res.get("score", 0.0))
            votes[label] = votes.get(label, 0.0) + score * w
            agent_results.append(res)
            feat.append(score)
        # If meta classifier exists, use it
        if self.meta is None:
            try:
                if os.path.exists(self.meta_path):
                    self.meta = joblib.load(self.meta_path)
            except Exception:
                self.meta = None
        if self.meta is not None:
            # meta expects array-shaped inputs
            probs = self.meta.predict_proba([feat])[0]
            best_ix = int(np.argmax(probs))
            final_label = self.meta.classes_[best_ix]
            confidence = float(probs[best_ix])
            return {"label": final_label, "score": confidence, "agent_results": agent_results, "meta_used": True}
        # else soft voting
        final_label = max(votes.items(), key=lambda x: x[1])[0]
        total = sum(votes.values()) + 1e-9
        confidence = votes[final_label] / total
        return {"label": final_label, "score": float(confidence), "agent_results": agent_results, "meta_used": False}

    def features_from_agents(self, agent_outputs):
        """
        Given a list of agent result dicts (order must match self.agents), produce a numeric feature vector.
        """
        return [float(res.get("score", 0.0)) for res in agent_outputs]

    def save_meta(self, clf):
        os.makedirs(os.path.dirname(self.meta_path) or ".", exist_ok=True)
        joblib.dump(clf, self.meta_path)
        self.meta = clf


# ==========================================================
# NEW ADDITION — Hallucination Detection Pipeline Wrapper
# ==========================================================
"""
This helper lets any agent (or external code) use the advanced
hallucination detection model (feature-engineered stacking ensemble).
"""

PIPELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "..", "models", "hallucination_pipeline.joblib"
)

_pipeline_cache = None

def _load_hallucination_pipeline(path: Optional[str] = None):
    """Load hallucination detection pipeline lazily and cache it."""
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    from models import load_model_pipeline
    p = path or PIPELINE_PATH
    # check both absolute and relative
    if not os.path.exists(p):
        alt = os.path.join(os.getcwd(), "models", "hallucination_pipeline.joblib")
        if os.path.exists(alt):
            p = alt
        else:
            raise FileNotFoundError(
                f"Hallucination pipeline not found. Tried:\n - {p}\n - {alt}\nPlease run `python main.py new_train` first."
            )

    _pipeline_cache = load_model_pipeline(p)
    return _pipeline_cache

def predict_hallucination(claim: str, evidence_text: str, pipeline_path: Optional[str] = None) -> str:
    """
    Predict hallucination label for a single (claim, evidence_text) pair
    using the trained feature-engineered stacking ensemble.
    """
    pipeline = _load_hallucination_pipeline(pipeline_path)
    fe = pipeline["feature_engineer"]
    model = pipeline["model"]
    le = pipeline["label_encoder"]

    feats = fe.extract_all_features(claim, evidence_text)
    X = pd.DataFrame([feats])
    pred_encoded = model.predict(X)[0]
    label = le.inverse_transform([int(pred_encoded)])[0]
    return label

class HallucinationPipelineHelper:
    """
    Helper class wrapper (optional) around the hallucination model
    for batch or single predictions.
    """

    def __init__(self, pipeline_path: Optional[str] = None):
        self.pipeline = _load_hallucination_pipeline(pipeline_path)
        self.fe = self.pipeline["feature_engineer"]
        self.model = self.pipeline["model"]
        self.le = self.pipeline["label_encoder"]

    def predict_one(self, claim: str, evidence_text: str) -> str:
        feats = self.fe.extract_all_features(claim, evidence_text)
        import pandas as pd
        X = pd.DataFrame([feats])
        pred_encoded = self.model.predict(X)[0]
        return self.le.inverse_transform([int(pred_encoded)])[0]

    def predict_batch(self, df: pd.DataFrame):
        """
        Predict on a dataframe with columns claim, evidence_text
        """
        from feature_engineer import create_enhanced_features_dataset
        X = create_enhanced_features_dataset(df, self.fe)
        preds = self.model.predict(X)
        labels = self.le.inverse_transform(preds.astype(int))
        out = df.copy()
        out["pred_label"] = labels
        return out