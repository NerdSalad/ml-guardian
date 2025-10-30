# ml_guardian/agents/clustering.py
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import os

class ClusteringGuardian:
    """
    Trains a TF-IDF + IsolationForest on claim+evidence concatenations to estimate outlierness.
    If model not fit, detect returns neutral score 0.5.
    """

    def __init__(self, tfidf_kwargs=None, iso_kwargs=None, model_dir: str = "mlenv/clustering"):
        self.tfidf_kwargs = tfidf_kwargs or {"max_features": 20000, "ngram_range": (1,2)}
        self.iso_kwargs = iso_kwargs or {"n_estimators": 100, "random_state": 42, "contamination": 0.02}
        self.model_dir = model_dir
        self.vectorizer = None
        self.iso = None
        os.makedirs(self.model_dir, exist_ok=True)

    def fit(self, texts):
        """
        Fit TF-IDF and IsolationForest. texts: iterable of str (claim + evidence combined).
        """
        self.vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        X = self.vectorizer.fit_transform(texts)
        self.iso = IsolationForest(**self.iso_kwargs)
        self.iso.fit(X)
        joblib.dump(self.vectorizer, os.path.join(self.model_dir, "tfidf.joblib"))
        joblib.dump(self.iso, os.path.join(self.model_dir, "iso.joblib"))

    def load(self):
        try:
            self.vectorizer = joblib.load(os.path.join(self.model_dir, "tfidf.joblib"))
            self.iso = joblib.load(os.path.join(self.model_dir, "iso.joblib"))
            return True
        except Exception:
            return False

    def detect(self, claim: str, evidence: str) -> dict:
        # neutral if not trained
        if (self.vectorizer is None or self.iso is None) and not self.load():
            return {"agent": "clustering", "label": "NOT ENOUGH INFO", "score": 0.5, "details": {"note": "not trained"}}
        txt = (claim or "") + " " + (evidence or "")
        X = self.vectorizer.transform([txt])
        # IsolationForest.score_samples returns anomaly score (higher = more normal), normalize to 0..1
        score = float(self.iso.score_samples(X)[0])
        # score is unbounded negative..positive, we can convert via logistic
        s_norm = 1.0 / (1.0 + np.exp(-score))
        label = "SUPPORTED" if s_norm >= 0.5 else "NOT ENOUGH INFO"
        return {"agent": "clustering", "label": label, "score": float(s_norm), "details": {"raw_score": float(score)}}