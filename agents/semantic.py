# ml_guardian/agents/semantic.py
import math
from typing import List, Tuple

class SemanticGuardian:
    """
    Semantic similarity guardian using sentence-transformers.
    Efficient: lazy loads model, encodes in batches.
    Returns label in {"SUPPORTED", "NOT ENOUGH INFO", "REFUTED"} (REFUTED is not inferred here).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.65, batch_size: int = 64, device: str = None):
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self._model = None
        self._util = None
        self._device = device  # "cuda" or "cpu" or None (auto)

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer, util
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for SemanticGuardian. Install it: pip install sentence-transformers") from e
        # device resolution
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        self._model = SentenceTransformer(self.model_name, device=self._device)
        self._util = util

    def _batch_encode(self, texts: List[str]):
        self._ensure_model()
        return self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)

    def score(self, claims: List[str], evidences: List[str]) -> List[float]:
        """
        Vectorized scoring: returns cosine similarities list aligned with input lists.
        """
        if len(claims) != len(evidences):
            raise ValueError("claims and evidences must have same length")
        if not claims:
            return []
        self._ensure_model()
        # encode claims and evidences separately in batches
        claim_embs = self._batch_encode(claims)
        evi_embs = self._batch_encode(evidences)
        # compute cosine similarities
        sims = []
        for i in range(len(claims)):
            s = float(self._util.cos_sim(claim_embs[i], evi_embs[i]).item())
            sims.append(s)
        return sims

    def detect(self, claim: str, evidence: str) -> dict:
        s = self.score([claim], [evidence])[0]
        label = "SUPPORTED" if s >= self.threshold else "NOT ENOUGH INFO"
        return {"agent": "semantic", "label": label, "score": float(s), "details": {"similarity": float(s), "threshold": self.threshold}}

    def batch_detect(self, claim_list, evidence_list):
        sims = self.score(claim_list, evidence_list)
        results = []
        for s in sims:
            lab = "SUPPORTED" if s >= self.threshold else "NOT ENOUGH INFO"
            results.append({"agent": "semantic", "label": lab, "score": float(s), "details": {"similarity": float(s), "threshold": self.threshold}})
        return results