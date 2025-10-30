# ml_guardian/agents/heuristic.py
import re

_NUM_RE = re.compile(r"\d+[.,]?\d*")
_NEG_WORDS = {"not", "never", "no", "n't", "none", "neither", "nor"}

def _extract_numbers(text):
    return _NUM_RE.findall(text or "")

def _has_negation(text):
    txt = (text or "").lower()
    return any(w in txt for w in _NEG_WORDS)

def _lexical_overlap(a, b):
    if not a or not b:
        return 0.0
    sa = set(re.findall(r"\\w+", a.lower()))
    sb = set(re.findall(r"\\w+", b.lower()))
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)

class HeuristicGuardian:
    """
    Rule-based guardian combining number checks, negation agreement, and lexical overlap.
    Produces a score in [0,1] and label based on threshold.
    """

    def __init__(self, overlap_weight: float = 0.5, num_penalty: float = 0.35, negation_boost: float = 0.15, threshold: float = 0.55):
        self.overlap_weight = overlap_weight
        self.num_penalty = num_penalty
        self.negation_boost = negation_boost
        self.threshold = threshold

    def detect(self, claim: str, evidence: str) -> dict:
        overlap = _lexical_overlap(claim, evidence)
        claim_nums = _extract_numbers(claim)
        evi_nums = _extract_numbers(evidence)
        score = overlap * self.overlap_weight + (1 - self.overlap_weight) * 0.5  # baseline
        # penalize missing numbers
        if claim_nums and not evi_nums:
            score -= self.num_penalty
        # boost when negation matches
        if _has_negation(claim) and _has_negation(evidence):
            score += self.negation_boost
        score = max(0.0, min(1.0, score))
        label = "SUPPORTED" if score >= self.threshold else "NOT ENOUGH INFO"
        return {"agent": "heuristic", "label": label, "score": float(score),
                "details": {"lexical_overlap": overlap, "claim_numbers": claim_nums[:3], "evidence_numbers": evi_nums[:3]}}