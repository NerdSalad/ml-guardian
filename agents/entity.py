# ml_guardian/agents/entity.py
import re

class EntityGuardian:
    """
    Entity-level guardian. Tries to use spaCy NER. If spaCy isn't available or text is short (wiki titles),
    falls back to token/title matching.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", threshold: float = 0.25):
        self.spacy_model = spacy_model
        self.threshold = threshold
        self._nlp = None
        self._spacy_available = None

    def _ensure_spacy(self):
        if self._spacy_available is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(self.spacy_model)
            self._spacy_available = True
        except Exception:
            self._nlp = None
            self._spacy_available = False

    def _extract_entities_spacy(self, text):
        doc = self._nlp(text)
        return {ent.text.strip().lower() for ent in doc.ents if ent.text.strip()}

    def _fallback_entities(self, text):
        # if evidence is a wiki-title list like "Foo_Bar Baz", split on non-alpha and underscores
        tokens = re.split(r"[_\\W]+", text)
        tokens = {t.lower() for t in tokens if len(t) > 1}
        return tokens

    def detect(self, claim: str, evidence: str) -> dict:
        self._ensure_spacy()
        if self._spacy_available and claim.strip() and evidence.strip():
            try:
                claim_ents = self._extract_entities_spacy(claim)
                evidence_ents = self._extract_entities_spacy(evidence)
            except Exception:
                claim_ents = self._fallback_entities(claim)
                evidence_ents = self._fallback_entities(evidence)
        else:
            claim_ents = self._fallback_entities(claim)
            evidence_ents = self._fallback_entities(evidence)

        overlap = 0.0
        if claim_ents:
            overlap = len(claim_ents & evidence_ents) / max(1, len(claim_ents))
        label = "SUPPORTED" if overlap >= self.threshold else "NOT ENOUGH INFO"
        return {"agent": "entity", "label": label, "score": float(overlap), "details": {"claim_entities": list(claim_ents)[:10], "evidence_entities": list(evidence_ents)[:10]}}