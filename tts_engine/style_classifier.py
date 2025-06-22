# mood_classifier.py

from sentence_transformers import SentenceTransformer, util
import numpy as np
from style_benchmark import anchor_texts, anchor_labels, MANUAL_OVERRIDES
import re

print("[MoodClassifier] Loading model and anchors...")
model = SentenceTransformer("all-MiniLM-L6-v2")
anchor_embeddings = model.encode(anchor_texts, normalize_embeddings=True)
print("[MoodClassifier] Ready.")

# ==============================================================================
# == MOOD DETECTION CONFIGURATION ==
# ==============================================================================
STYLE_FALLBACK_MAP = {
    "flirt": ["banter", "intimate", "neutral"],
    "banter": ["intimate", "neutral"],
    "intimate": ["neutral"],
    "grief": ["intimate", "neutral"],
    "emergency": ["calm", "neutral"],
    "philosophical": ["ponder", "neutral"],
    "affirming": ["neutral"],
    "neutral": [],
}

TAG_SCORE_MINIMUMS = {
    "flirt": 0.38,
    "intimate": 0.40,
    "banter": 0.35,
    "grief": 0.30,
    "emergency": 0.30,
    "critical": 0.35,
}

HIGH_CONFIDENCE = 0.50
MID_CONFIDENCE  = 0.38
LOW_CONFIDENCE  = 0.30

def classify(text: str, verbose: bool = False) -> dict | str:
    """
    Classify the emotional mood of the input string.

    Args:
        text (str): The input sentence to classify.
        verbose (bool): If True, returns tag + source + score. If False, returns tag only.

    Returns:
        str or dict: Either just the tag or full result depending on `verbose`.
    """
    clean_text = text.strip().lower()

    def normalize(s):
        return re.sub(r"[^a-zA-Z0-9]+", " ", s).strip()

    normalized_input = normalize(clean_text)

    for phrase, tag in MANUAL_OVERRIDES.items():
        if normalize(phrase) == normalized_input:
            return {"tag": tag, "score": 1.0, "source": "override"} if verbose else tag

    embedding = model.encode([clean_text], normalize_embeddings=True)
    scores = util.dot_score(embedding, anchor_embeddings)[0].cpu().numpy()
    best_idx = int(np.argmax(scores))
    tag = anchor_labels[best_idx]
    return {"tag": tag, "score": float(scores[best_idx]), "source": "model"} if verbose else tag

def resolve_fallback(tag: str, score: float) -> str:
    tag_min = TAG_SCORE_MINIMUMS.get(tag, LOW_CONFIDENCE)
    if score < tag_min:
        return "neutral"

    fallback_chain = STYLE_FALLBACK_MAP.get(tag, [])
    if score >= HIGH_CONFIDENCE:
        return tag
    elif score >= MID_CONFIDENCE:
        return tag
    elif fallback_chain:
        return fallback_chain[0]
    return "neutral"
