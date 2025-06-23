# style_classifier.py

import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
# Import the NON_EMERGENCY_ANCHORS which are now crucial
from master_anchors import NON_EMERGENCY_ANCHORS

# ==============================================================================
# == The Definitive Hybrid Cascade Model ==
# ==============================================================================

class Classifier:
    """A generic sentence classifier engine."""
    def __init__(self, model: SentenceTransformer, anchor_config: dict):
        self.model = model
        self.anchors, self.labels = self._prepare_anchors(anchor_config)
        if self.anchors:
            self.embeddings = self._encode(self.anchors)
        else:
            self.embeddings = np.array([])

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

    def _prepare_anchors(self, config: dict) -> tuple[list[str], list[str]]:
        texts, labels = [], []
        for label, text_list in config.items():
            for text in text_list:
                texts.append(text.strip())
                labels.append(label)
        return texts, labels

    def classify(self, text: str) -> dict:
        if self.embeddings.size == 0:
            return {"tag": "unknown", "score": 0.0}
        input_embedding = self._encode([text.strip()])
        scores = util.dot_score(input_embedding, self.embeddings)[0].cpu().numpy()
        best_idx = int(np.argmax(scores))
        return {"tag": self.labels[best_idx], "score": float(scores[best_idx])}


class VetoSystem:
    """
    Implements the definitive Hybrid Cascade model: An emergency veto followed by
    a two-step gate and specialist classification, built with the correct tag definitions.
    """
    def __init__(self, master_anchor_config: dict, overrides_config: dict):
        print("[VetoSystem] Initializing Definitive Hybrid Cascade...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        self.overrides = {self._normalize(k): v for k, v in overrides_config.items()}

        # --- Define the Ground Truth for Social Tags ---
        social_keys = {"banter", "flirt", "intimate"}

        # --- Classifier 0: The Emergency Veto (MUST RUN FIRST) ---
        print("  - Building Emergency Veto classifier (Step 0)...")
        # THIS IS THE KEY ARCHITECTURAL FIX: Use negative examples.
        emergency_veto_config = {
            "emergency": master_anchor_config.get("emergency", []),
            "not_emergency": NON_EMERGENCY_ANCHORS
        }
        self.emergency_checker = Classifier(model, emergency_veto_config)

        # --- Classifier 1: The Main Gate (built with correct social definition) ---
        print("  - Building Main Gate classifier (Step 1)...")
        main_gate_config = self._build_main_gate_config(master_anchor_config, social_keys)
        self.main_gate_classifier = Classifier(model, main_gate_config)

        # --- Classifier 2: The Social Specialist ---
        print("  - Building Social Specialist classifier (Step 2)...")
        social_specialist_config = {key: master_anchor_config.get(key, []) for key in social_keys}
        self.social_specialist_classifier = Classifier(model, social_specialist_config)

        # --- System Configuration ---
        self.EMERGENCY_VETO_THRESHOLD = 0.65
        self.SOCIAL_GATE_THRESHOLD = 0.35
        print(f"  - Emergency Veto Threshold set to: {self.EMERGENCY_VETO_THRESHOLD}")
        print(f"  - Social Gate Threshold set to: {self.SOCIAL_GATE_THRESHOLD}")
        print("[VetoSystem] Ready.")

    def _build_main_gate_config(self, master_config: dict, social_keys: set) -> dict:
        """Dynamically builds the configuration for the Step 1 classifier."""
        config = {}
        social_anchors = []

        for key, anchors in master_config.items():
            if key in social_keys:
                social_anchors.extend(anchors)
            else: # Includes 'emergency', 'critical', 'grief', 'affirming', etc.
                config[key] = anchors

        config["social"] = social_anchors
        return config

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "", s).strip().lower()

    def get_classification(self, text: str) -> dict:
        normalized_input = self._normalize(text)
        if override_tag := self.overrides.get(normalized_input):
            return {"tag": override_tag, "score": 1.0, "source": "override"}

        # --- STEP 0: THE EMERGENCY VETO ---
        emergency_result = self.emergency_checker.classify(text)
        # The veto now only triggers if the TAG is 'emergency' AND the score is high.
        if emergency_result['tag'] == 'emergency' and emergency_result['score'] >= self.EMERGENCY_VETO_THRESHOLD:
            emergency_result['source'] = 'emergency_veto'
            return emergency_result

        # --- STEP 1: Classify against the Main Gate ---
        gate_result = self.main_gate_classifier.classify(text)

        # --- STEP 2: The Conditional Social Check ---
        if gate_result['tag'] == 'social':
            # If the gate says 'social', we ALWAYS consult the specialist.
            specialist_result = self.social_specialist_classifier.classify(text)
            specialist_result['source'] = 'social_specialist'
            return specialist_result
        else:
            # The gate's result was NOT 'social' (e.g., it was affirming, critical, etc.)
            # We return its finding directly.
            gate_result['source'] = 'main_gate'
            return gate_result