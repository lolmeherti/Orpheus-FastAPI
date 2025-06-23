# llm_classifier.py

import httpx
import json
import logging
from pathlib import Path

# (Constants like DEFAULT_LM_STUDIO_CHAT_URL, etc., remain the same)
DEFAULT_LM_STUDIO_CHAT_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL_NAME = "hermes"
POSSIBLE_TAGS = [
    "Affirming", "Banter", "Critical", "Emergency",
    "Flirt", "Grief", "Intimate", "Neutral", "Philosophical"
]

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class LLMStyleClassifier:
    def __init__(self,
                 system_prompt_path: str | Path,
                 lm_studio_url: str = DEFAULT_LM_STUDIO_CHAT_URL,
                 model_name: str = DEFAULT_MODEL_NAME,
                 temperature: float = 0.3,
                 max_tokens: int = 30):

        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
            logger.info(f"LLMStyleClassifier initialized with prompt from: {system_prompt_path}")
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {system_prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading system prompt from {system_prompt_path}: {e}")
            raise

    def get_classification(self, user_input: str) -> dict:
        logger.info(f"CLASSIFIER_START: Getting classification for: '{user_input[:50]}...'")

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_input}]
        payload = {"model": self.model_name, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": False}

        try:
            with httpx.Client(timeout=httpx.Timeout(10.0, read=30.0)) as client:
                response = client.post(self.lm_studio_url, json=payload)
                response.raise_for_status()

            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            predicted_tag_raw = content.strip().strip(".,;:\"'")

            for possible_tag in POSSIBLE_TAGS:
                if possible_tag.lower() == predicted_tag_raw.lower():
                    logger.info(f"CLASSIFIER_SUCCESS: Predicted tag '{possible_tag}' for input: '{user_input[:50]}...'")
                    return {"tag": possible_tag, "score": 1.0, "source": "llm_classifier"}

            logger.warning(f"CLASSIFIER_UNEXPECTED_TAG: LLM returned '{predicted_tag_raw}' for input '{user_input[:50]}...'")
            return {"tag": f"Unexpected: {predicted_tag_raw}", "score": 0.0, "source": "llm_classifier_unexpected_tag"}

        except httpx.RequestError as e:
            logger.error(f"CLASSIFIER_REQUEST_ERROR: {e}")
            return {"tag": "RequestError", "score": 0.0, "source": "llm_classifier_error"}
        except httpx.HTTPStatusError as e:
            logger.error(f"CLASSIFIER_HTTP_ERROR: Status {e.response.status_code} - {e.response.text}")
            return {"tag": f"HTTPError:{e.response.status_code}", "score": 0.0, "source": "llm_classifier_error"}
        except Exception as e:
            logger.error(f"CLASSIFIER_UNEXPECTED_ERROR: {e}", exc_info=True)
            return {"tag": "UnexpectedError", "score": 0.0, "source": "llm_classifier_error"}