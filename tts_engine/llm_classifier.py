# --- START OF FILE llm_classifier.py ---

import httpx
import json
import time
import logging
from pathlib import Path

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
                 temperature: float = 0.3, # Made temperature configurable
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
        """
        Sends the user input to the LLM for style classification.
        Returns a dictionary similar to VetoSystem's output:
        {"tag": "PredictedTag", "score": 1.0 (or LLM confidence if available), "source": "llm_classifier"}
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        tag_to_return = "unknown" # Default
        source_to_return = "llm_classifier_error"

        score = 0.0

        try:
            with httpx.Client(timeout=httpx.Timeout(10.0, read=30.0), trust_env=False) as client:
                response = client.post(self.lm_studio_url, json=payload)
                response.raise_for_status()

            response_data = response.json()

            if response_data.get("choices") and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                predicted_tag_raw = content.strip().strip(".,;:\"'")

                # Validate against POSSIBLE_TAGS
                for possible_tag in POSSIBLE_TAGS:
                    if possible_tag.lower() == predicted_tag_raw.lower():
                        tag_to_return = possible_tag # Use canonical casing
                        score = 1.0
                        source_to_return = "llm_classifier"
                        break
                else:
                    logger.warning(f"LLM returned an unexpected tag: '{predicted_tag_raw}' for input '{user_input}'. Full response: {content}")
                    tag_to_return = f"Unexpected: {predicted_tag_raw}"
                    source_to_return = "llm_classifier_unexpected_tag"
            else:
                logger.error(f"LLM response malformed or empty for input '{user_input}': {response_data}")
                tag_to_return = "MalformedResponse"
                source_to_return = "llm_classifier_malformed"

        except httpx.RequestError as e_req:
            logger.error(f"LLM request failed (RequestError) for input '{user_input}': {e_req}")
            tag_to_return = "RequestError"
        except httpx.HTTPStatusError as e_stat:
            logger.error(f"LLM request failed (HTTPStatusError {e_stat.response.status_code}) for input '{user_input}': {e_stat.response.text}")
            tag_to_return = f"HTTPError:{e_stat.response.status_code}"
        except json.JSONDecodeError as e_json:
            logger.error(f"Failed to decode LLM JSON response for input '{user_input}': {e_json}")
            tag_to_return = "JSONDecodeError"
        except Exception as e_unexp:
            logger.error(f"An unexpected error occurred for input '{user_input}': {e_unexp}")
            tag_to_return = "UnexpectedError"

        return {"tag": tag_to_return, "score": score, "source": source_to_return}

# --- Batch Testing Function (can remain for standalone testing) ---
TEST_PHRASES = [
    # (Your list of test phrases can remain here)
    "That's absolutely fantastic news!", "You must be joking, right?",
    "This is completely unacceptable service.", "There's a fire in the kitchen, call 911!",
    "Hey there, handsome. Got any plans tonight?", "I'm devastated by this loss.",
    "I feel so close to you right now.", "The sky is blue today.",
    "Do we have free will, or is everything predetermined?", "I really appreciate your help on this.",
    "Seriously? You think that's a good idea?", "I'm extremely disappointed with the outcome.",
    "My car broke down in the middle of nowhere and my phone is dying!",
    "Are you a magician? Because whenever I look at you, everyone else disappears.",
    "It's hard to imagine life without them.", "Our bond is something I treasure deeply.",
    "Just grabbing a coffee.", "What is the nature of consciousness?",
    "Could you pass the salt, please?", "You always know how to make me laugh.",
    "I'm not sure I agree with that assessment.", "Get out of my house now!",
    "Truly a genius at work here.", "You're a real comedian.",
    "Wow.. Another stunningly helpful suggestion, thank you.",
    "Slow down, einstein, you're gonna break the internet.",
    "That sound you hear is my soul leaving my body.", "somebody give this AI a raise",
    "just when i thought you couldn't get any better...", "yikes. even Clippy did better",
    "wow, captain obvious strikes again", "i've seen more processing power in a potato.",
]

def run_batch_classification_standalone():
    """
    Iterates through TEST_PHRASES, gets classifications, and prints results.
    This is for standalone testing of this module.
    """
    script_dir = Path(__file__).parent.resolve()
    default_prompt_path_for_test = script_dir.parent / "personas" / "mood_classifier_bot.txt"

    if not default_prompt_path_for_test.exists():
        print(f"ERROR: Standalone test requires prompt file at: {default_prompt_path_for_test}")
        print("Please create 'mood_classifier_bot.txt' or adjust the path in run_batch_classification_standalone().")
        fallback_prompt_path = script_dir / "mood_classifier_bot.txt"
        if fallback_prompt_path.exists():
            default_prompt_path_for_test = fallback_prompt_path
            print(f"Using fallback prompt path: {default_prompt_path_for_test}")
        else:
            print(f"Fallback prompt path also not found: {fallback_prompt_path}")
            return


    print("LLM Mood/Style Classifier Standalone Batch Test")
    try:
        classifier = LLMStyleClassifier(system_prompt_path=default_prompt_path_for_test)
    except Exception as e:
        print(f"Failed to initialize LLMStyleClassifier for standalone test: {e}")
        return

    print(f"Processing {len(TEST_PHRASES)} test phrases...")
    print("-" * 70)
    print(f"{'Input Phrase':<80} | {'Predicted Tag':<20} | {'Score':<5} | {'Source'}")
    print("-" * 70)

    total_start_time = time.monotonic()
    for phrase in TEST_PHRASES:
        result = classifier.get_classification(phrase)
        display_phrase = (phrase[:77] + '...') if len(phrase) > 80 else phrase
        print(f"{display_phrase:<80} | {result['tag']:<20} | {result['score']:.1f} | {result['source']}")

    total_duration = time.monotonic() - total_start_time
    print("-" * 70)
    print(f"Finished processing {len(TEST_PHRASES)} phrases in {total_duration:.2f} seconds.")
    if len(TEST_PHRASES) > 0:
        print(f"Average time per phrase: {total_duration / len(TEST_PHRASES) * 1000:.0f} ms.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logger.info("Running llm_classifier.py as standalone script for batch testing.")
    run_batch_classification_standalone()