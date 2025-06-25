# config.py
# Centralized configuration for the conversation agent.

import os
from pathlib import Path

# ==============================================================================
# == PATHS & DIRECTORIES ==
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CACHE_DIR = ROOT_DIR / "cached_clips"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "atomic_fillers").mkdir(exist_ok=True)
(CACHE_DIR / "mid_fillers").mkdir(exist_ok=True)

CLEANUP_FOLDERS = [OUTPUT_DIR]

SUMMARY_BOT_TEMPLATE = ROOT_DIR / "personas/chat_summary_bot.txt"
MOOD_CLASSIFIER_BOT = ROOT_DIR / "personas/mood_classifier_bot.txt"
PERSONA_PROMPT_TEMPLATE = ROOT_DIR / "personas/tts_default.txt"
SCRAPE_SUMMARY_BOT_TEMPLATE = ROOT_DIR / "personas/scrape_summary_bot.txt"


# ==============================================================================
# == API & SERVICE CONFIGURATION ==
# ==============================================================================
# --- Core Application Settings ---
LM_STUDIO_CHAT_URL = "http://127.0.0.1:1234/v1/chat/completions"
ORPHEUS_API_URL = "http://127.0.0.1:5005/v1/audio/speech"
VOICE = "tara"
TTS_REQUEST_TIMEOUT = (10, 30)

# --- LLM & History Management ---
MIN_WORDS_FOR_CHUNK = 5
TOKEN_SOFT_LIMIT = 2500
TOKEN_HARD_LIMIT = 7700
KEEP_RECENT = 2

# ==============================================================================
# == WEB API & SERVICE CONFIGURATION ==
# ==============================================================================
TOOL_SERVER_URL="http://127.0.0.1:5007"

# ==============================================================================
# == VAD & ASR CONFIGURATION ==
# ==============================================================================
# --- VAD (Voice Activity Detection) Settings ---
VAD_INTERRUPT_TIMEOUT_S = 0.2
VAD_DEFAULT_SILENCE_TIMEOUT_S = 1.2
VAD_LONG_PROMPT_TIMEOUT_S = 2.0
VAD_LONG_PROMPT_TRIGGER_S = 1.5
VAD_MIN_SPEECH_S = 0.25
VAD_SPEECH_CONFIDENCE_THRESHOLD = 0.3

# --- ASR (Automatic Speech Recognition) Settings ---
MIN_WORDS_ASR = 1
ECHO_SIMILARITY_THRESHOLD = 0.7
INTERRUPTION_DURATION_THRESHOLD_S = 4
INTERRUPT_KEYWORDS = {"stop", "shut up", "hold on", "wait", "enough", "nevermind", "cancel"}


# ==============================================================================
# == AUDIO PROCESSING CONFIGURATION ==
# ==============================================================================
REF_AUDIO_CLIP_PATH = CACHE_DIR / "alright_i_hear_you_pending_cache.wav"
REFERENCE_F0 = 0.0

CROSSFADE_MS = 80
CHUNK_MS = 20
SILENCE_THRES_DB = -50
FADE_IN_MS = 100
TARGET_LUFS = -27.2
SAFE_PAUSE_MS = 250
EDGE_FADE_MS = 5


# ==============================================================================
# == ENVIRONMENT SETUP ==
# ==============================================================================
# Suppress noisy library logs
os.environ["NEMO_DISABLE_TQDM"] = "1"