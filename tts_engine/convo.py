#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# convo.py

import json, os, queue, re, sys, threading, traceback, uuid, random, time, logging, pathlib
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

# Core libraries
import numpy as np
import requests
import torch
import httpx
import http.client
from faster_whisper import WhisperModel

# Audio libraries
import sounddevice as sd
import soundfile as sf
import pyaudio
from pydub import AudioSegment

# --- Custom Library Imports ---
from streaming_chunker import StreamingChunker
from audio_processing import preprocess_clip
from filler_queue_builder import FillerQueueBuilder
from llm_classifier import LLMStyleClassifier

os.environ["NEMO_DISABLE_TQDM"] = "1"

# --- VAD & ASR Configuration ---
VAD_INTERRUPT_TIMEOUT_S = 0.2
VAD_DEFAULT_SILENCE_TIMEOUT_S = 1.2
VAD_LONG_PROMPT_TIMEOUT_S = 2.0
VAD_LONG_PROMPT_TRIGGER_S = 3.0
INTERRUPTION_DURATION_THRESHOLD_S = 4
MIN_WORDS_ASR = 1
VAD_MIN_SPEECH_S = 0.25
VAD_SPEECH_CONFIDENCE_THRESHOLD = 0.3
ECHO_SIMILARITY_THRESHOLD = 0.7

# --- Core Application Settings ---
LM_STUDIO_CHAT_URL = "http://127.0.0.1:1234/v1/chat/completions"
ORPHEUS_API_URL = "http://127.0.0.1:5005/v1/audio/speech"
VOICE = "tara"
MIN_WORDS_FOR_CHUNK = 8
TOKEN_SOFT_LIMIT = 2500
TOKEN_HARD_LIMIT = 7700
KEEP_RECENT = 2

# --- USER PATHS (Corrected and Robust) ---
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CACHE_DIR = ROOT_DIR / "cached_clips"

filler_builder = FillerQueueBuilder(
    atomic_dir=CACHE_DIR / "atomic_fillers",
    mid_dir=CACHE_DIR   / "mid_fillers"
)

SUMMARY_BOT_TEMPLATE = ROOT_DIR / "personas/chat_summary_bot.txt"
MOOD_CLASSIFIER_BOT = ROOT_DIR / "personas/mood_classifier_bot.txt"
PERSONA_PROMPT_TEMPLATE = ROOT_DIR / "personas/tts_default.txt"

OUTPUT_DIR.mkdir(exist_ok=True)
CLEANUP_FOLDERS = [OUTPUT_DIR, ROOT_DIR / "outputs"]
TTS_REQUEST_TIMEOUT = (10, 30)

# ==============================================================================
# == CENTRALIZED AUDIO CONFIGURATION ==
# ==============================================================================
REF_AUDIO_CLIP_PATH = CACHE_DIR / "alright_i_hear_you_pending_cache.wav"
CROSSFADE_MS = 80
CHUNK_MS = 20
SILENCE_THRES_DB = -50
FADE_IN_MS = 100
TARGET_LUFS = -27.2
SAFE_PAUSE_MS = 250
EDGE_FADE_MS = 5
REFERENCE_F0 = 0.0

# --- Global State ---
audio_q, prefetch_q, user_q, memory_q = (queue.Queue() for _ in range(4))
interruption_requested, tts_actively_playing, program_is_shutting_down = threading.Event(), threading.Event(), threading.Event()
INTERRUPT_KEYWORDS = {"stop", "shut up", "hold on", "wait", "enough", "nevermind", "cancel"}

last_tts_text = ""
current_generation_id = None

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)-15s] %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger("http.client").setLevel(logging.WARNING)

llm_classifier_instance = None

# --- Helper Functions ---
def strip_action_and_emoji(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"<[^>]+>", '', text) # Strip <action> tags
    text = re.sub(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]', '', text) # Strip *emphasis*, _italics_, (parentheticals), and emojis
    text = re.sub(r'<\|.*?\|>', '', text) # Strip special tokens like <|im_start|>
    return " ".join(text.split()) # Normalize whitespace

def rough_tokens(txt: str) -> int:
    return max(1, len(txt) // 4)

# --- Worker Threads ---

def asr_listener():
    try:
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"
        logging.info(f"ASR_LOAD_START: Loading ASR model: small.en ({device}, {compute_type})")
        asr_model = WhisperModel("small.en", device=device, compute_type=compute_type)
        logging.info("ASR_LOAD_DONE: ASR model loaded.")

        logging.info("VAD_LOAD_START: Loading VAD model...")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils # Unpack utils
        logging.info("VAD_LOAD_DONE: VAD model loaded.")

    except Exception as e:
        logging.critical(f"ASR_VAD_LOAD_FATAL: Could not load ASR/VAD models: {e}\n{traceback.format_exc()}", exc_info=True)
        program_is_shutting_down.set()
        return

    sr, vad_chunk_size = 16000, 512
    is_speaking, audio_buffer, silence_counter = False, [], 0

    logging.info("ASR_LISTENER_READY: Waiting for speech...")
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=vad_chunk_size, callback=None) as stream:
            while not program_is_shutting_down.is_set():
                frame, overflowed = stream.read(vad_chunk_size)
                if overflowed:
                    logging.warning("ASR_OVERFLOW: Input stream overflowed.")

                is_speech = vad_model(torch.from_numpy(frame.flatten()), sr).item() > VAD_SPEECH_CONFIDENCE_THRESHOLD

                if is_speaking:
                    audio_buffer.append(frame)
                    if not is_speech:
                        silence_counter += 1
                        current_audio_duration_s = len(audio_buffer) * (vad_chunk_size / sr)
                        active_timeout_s = VAD_DEFAULT_SILENCE_TIMEOUT_S
                        if tts_actively_playing.is_set():
                            active_timeout_s = VAD_INTERRUPT_TIMEOUT_S
                        elif current_audio_duration_s > VAD_LONG_PROMPT_TRIGGER_S:
                            active_timeout_s = VAD_LONG_PROMPT_TIMEOUT_S

                        if silence_counter * (vad_chunk_size / sr) >= active_timeout_s:
                            is_speaking = False
                            full_audio_np = np.concatenate(audio_buffer).squeeze()
                            audio_buffer = []
                            effective_audio_duration_s = len(full_audio_np) / sr

                            if effective_audio_duration_s >= VAD_MIN_SPEECH_S:
                                # logging.debug(f"ASR_TRANSCRIBE_START: Transcribing audio of duration {effective_audio_duration_s:.2f}s")
                                # trans_start_time = time.monotonic()
                                segments, info = asr_model.transcribe(
                                    full_audio_np,
                                    language="en",
                                    beam_size=1
                                )
                                # trans_duration = time.monotonic() - trans_start_time
                                txt = " ".join(segment.text for segment in segments).strip()
                                # logging.debug(f"ASR_TRANSCRIBE_DONE: Transcription time: {trans_duration:.3f}s. Detected: '{txt}'")


                                if txt:
                                    is_echo = False
                                    if last_tts_text:
                                        similarity = SequenceMatcher(None, txt.lower(), last_tts_text.lower()).ratio()
                                        is_echo = similarity > ECHO_SIMILARITY_THRESHOLD
                                        # if is_echo:
                                            # logging.info(f"ASR_ECHO_CHECK: Possible echo. Similarity: {similarity:.2f}. User: '{txt}', LastTTS: '{last_tts_text}'")


                                    if not is_echo and len(txt.split()) >= MIN_WORDS_ASR:
                                        logging.info(f"USER_SPEECH: '{txt}' (Audio: {effective_audio_duration_s:.2f}s)")
                                        print(f"\n🗣️ User: '{txt}' (Audio duration: {effective_audio_duration_s:.2f}s)") # Keep for UI

                                        if tts_actively_playing.is_set():
                                            normalized_input_for_interrupt = txt.lower().strip(".?!, ")
                                            is_keyword_interrupt = any(keyword in normalized_input_for_interrupt for keyword in INTERRUPT_KEYWORDS)
                                            is_short_interrupt = effective_audio_duration_s < INTERRUPTION_DURATION_THRESHOLD_S

                                            if is_keyword_interrupt:
                                                logging.info("ASR_INTERRUPT: Keyword interrupt detected during TTS.")
                                                interruption_requested.set()
                                            elif is_short_interrupt and not is_keyword_interrupt : # only set if not already keyword
                                                logging.info("ASR_INTERRUPT: Short utterance during TTS, flagging as potential interrupt.")
                                                interruption_requested.set()
                                        user_q.put((txt, effective_audio_duration_s))
                                    elif is_echo:
                                        logging.info(f"ASR_ECHO_SUPPRESSED: Echo detected and suppressed: '{txt}'")
                                    elif txt:
                                        logging.info(f"ASR_TOO_SHORT: Transcription too short, discarded: '{txt}'")
                            silence_counter = 0
                    else:
                        silence_counter = 0
                elif is_speech:
                    is_speaking = True
                    silence_counter = 0
                    audio_buffer = [frame]
    except sd.PortAudioError as pae:
        logging.critical(f"ASR_PORTAUDIO_ERROR: {pae}. Check audio device setup.", exc_info=True)
        program_is_shutting_down.set()
    except Exception as e:
        if not program_is_shutting_down.is_set():
            logging.error(f"ASR_LISTENER_ERROR: {e}\n{traceback.format_exc()}", exc_info=True)
    finally:
        logging.info("ASR_LISTENER_SHUTDOWN: ASR Listener shutting down.")


def kb_listener():
    logging.info("KB_LISTENER_READY: Keyboard listener started.")
    while not program_is_shutting_down.is_set():
        try:
            line = input("\n⌨️  ").strip()
            if program_is_shutting_down.is_set(): break
            if line:
                logging.info(f"USER_KB_INPUT: '{line}'")
                if tts_actively_playing.is_set() and line.lower() in INTERRUPT_KEYWORDS:
                    logging.info("KB_INTERRUPT: Keyboard interrupt keyword detected during TTS.")
                    interruption_requested.set()
                user_q.put((line, 0.0))
        except (EOFError, KeyboardInterrupt):
            logging.info("KB_LISTENER_EOF_OR_INTERRUPT: Shutting down keyboard listener.")
            break
        except Exception as e:
            if not program_is_shutting_down.is_set():
                logging.error(f"KB_LISTENER_ERROR: {e}", exc_info=True)
    logging.info("KB_LISTENER_SHUTDOWN: Keyboard listener finished.")


def audio_player_thread():
    global last_tts_text
    pa = pyaudio.PyAudio()
    stream = None
    active_gen_id = None
    current_stream_rate = None
    prev_tail = np.array([], dtype=np.int16)

    def open_stream(rate):
        nonlocal stream, current_stream_rate
        if stream:
            stream.stop_stream()
            stream.close()
        # logging.debug(f"AUDIO_PLAYER_OPEN_STREAM: Opening PyAudio stream at {rate} Hz.")
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True, frames_per_buffer=int(rate * CHUNK_MS / 1000))
        current_stream_rate = rate
        return stream

    def play_audio_data(data_samples, sample_rate):
        nonlocal stream, prev_tail # Added stream here
        if not stream or current_stream_rate != sample_rate:
            stream = open_stream(sample_rate)

        fade_len_samples = int(sample_rate * CROSSFADE_MS / 1000)
        actual_fade_len = min(len(prev_tail), len(data_samples), fade_len_samples)

        if actual_fade_len > 0:
            fade_out_curve = np.linspace(1.0, 0.0, actual_fade_len, dtype=np.float32)
            fade_in_curve = np.linspace(0.0, 1.0, actual_fade_len, dtype=np.float32)

            # Apply fade to copies to avoid modifying original arrays directly if they are slices
            tail_to_fade = prev_tail[:actual_fade_len].astype(np.float32)
            data_to_fade = data_samples[:actual_fade_len].astype(np.float32)

            blended_head = (tail_to_fade * fade_out_curve + data_to_fade * fade_in_curve).astype(np.int16)
            data_to_play = np.concatenate((blended_head, data_samples[actual_fade_len:]))
        else:
            data_to_play = data_samples

        # logging.debug(f"AUDIO_PLAYER_WRITE_STREAM: Writing {len(data_to_play)} samples to PyAudio stream.")
        stream.write(data_to_play.tobytes())

        if len(data_samples) >= fade_len_samples:
            prev_tail = data_samples[-fade_len_samples:]
        else:
            prev_tail = data_samples[:] # Take the whole thing if shorter than fade_len

    logging.info("AUDIO_PLAYER_READY: Audio player thread started.")
    while not program_is_shutting_down.is_set():
        try:
            gen_id, data_item, text_item = audio_q.get(timeout=1.0) # Adjusted for clarity
            # logging.debug(f"AUDIO_PLAYER_Q_GET: Received from audio_q. Text: '{str(text_item)[:50]}...', GenID: {gen_id}")

            if data_item is None:
                logging.info("AUDIO_PLAYER_SHUTDOWN_SIGNAL: Received None, shutting down.")
                break

            if text_item == "[PAUSE]":
                pause_duration_ms = data_item
                if not interruption_requested.is_set():
                    # logging.info(f"AUDIO_PLAYER_PAUSE: Pausing for {pause_duration_ms}ms.")
                    time.sleep(pause_duration_ms / 1000.0)
                # else:
                    # logging.info(f"AUDIO_PLAYER_PAUSE_SKIPPED: Pause skipped due to interruption request.")
                audio_q.task_done()
                continue

            audio_path_obj = data_item

            try:
                if gen_id != active_gen_id:
                    # logging.debug(f"AUDIO_PLAYER_NEW_GEN_ID: New generation ID {gen_id}. Resetting active_gen_id and prev_tail.")
                    active_gen_id = gen_id
                    prev_tail = np.array([], dtype=np.int16) # Reset crossfade tail for new response

                if interruption_requested.is_set():
                    # logging.info(f"AUDIO_PLAYER_INTERRUPT_SKIP: Skipping playback of '{str(text_item)[:50]}' due to interruption.")
                    audio_q.task_done()
                    continue

                tts_actively_playing.set()
                is_filler = text_item == "[FILLER]"

                log_text_display = audio_path_obj.name if is_filler else text_item
                if is_filler:
                    logging.info(f"AUDIO_PLAYER_PLAY_FILLER: Playing filler: {log_text_display}")
                    print(f"⏳ Filler: {audio_path_obj.name}")
                else:
                    logging.info(f"AUDIO_PLAYER_PLAY_TTS: Playing TTS: '{log_text_display[:60]}...' ({audio_path_obj.name})")
                    print(f"🎵 TTS: '{text_item}'")
                    last_tts_text = text_item

                current_audio_samples, current_sr = sf.read(str(audio_path_obj), dtype='int16')
                # load_duration = (time.monotonic() - load_start_time) * 1000
                # logging.debug(f"AUDIO_PLAYER_SF_READ: Loaded {audio_path_obj.name} in {load_duration:.0f}ms, SR: {current_sr}")
                play_audio_data(current_audio_samples, current_sr)

                while not interruption_requested.is_set():
                    try:
                        next_gen_id, next_data_item, next_text_item = audio_q.get(timeout=0.05) # Short timeout to check queue
                        if next_gen_id != active_gen_id: # Belongs to a newer generation/interruption
                            # logging.debug("AUDIO_PLAYER_CHAIN_BREAK_NEW_GEN: Next item from different gen_id. Re-queuing.")
                            audio_q.put((next_gen_id, next_data_item, next_text_item)) # Put it back
                            break

                        if next_text_item == "[PAUSE]":
                            pause_duration_ms = next_data_item
                            if not interruption_requested.is_set(): # Re-check before actual sleep
                                # logging.info(f"AUDIO_PLAYER_CHAIN_PAUSE: Pausing for {pause_duration_ms}ms.")
                                time.sleep(pause_duration_ms / 1000.0)
                            # else:
                                # logging.info(f"AUDIO_PLAYER_CHAIN_PAUSE_SKIPPED: Pause skipped due to interruption.")
                            audio_q.task_done()
                            continue

                        next_audio_path = next_data_item
                        is_next_filler = next_text_item == "[FILLER]"

                        log_text_display_next = next_audio_path.name if is_next_filler else next_text_item
                        if is_next_filler:
                            logging.info(f"AUDIO_PLAYER_CHAIN_FILLER: Playing chained filler: {log_text_display_next}")
                            print(f"⏳ Filler: {next_audio_path.name}") # Keep for UI
                        else:
                            logging.info(f"AUDIO_PLAYER_CHAIN_TTS: Playing chained TTS: '{log_text_display_next[:60]}...' ({next_audio_path.name})")
                            print(f"🎵 TTS: '{next_text_item}'") # Keep for UI
                            last_tts_text = next_text_item

                        next_samples, next_sr = sf.read(str(next_audio_path), dtype='int16')
                        # logging.debug(f"AUDIO_PLAYER_CHAIN_SF_READ: Loaded {next_audio_path.name} in {load_duration_next:.0f}ms, SR: {next_sr}")

                        play_audio_data(next_samples, next_sr)
                        current_sr = next_sr # Update current sample rate if it changed

                        if next_audio_path.exists() and not (CACHE_DIR.resolve() in next_audio_path.resolve().parents):
                            try:
                                # logging.debug(f"AUDIO_PLAYER_DELETE_OUTPUT: Deleting played output file: {next_audio_path}")
                                next_audio_path.unlink(missing_ok=True)
                            except Exception as e_del:
                                logging.warning(f"AUDIO_PLAYER_DELETE_FAIL: Could not delete {next_audio_path}: {e_del}")
                        audio_q.task_done()
                    except queue.Empty:
                        # logging.debug("AUDIO_PLAYER_CHAIN_EMPTY: No more chained items for this generation_id currently.")
                        break
            finally:
                audio_q.task_done()
                if audio_path_obj.exists() and not (CACHE_DIR.resolve() in audio_path_obj.resolve().parents):
                    try:
                        # logging.debug(f"AUDIO_PLAYER_DELETE_OUTPUT: Deleting played output file (initial): {audio_path_obj}")
                        audio_path_obj.unlink(missing_ok=True)
                    except Exception as e_del:
                        logging.warning(f"AUDIO_PLAYER_DELETE_FAIL: Could not delete {audio_path_obj}: {e_del}")


        except queue.Empty:
            if tts_actively_playing.is_set():
                # logging.debug("AUDIO_PLAYER_Q_EMPTY_CLEAR_STATE: Queue empty, TTS was active. Clearing state.")
                active_gen_id = None # Reset so next item starts fresh
                tts_actively_playing.clear()
                # last_tts_text = "" # Consider if last_tts_text should persist longer or clear here
                prev_tail = np.array([], dtype=np.int16) # Clear crossfade tail
        except Exception as e:
            logging.error(f"AUDIO_PLAYER_ERROR: {e}\n{traceback.format_exc()}", exc_info=True)
            if stream:
                try: stream.stop_stream(); stream.close()
                except: pass
                stream = None
            active_gen_id = None
            tts_actively_playing.clear()


    if stream:
        try: stream.stop_stream(); stream.close()
        except Exception as e_close: logging.error(f"AUDIO_PLAYER_STREAM_CLOSE_ERROR: {e_close}")
    pa.terminate()
    logging.info("AUDIO_PLAYER_SHUTDOWN: Audio player thread finished.")


def tts_requester_thread(text_chunk, generation_id):
    thread_name = threading.current_thread().name # Get assigned name
    try:
        # logging.debug(f"TTS_REQUESTER_START: Processing chunk: '{text_chunk[:50]}...' GenID: {generation_id}")
        if program_is_shutting_down.is_set():
            logging.info(f"TTS_REQUESTER_ABORT_SHUTDOWN: Aborting due to program shutdown: '{text_chunk[:50]}...'")
            return
        if current_generation_id and generation_id != current_generation_id:
            logging.info(f"TTS_REQUESTER_ABORT_STALE_ID: Aborting stale generation ID ({generation_id} vs {current_generation_id}): '{text_chunk[:50]}...'")
            return

        raw_audio_path = tts_request(text_chunk, generation_id) # This function now has logging

        if not raw_audio_path:
            logging.warning(f"TTS_REQUESTER_NO_AUDIO_PATH: TTS request returned no audio path for chunk: '{text_chunk[:50]}...'")
            return

        # logging.debug(f"TTS_REQUESTER_PREPROCESS_START: Preprocessing '{raw_audio_path.name}' for chunk: '{text_chunk[:50]}...'")
        preprocess_start_time = time.monotonic()
        try:
            raw_segment = AudioSegment.from_file(raw_audio_path, format="wav")
            processed_segment = preprocess_clip(
                seg=raw_segment, ref_f0=REFERENCE_F0, silence_thresh_db=SILENCE_THRES_DB,
                safe_pause_ms=SAFE_PAUSE_MS, edge_fade_ms=EDGE_FADE_MS,
                fade_in_ms=FADE_IN_MS, target_lufs=TARGET_LUFS
            )
            processed_segment.export(raw_audio_path, format="wav") # Overwrite with processed
        except Exception as e_proc:
            logging.error(f"TTS_REQUESTER_PREPROCESS_FAIL: Failed to preprocess {raw_audio_path.name}: {e_proc}", exc_info=True)
            try: raw_audio_path.unlink(missing_ok=True) # Attempt to clean up failed file
            except: pass
            return

        preprocess_duration_ms = (time.monotonic() - preprocess_start_time) * 1000
#         logging.info(f"TTS_REQUESTER_PREPROCESSED: Finished preprocessing ({preprocess_duration_ms:.0f}ms). Queuing audio '{raw_audio_path.name}' for: '{text_chunk[:50]}...'")

        audio_q.put((generation_id, raw_audio_path, text_chunk))
    except Exception as e:
        logging.error(f"TTS_REQUESTER_ERROR: Failed for chunk '{text_chunk[:50]}...': {e}\n{traceback.format_exc()}", exc_info=True)


def prefetch_worker():
    logging.info("PREFETCH_WORKER_READY: Prefetch worker started.")
    while not program_is_shutting_down.is_set():
        try:
            generation_id, chunks_to_prefetch = prefetch_q.get(timeout=0.2)
            # logging.debug(f"PREFETCH_WORKER_Q_GET: Got {len(chunks_to_prefetch)} chunk(s) from prefetch_q. First: '{chunks_to_prefetch[0][:50]}...' GenID: {generation_id}")

            if chunks_to_prefetch is None: # Shutdown signal
                logging.info("PREFETCH_WORKER_SHUTDOWN_SIGNAL: Received None, shutting down.")
                break
            if generation_id != current_generation_id:
                logging.info(f"PREFETCH_WORKER_SKIP_STALE_ID: Stale generation ID ({generation_id} vs {current_generation_id}). Discarding chunks.")
                prefetch_q.task_done()
                continue

            fetch_threads = []
            for i, text_chunk in enumerate(chunks_to_prefetch):
                if program_is_shutting_down.is_set() or interruption_requested.is_set():
                    logging.info("PREFETCH_WORKER_ABORT_CHUNK_LOOP: Aborting chunk processing due to shutdown/interruption.")
                    break
                if not text_chunk.strip() or re.fullmatch(r"<[^>]+>", text_chunk.strip()): # Skip empty or action-tag-only chunks
                    # logging.debug(f"PREFETCH_WORKER_SKIP_EMPTY_CHUNK: Skipping empty or meta chunk: '{text_chunk}'")
                    continue

                thread_name = f"TTSReq-{generation_id.hex[:4]}-{i}"
                # logging.info(f"PREFETCH_WORKER_SPAWN_TTS: Spawning TTS requester ({thread_name}) for chunk: '{text_chunk[:50]}...'")
                thread = threading.Thread(target=tts_requester_thread, args=(text_chunk, generation_id), name=thread_name)
                thread.daemon = True
                fetch_threads.append(thread)
                thread.start()

            for thread in fetch_threads:
                thread.join()
            prefetch_q.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"PREFETCH_WORKER_ERROR: {e}\n{traceback.format_exc()}", exc_info=True)
    logging.info("PREFETCH_WORKER_SHUTDOWN: Prefetch worker finished.")


def tts_request(txt, generation_id):
    if not txt.strip():
        logging.warning(f"TTS_API_SKIP_EMPTY: Skipping TTS request for empty text. GenID: {generation_id}")
        return None
    if current_generation_id and generation_id != current_generation_id:
        logging.warning(f"TTS_API_SKIP_STALE_ID: Skipping stale TTS request ({generation_id} vs {current_generation_id}) for: '{txt[:50]}...'")
        return None

    try:
        payload = {"input": txt, "model": "orpheus", "voice": VOICE, "response_format": "wav", "speed": 1.0}
        logging.info(f"TTS_API_SEND: Sending request to Orpheus: '{txt[:50]}...' GenID: {generation_id}")
        api_call_start_time = time.monotonic()
        response = requests.post(ORPHEUS_API_URL, json=payload, timeout=TTS_REQUEST_TIMEOUT)
        api_call_duration_ms = (time.monotonic() - api_call_start_time) * 1000
        logging.info(f"TTS_API_RECEIVE: Orpheus response status {response.status_code} ({api_call_duration_ms:.0f}ms) for: '{txt[:50]}...'")

        response.raise_for_status()
        wav_data = response.content

        if len(wav_data) < 1000:
            logging.warning(f"TTS_API_INVALID_DATA: Orpheus returned insufficient data (len: {len(wav_data)}) for: '{txt[:50]}...'")
            return None

        file_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}_{generation_id.hex[:6]}_{uuid.uuid4().hex[:4]}.wav"
        output_path = OUTPUT_DIR / file_name
        output_path.write_bytes(wav_data)
        # logging.debug(f"TTS_API_SAVE: Saved Orpheus audio to {output_path.name} for '{txt[:50]}...'")
        return output_path
    except requests.exceptions.Timeout:
        logging.error(f"TTS_API_TIMEOUT: Orpheus request timed out ({TTS_REQUEST_TIMEOUT}s) for '{txt[:50]}...'")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"TTS_API_ERROR: Orpheus request failed for '{txt[:50]}...': {e}")
        return None
    except Exception as e_unexp:
        logging.error(f"TTS_API_UNEXPECTED_ERROR: Unexpected error during TTS request for '{txt[:50]}...': {e_unexp}", exc_info=True)
        return None


def cleanup_old_audio():
    logging.info("CLEANUP_AUDIO_START: Cleaning up old audio files...")
    cleaned_count = 0
    for folder_to_clean in set(CLEANUP_FOLDERS): # Use set to avoid duplicates
        if not folder_to_clean.is_dir():
            # logging.debug(f"CLEANUP_AUDIO_SKIP_FOLDER: Folder not found, skipping: {folder_to_clean}")
            continue
        # logging.debug(f"CLEANUP_AUDIO_SCAN_FOLDER: Scanning folder: {folder_to_clean}")
        for f_path in folder_to_clean.glob("*.wav"):
            try:
                resolved_f_path = f_path.resolve()
                is_in_cache = False
                if CACHE_DIR.is_dir(): # Check if CACHE_DIR itself exists
                    if CACHE_DIR.resolve() == resolved_f_path.parent or CACHE_DIR.resolve() in resolved_f_path.parents:
                         is_in_cache = True

                if is_in_cache:
                    # logging.debug(f"CLEANUP_AUDIO_KEEP_CACHE: Keeping cached file: {resolved_f_path}")
                    continue

                # logging.debug(f"CLEANUP_AUDIO_DELETE: Deleting old audio file: {resolved_f_path}")
                f_path.unlink(missing_ok=True) # Delete the file
                cleaned_count +=1
            except Exception as e:
                logging.warning(f"CLEANUP_AUDIO_FAIL: Could not delete {f_path}: {e}")
    logging.info(f"CLEANUP_AUDIO_DONE: Finished cleanup. Deleted {cleaned_count} file(s).")


def hermes_chat(msgs):
    try:
        request_time_outer = time.monotonic()
        buffer = ""
        first_token_received = False
        timeout_config = httpx.Timeout(10.0, read=60.0)

        with httpx.Client(timeout=timeout_config, trust_env=False) as client:
            request_time_inner = time.monotonic() # Time just before client.stream()
            with client.stream("POST",
                               LM_STUDIO_CHAT_URL,
                               json={"model": "grok-3-reasoning-gemma3-12b-distilled", "messages": msgs, "temperature": 0.6, "stream": True}
                               ) as r:

                time_before_raise_status = time.monotonic()
                r.raise_for_status() # This confirms headers are received and status is OK
                time_after_raise_status = time.monotonic()

                headers_received_latency_ms = (time_after_raise_status - request_time_inner) * 1000
                logging.info(f"LLM_HEADERS_RECEIVED (httpx.Client): HTTP headers received and status OK ({headers_received_latency_ms:.0f}ms from client.stream call)")

                for chunk in r.iter_text():
                    if not first_token_received and chunk.strip():
                        ttft_from_headers_ok = (time.monotonic() - time_after_raise_status) * 1000
                        ttft_overall = (time.monotonic() - request_time_outer) * 1000

                        logging.info(f"LLM_STREAM_TTFT (httpx.Client): First token data received. Overall TTFT: {ttft_overall:.0f}ms. From Headers OK: {ttft_from_headers_ok:.0f}ms. Chunk: '{chunk.strip()[:60]}'")
                        first_token_received = True

                    if interruption_requested.is_set() or program_is_shutting_down.is_set():
                        logging.info("LLM_STREAM_ABORT (httpx.Client): Aborting LLM stream.")
                        break

                    buffer += chunk
                    while 'data: ' in buffer and '\n' in buffer:
                        event_start = buffer.find('data: ')
                        line_end = buffer.find('\n', event_start)
                        if line_end == -1: break
                        line = buffer[event_start:line_end]
                        buffer = buffer[line_end+1:]
                        try:
                            data_str = line.split('data: ', 1)[1].strip()
                            if data_str == '[DONE]': return
                            if data_str:
                                delta = json.loads(data_str)['choices'][0]['delta']
                                if 'content' in delta and delta['content'] is not None:
                                    yield delta['content']
                        except (json.JSONDecodeError, IndexError, KeyError): pass
    except httpx.RequestError as e_req:
        if not program_is_shutting_down.is_set():
            logging.error(f"LLM_STREAM_REQUEST_ERROR (httpx.Client): LLM request failed: {e_req}", exc_info=True)
    except httpx.HTTPStatusError as e_stat:
        if not program_is_shutting_down.is_set():
            logging.error(f"LLM_STREAM_HTTP_ERROR (httpx.Client): LLM request failed with status {e_stat.response.status_code}: {e_stat.response.text}", exc_info=True)
    except Exception as e_unexp:
        if not program_is_shutting_down.is_set():
            logging.error(f"LLM_STREAM_UNEXPECTED_ERROR (httpx.Client): {e_unexp}\n{traceback.format_exc()}", exc_info=True)


def summarise(prev_summary, new_transcript):
    logging.info(f"SUMMARY_START: Summarizing transcript. Prev summary len: {len(prev_summary or '')}, New transcript len: {len(new_transcript)}")
    try:
        content = f"PREVIOUS SUMMARY:\n{prev_summary or 'None.'}\n\nNEW TRANSCRIPT TO ADD:\n{new_transcript}"
        if not SUMMARY_PROMPT:
            logging.error("SUMMARY_FAIL_NO_PROMPT: SUMMARY_PROMPT is not loaded.")
            return prev_summary or "[summary failed: no prompt]"

        r = requests.post(LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}], "temperature": 0.6, "max_tokens": 1024}, timeout=200)
        r.raise_for_status()
        r.encoding = 'utf-8'
        summary_text = r.json()['choices'][0]['message']['content'].strip()
        final_summary = summary_text.replace("Updated Summary:", "").strip()
        logging.info(f"SUMMARY_DONE: Summarization complete. New summary len: {len(final_summary)}")
        return final_summary
    except Exception as e:
        logging.error(f"SUMMARY_FAIL: Summarization failed: {e}", exc_info=True)
        return prev_summary or "[summary failed: exception]"


def play_startup_greeting_thread(audio_path_obj):
    try:
        logging.info(f"GREETING_PLAY_START: Playing startup greeting: {audio_path_obj.name}")
        data_audio, sr_audio = sf.read(audio_path_obj, dtype='float32') # float32 for sd.play
        sd.play(data_audio, sr_audio)
        sd.wait()
        logging.info(f"GREETING_PLAY_DONE: Finished playing startup greeting: {audio_path_obj.name}")
    except Exception as e:
        logging.warning(f"GREETING_PLAY_ERROR: Error playing greeting '{audio_path_obj.name}': {e}", exc_info=True)
        # print(f"⚠️ Error playing greeting '{audio_path_obj.name}': {e}", file=sys.stderr) # UI


def warmup_tts():
    logging.info("TTS_WARMUP: Sending warmup request to Orpheus TTS...")
    try:
        warmup_text = "The system is ready."
        payload = {"input": warmup_text, "model": "orpheus", "voice": VOICE, "response_format": "wav", "speed": 1.0}
        response = requests.post(ORPHEUS_API_URL, json=payload, timeout=(5,10))
        response.raise_for_status()
        logging.info(f"TTS_WARMUP: Warmup request with '{warmup_text}' successful (status {response.status_code}). Duration: {response.elapsed.total_seconds()*1000:.0f}ms")
    except Exception as e:
        logging.warning(f"TTS_WARMUP_FAIL: Orpheus TTS warmup request failed: {e}")

def main():
    global current_generation_id, REFERENCE_F0, PERSONA_TEMPLATE, SUMMARY_PROMPT,llm_classifier_instance # Ensure globals are intended

    # Logging config is now at the top
    logging.info("MAIN_INIT: Application starting...")
    cleanup_old_audio() # Initial cleanup
    interruption_requested.clear(); tts_actively_playing.clear(); program_is_shutting_down.clear()
    chat_history=[{"role":"system","content":""}]; current_memories = "" # Initialize chat history and memories

    # Load Persona and Summary Prompts (global vars are already set if __name__ == "__main__" ran)
    if not PERSONA_TEMPLATE:
        logging.critical("MAIN_FATAL_NO_PERSONA: PERSONA_TEMPLATE is not loaded. Exiting.")
        sys.exit(1)
    if not SUMMARY_PROMPT:
        logging.critical("MAIN_FATAL_NO_SUMMARY_PROMPT: SUMMARY_PROMPT is not loaded. Exiting.")
        sys.exit(1)
    chat_history[0]["content"] = PERSONA_TEMPLATE
    if not MOOD_CLASSIFIER_BOT.exists():
        logging.critical(f"MAIN_FATAL_NO_MOOD_PROMPT: MOOD_CLASSIFIER_BOT file not found at {MOOD_CLASSIFIER_BOT}. Mood classification will be disabled. Exiting.")
        sys.exit(1)

    # --- Initialize LLM Style Classifier ---
    try:
        logging.info("MAIN_INIT_CLASSIFIER: Initializing LLMStyleClassifier...")
        llm_classifier_instance = LLMStyleClassifier(system_prompt_path=MOOD_CLASSIFIER_BOT)
        logging.info("MAIN_INIT_CLASSIFIER_DONE: LLMStyleClassifier initialized successfully.")
    except Exception as e:
        logging.critical(f"MAIN_FATAL_CLASSIFIER_INIT: Failed to initialize LLMStyleClassifier even though prompt file exists: {e}. Exiting.", exc_info=True)
        sys.exit(1)

    if CACHE_DIR.is_dir():
        interrupt_audio_files = list(CACHE_DIR.glob("*_interrupt_cache.wav"))
        greeting_files = list(CACHE_DIR.glob("*_greeting_cache.wav"))
        logging.info(f"MAIN_CACHE_SCAN: Found {len(interrupt_audio_files)} interrupt clips and {len(greeting_files)} greeting clips.")
    else:
        logging.warning(f"MAIN_CACHE_NOT_FOUND: Cache directory '{CACHE_DIR}' not found. No cached clips loaded.")
        interrupt_audio_files, greeting_files = [], []

    if greeting_files:
        chosen_greeting_path = random.choice(greeting_files)
        logging.info(f"MAIN_GREETING_SELECTED: Selected greeting: {chosen_greeting_path.name}")
        greeting_thread = threading.Thread(target=play_startup_greeting_thread, args=(chosen_greeting_path,), daemon=True, name="GreetingPlayer")
        greeting_thread.start()
    else:
        logging.info("MAIN_NO_GREETING: No greeting files found to play.")


    def memory_manager_worker():
        nonlocal current_memories # Closure for current_memories
        logging.info("MEMORY_WORKER_READY: Memory manager worker started.")
        while not program_is_shutting_down.is_set():
            try:
                prev_mems, transcript_to_add = memory_q.get(timeout=0.2)
                if prev_mems is None and transcript_to_add is None : # Shutdown signal
                    logging.info("MEMORY_WORKER_SHUTDOWN_SIGNAL: Received None, shutting down.")
                    break
                # logging.debug(f"MEMORY_WORKER_Q_GET: Received transcript of len {len(transcript_to_add)} for summarization.")
                current_memories = summarise(prev_mems, transcript_to_add) # summarise has its own logging
                memory_q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"MEMORY_WORKER_ERROR: {e}\n{traceback.format_exc()}", exc_info=True)
        logging.info("MEMORY_WORKER_SHUTDOWN: Memory manager worker finished.")

    thread_targets = [
        (audio_player_thread, "AudioPlayer"),
        (prefetch_worker, "Prefetcher"),
        (asr_listener, "ASRListener"),
        (kb_listener, "KBListener"),
        (memory_manager_worker, "MemoryManager")
    ]
    threads = []
    for target_func, name in thread_targets:
        thread = threading.Thread(target=target_func, name=name, daemon=True)
        threads.append(thread)
        thread.start()
        logging.info(f"MAIN_THREAD_START: Started thread: {name}")

    warmup_tts()

    print("🎙️ System ready. Talk or type anytime. Check logs for detailed info.") # UI message

    try:
        while not program_is_shutting_down.is_set():
            try:
                # logging.debug("MAIN_LOOP_WAIT_USER_INPUT: Waiting for user input from user_q...")
                current_user_input, input_duration_s = user_q.get(timeout=0.2) # input_duration_s for speech, 0 for text
                # logging.info(f"MAIN_USER_INPUT_RECEIVED: Raw input from user_q: '{current_user_input}'")
                user_q.task_done()
            except queue.Empty:
                if program_is_shutting_down.is_set(): break
                continue

            turn_start_time = time.monotonic()
            logging.info(f"MAIN_TURN_START: Processing user input: '{current_user_input[:100]}...'")

            resolved_tag = "neutral" # Default if classification fails or is skipped
            mood_info = {"tag": "neutral", "score": 0.0, "source": "default_fallback"} # Provide a default mood_info

            if llm_classifier_instance: # Check if LLMStyleClassifier was initialized
                try:
                    mood_info = llm_classifier_instance.get_classification(current_user_input)
                    resolved_tag = mood_info["tag"]

                    if mood_info["source"] != "llm_classifier": # Check if classification was fully successful
                        logging.warning(f"LLM classification issue: tag='{mood_info['tag']}', source='{mood_info['source']}'. Defaulting resolved_tag to 'neutral'.")
                        resolved_tag = "neutral"

                    logging.info(
                        f"MOOD_CLASSIFIED_LLM: tag='{mood_info['tag']}' (score={mood_info['score']:.1f}, source='{mood_info['source']}') → resolved_tag: '{resolved_tag}' for: '{current_user_input}'"
                    )
                except Exception as e:
                    logging.warning(f"MOOD_CLASSIFIER_ERROR (LLM Call): Uncaught exception during get_classification: {e}", exc_info=True)
            else:
                logging.warning("MOOD_CLASSIFICATION_SKIPPED: LLMStyleClassifier not initialized.")

            if interruption_requested.is_set():
                logging.info("MAIN_INTERRUPT_HANDLING_START: Interruption was requested. Clearing queues and playing sound.")
                sd.stop() # Stop any ongoing sounddevice playback
                # Clear queues more robustly
                queues_to_clear = [audio_q, prefetch_q] # Don't clear user_q or memory_q here usually
                for q_to_clear in queues_to_clear:
                    # logging.debug(f"MAIN_INTERRUPT_CLEAR_Q: Clearing queue {q_to_clear}")
                    while not q_to_clear.empty():
                        try: q_to_clear.get_nowait(); q_to_clear.task_done()
                        except queue.Empty: break
                        except Exception as e_q_clear: logging.warning(f"MAIN_INTERRUPT_Q_CLEAR_ITEM_FAIL: {e_q_clear}")

                tts_actively_playing.clear() # Ensure this is cleared

                if interrupt_audio_files:
                    try:
                        interrupt_clip_path = random.choice(interrupt_audio_files)
                        # logging.info(f"MAIN_INTERRUPT_PLAY_SOUND: Playing interrupt sound: {interrupt_clip_path.name}")
                        data_int, sr_int = sf.read(interrupt_clip_path, dtype='float32')
                        sd.play(data_int, sr_int); sd.wait()
                    except Exception as e:
                        logging.warning(f"MAIN_INTERRUPT_SOUND_FAIL: Could not play interrupt clip: {e}", exc_info=True)
                        # print(f"‼️ Could not play interrupt clip: {e}", file=sys.stderr)

                interruption_requested.clear() # Clear the event AFTER handling
                logging.info("MAIN_INTERRUPT_HANDLING_DONE: Interruption handling complete.")

                # If the input that caused interrupt was just a keyword, don't process it further as LLM input
                normalized_interrupt_input = current_user_input.lower().strip(".?!, ")
                if any(keyword in normalized_interrupt_input for keyword in INTERRUPT_KEYWORDS):
                    logging.info("MAIN_INTERRUPT_INPUT_IS_KEYWORD: Input was an interrupt keyword, skipping LLM response for this input.")
                    current_generation_id = None # Invalidate current gen ID
                    continue # Go back to waiting for new user input

            # If we are here, either no interruption, or interruption was handled and input is not just a keyword.
            current_generation_id = uuid.uuid4() # New ID for this new conversational turn / LLM response
            logging.info(f"MAIN_NEW_GEN_ID: Set new generation ID: {current_generation_id.hex[:8]}")

            # Queue fillers immediately
            # logging.debug(f"MAIN_FILLER_QUEUE_START: Queuing fillers for gen_id {current_generation_id.hex[:8]}")
            fillers_queued_count = 0
#             for item in filler_builder.build_queue_items(): # This should be quick
#                 if isinstance(item, Path): # It's an audio file path
#                     audio_q.put((current_generation_id, item, "[FILLER]"))
#                     fillers_queued_count +=1
#                 elif isinstance(item, int): # It's a pause duration in ms
#                     audio_q.put((current_generation_id, item, "[PAUSE]"))
                    # fillers_queued_count +=1 # Or count pauses differently
            # logging.info(f"MAIN_FILLER_QUEUE_DONE: Queued {fillers_queued_count} filler/pause items.")


            if current_user_input.lower() in {"quit","exit"}:
                logging.info("MAIN_USER_QUIT: User requested quit. Shutting down.")
                program_is_shutting_down.set()
                break
            if current_user_input.lower() == "mem":
                logging.info(f"MAIN_CMD_MEM: Displaying current memories (len: {len(current_memories)}).")
                print(f"\n--- MEMORY ---\n{current_memories or '[none]'}\n---")
                continue
            if current_user_input.lower() == "dump":
                logging.info("MAIN_CMD_DUMP: Dumping chat history.")
                print("\n--- CHAT HISTORY DUMP ---")
                for m in chat_history: print(f"[{m['role']}] {m['content']}")
                print("-------------------------")
                continue

            # Prepare messages for LLM
            # logging.debug("MAIN_LLM_PREP_MSG: Preparing messages for LLM.")
            system_prompt_content = PERSONA_TEMPLATE + (f"\n\n--- CURRENT CONVERSATION SUMMARY ---\n{current_memories}" if current_memories else "")

            messages_for_llm = list(chat_history) # Create a mutable copy
            messages_for_llm[0] = {"role": "system", "content": system_prompt_content} # Update/set system prompt
            messages_for_llm.append({"role": "user", "content": current_user_input})
            # logging.debug(f"MAIN_LLM_PREP_MSG_DONE: System prompt length: {len(system_prompt_content)}, Total messages for LLM: {len(messages_for_llm)}")


            llm_full_response = ""
            llm_stream_initiated_time = None
            first_chunk_to_prefetch_time = None
            try:
                chunker_instance = StreamingChunker(min_words=MIN_WORDS_FOR_CHUNK) # Create new instance per response
                # logging.info("MAIN_LLM_CALL_START: Calling hermes_chat (LLM).")
                llm_stream_initiated_time = time.monotonic()
                token_stream = (token for token in hermes_chat(messages_for_llm)) # hermes_chat has logging

                # logging.debug("MAIN_LLM_CHUNKER_START: Starting to process token stream with StreamingChunker.")
                first_chunk_generated_for_tts = False
                for sanitized_chunk in chunker_instance.chunker(token_stream):
                    if not first_chunk_generated_for_tts:
                        first_chunk_to_prefetch_time = time.monotonic()
                        latency_first_chunk = (first_chunk_to_prefetch_time - llm_stream_initiated_time) * 1000
                        logging.info(f"MAIN_LLM_FIRST_CHUNK_READY: First sanitized chunk for TTS ready ({latency_first_chunk:.0f}ms from LLM call start): '{sanitized_chunk[:60]}...'")
                        first_chunk_generated_for_tts = True

                    if interruption_requested.is_set(): # Check before queuing to prefetch
                        logging.info("MAIN_LLM_CHUNK_INTERRUPT: Interruption detected during LLM chunk processing. Breaking from chunker loop.")
                        break

                    # logging.debug(f"MAIN_LLM_CHUNK_TO_PREFETCH: Queueing chunk to prefetch_q: '{sanitized_chunk[:50]}...' GenID: {current_generation_id.hex[:8]}")
                    prefetch_q.put((current_generation_id, [sanitized_chunk])) # Prefetch worker expects a list
                    llm_full_response += sanitized_chunk + " " # Append for full response tracking

                # After loop finishes or breaks
                if interruption_requested.is_set():
                    logging.info("MAIN_LLM_STREAM_HANDLED_INTERRUPT: LLM stream processing was interrupted.")
                    # History will be updated with user input if it wasn't just a keyword. Assistant response part might be partial or none.
                elif not first_chunk_generated_for_tts:
                    logging.warning("MAIN_LLM_NO_CHUNKS_GENERATED: LLM stream ended or was processed by chunker without yielding any usable chunks for TTS.")
                # else:
                    # logging.info("MAIN_LLM_CHUNKER_DONE: Finished consuming LLM stream via chunker.")

            except Exception as e:
                logging.error(f"MAIN_LLM_STREAM_ERROR: Error during LLM streaming/chunking: {e}\n{traceback.format_exc()}", exc_info=True)
                # print(f"\n‼️ Error during LLM streaming/chunking: {e}", file=sys.stderr)
                # Still add user input to history even if LLM fails
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                continue # Skip to next user input cycle

            # --- Post-LLM Processing ---
            # Add user input to history if not an interrupt keyword that was already skipped
            # (If interrupt happened and input was NOT a keyword, it's processed up to here, and user input should be added)
            # This logic might need refinement based on desired interrupt behavior regarding history.
            # Assuming if we got here and didn't 'continue' due to interrupt keyword, user input is part of convo.
            if not (interruption_requested.is_set() and any(keyword in current_user_input.lower().strip(".?!, ") for keyword in INTERRUPT_KEYWORDS)):
                 if current_user_input: # Ensure it's not empty
                    chat_history.append({"role": "user", "content": current_user_input})


            assistant_response_for_chat_history = strip_action_and_emoji(llm_full_response).strip()
            if assistant_response_for_chat_history: # Only add if there's actual content
                logging.info(f"MAIN_ASSISTANT_RESPONSE_FULL_CLEAN: '{assistant_response_for_chat_history[:100]}...'")
                print(f"🤖 Assistant: {assistant_response_for_chat_history}") # UI
                chat_history.append({"role": "assistant", "content": assistant_response_for_chat_history})
            else:
                logging.info("MAIN_ASSISTANT_RESPONSE_EMPTY: No textual assistant response generated or all stripped.")
                # If LLM response was empty or only action tags, and an interruption happened,
                # user input might already be added. If no interruption, and LLM is just silent, add user input.
                # The 'if current_user_input:' above handles adding user msg if assistant is silent.

            # Token/History Management
            total_tokens_in_history = sum(rough_tokens(m['content']) for m in chat_history)
            # logging.debug(f"MAIN_HISTORY_TOKEN_CHECK: Total tokens: {total_tokens_in_history}, Soft limit: {TOKEN_SOFT_LIMIT}, Hard: {TOKEN_HARD_LIMIT}")
            if total_tokens_in_history > TOKEN_SOFT_LIMIT:
                messages_to_retain_count = KEEP_RECENT * 2 # user + assistant pairs
                if len(chat_history) > messages_to_retain_count + 1: # +1 for system prompt
                    # logging.info(f"MAIN_HISTORY_SUMMARIZE_TRIGGER: Token count {total_tokens_in_history} > {TOKEN_SOFT_LIMIT}. Summarizing older messages.")
                    # System prompt is chat_history[0]
                    messages_to_summarize = chat_history[1 : -(messages_to_retain_count)] # Exclude system and recent N pairs
                    if messages_to_summarize:
                        transcript_to_summarize = "\n".join(f"{m['role']}: {m['content']}" for m in messages_to_summarize)
                        # logging.debug(f"MAIN_HISTORY_SUMMARIZE_SEND: Sending {len(messages_to_summarize)} messages to memory_q for summarization.")
                        memory_q.put((current_memories, transcript_to_summarize)) # Send current mems and new text
                        # Rebuild chat history: system_prompt + N most recent messages
                        chat_history = [chat_history[0]] + chat_history[-(messages_to_retain_count):]
                        # logging.info(f"MAIN_HISTORY_SUMMARIZED: Chat history rebuilt. New length: {len(chat_history)}")

            # Hard token limit enforcement (less likely if soft limit summarization works)
            while sum(rough_tokens(m['content']) for m in chat_history) > TOKEN_HARD_LIMIT:
                if len(chat_history) > 2:
                    logging.warning(f"MAIN_HISTORY_HARD_LIMIT_TRIM: Hard token limit exceeded. Removing oldest message after system prompt.")
                    del chat_history[1]
                else:
                    logging.warning("MAIN_HISTORY_HARD_LIMIT_UNABLE_TO_TRIM: Hard token limit hit, but history too short to trim further.")
                    break

            turn_duration_ms = (time.monotonic() - turn_start_time) * 1000
            logging.info(f"MAIN_TURN_END: Finished processing turn. Duration: {turn_duration_ms:.0f}ms.")


    except KeyboardInterrupt:
        logging.info("\nMAIN_KEYBOARD_INTERRUPT: Keyboard interrupt received. Shutting down...")
        print("\n🚨 Shutting down...")
    except Exception as e:
        logging.critical(f"MAIN_UNHANDLED_EXCEPTION: {e}\n{traceback.format_exc()}", exc_info=True)
        # print(f"‼️ UNHANDLED EXCEPTION IN MAIN: {e}"); traceback.print_exc() # UI
    finally:
        logging.info("MAIN_FINALIZING: Initiating shutdown sequence for all threads...")
        program_is_shutting_down.set() # Signal all threads to shut down

        # logging.debug("MAIN_FINALIZING_SENTINELS: Sending shutdown sentinels to queues.")
        audio_q.put((None, None, None))
        prefetch_q.put((None, None))
        memory_q.put((None, None))
        # user_q is typically emptied by main loop, or input() in kb_listener will break

        for t in threads:
            # logging.debug(f"MAIN_FINALIZING_JOIN_THREAD: Joining thread {t.name}...")
            t.join(timeout=2.0)
            if t.is_alive():
                logging.warning(f"MAIN_FINALIZING_THREAD_ALIVE: Thread {t.name} did not shut down cleanly.")

        # sd.stop() # Ensure sounddevice is stopped if not already.
        cleanup_old_audio() # Final cleanup
        logging.info("MAIN_SHUTDOWN_COMPLETE: Application has shut down.")
        print("👋 Goodbye!")


if __name__ == "__main__":
    PERSONA_TEMPLATE = ""
    SUMMARY_PROMPT = ""
    try:
        with open(PERSONA_PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
            PERSONA_TEMPLATE = f.read()
        with open(SUMMARY_BOT_TEMPLATE, "r", encoding="utf-8") as f:
            SUMMARY_PROMPT = f.read()
        logging.info("MAIN_PROMPTS_LOADED: Persona and summary prompts loaded successfully.")
        main()
    except FileNotFoundError as e:
        logging.critical(f"MAIN_FATAL_PROMPT_MISSING: A required persona/summary file is missing: {e}. Please check paths.", exc_info=True)
        # print(f"‼️ FATAL: A required persona file is missing: {e}. Please check your paths.") # UI
        sys.exit(1)
    except Exception as e:
        logging.critical(f"MAIN_FATAL_STARTUP_ERROR: An unexpected startup error occurred: {e}\n{traceback.format_exc()}", exc_info=True)
        # print(f"‼️ An unexpected startup error occurred: {e}"); traceback.print_exc() # UI
        sys.exit(1)