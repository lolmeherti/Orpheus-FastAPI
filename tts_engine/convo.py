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
import whisper

# Audio libraries
import sounddevice as sd
import soundfile as sf
import pyaudio
from pydub import AudioSegment

# --- Custom Library Imports ---
from streaming_chunker import StreamingChunker
from audio_processing import preprocess_clip
from prosody_fast import get_f0
from filler_queue_builder import FillerQueueBuilder

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
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
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
REFERENCE_F0 = 0.0 # Pitch correction disabled by default

# --- Global State ---
audio_q, prefetch_q, user_q, memory_q = (queue.Queue() for _ in range(4))
interruption_requested, tts_actively_playing, program_is_shutting_down = threading.Event(), threading.Event(), threading.Event()
INTERRUPT_KEYWORDS = {"stop", "shut up", "hold on", "wait", "enough", "nevermind", "cancel"}
last_tts_text = ""
current_generation_id = None

# --- Helper Functions ---
def strip_action_and_emoji(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"<[^>]+>", '', text)
    text = re.sub(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]', '', text)
    text = re.sub(r'<\|.*?\|>', '', text)
    return " ".join(text.split())

def rough_tokens(txt: str) -> int:
    return max(1, len(txt) // 4)

# --- Worker Threads ---

def asr_listener():
    try:
        asr_model = whisper.load_model("small.en")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    except Exception as e:
        print(f"‼️ FATAL: Could not load ASR/VAD models: {e}"); program_is_shutting_down.set(); return

    sr, vad_chunk_size = 16000, 512
    is_speaking, audio_buffer, silence_counter = False, [], 0

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=vad_chunk_size) as stream:
            while not program_is_shutting_down.is_set():
                frame, _ = stream.read(vad_chunk_size)
                is_speech = vad_model(torch.from_numpy(frame.flatten()), sr).item() > VAD_SPEECH_CONFIDENCE_THRESHOLD
                if is_speaking:
                    audio_buffer.append(frame)
                    if not is_speech:
                        silence_counter += 1
                        active_timeout_s = VAD_INTERRUPT_TIMEOUT_S if tts_actively_playing.is_set() else VAD_DEFAULT_SILENCE_TIMEOUT_S
                        if len(audio_buffer) * (vad_chunk_size / sr) > VAD_LONG_PROMPT_TRIGGER_S:
                            active_timeout_s = VAD_LONG_PROMPT_TIMEOUT_S

                        if silence_counter * (vad_chunk_size / sr) >= active_timeout_s:
                            is_speaking = False
                            full_audio = np.concatenate(audio_buffer).squeeze()
                            audio_buffer = []
                            if len(full_audio) / sr >= VAD_MIN_SPEECH_S:
                                txt = asr_model.transcribe(full_audio, fp16=torch.cuda.is_available())['text'].strip()
                                if txt:
                                    is_echo = SequenceMatcher(None, txt.lower(), last_tts_text.lower()).ratio() > ECHO_SIMILARITY_THRESHOLD if last_tts_text else False
                                    if not is_echo and len(txt.split()) >= MIN_WORDS_ASR:
                                        print(f"\n🗣️ User: '{txt}'")
                                        if tts_actively_playing.is_set():
                                            interruption_requested.set()
                                        user_q.put((txt, len(full_audio) / sr))
                    else:
                        silence_counter = 0
                elif is_speech:
                    is_speaking, silence_counter = True, 0
                    audio_buffer = [frame]
    except Exception as e:
        if not program_is_shutting_down.is_set(): print(f"‼️ ASR Error: {e}")

def kb_listener():
    while not program_is_shutting_down.is_set():
        try:
            line = input("\n⌨️  ").strip()
            if program_is_shutting_down.is_set(): break
            if line:
                if tts_actively_playing.is_set() and line.lower() in INTERRUPT_KEYWORDS:
                    interruption_requested.set()
                user_q.put((line, 0.0))
        except (EOFError, KeyboardInterrupt):
            break

def audio_player_thread():
    global last_tts_text
    pa = pyaudio.PyAudio()
    stream = None
    active_gen_id = None
    current_stream_rate = None
    prev_tail = np.array([], dtype=np.int16)

    def open_stream(rate):
        nonlocal stream, current_stream_rate
        if stream: stream.stop_stream(); stream.close()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True, frames_per_buffer=int(rate * CHUNK_MS / 1000))
        current_stream_rate = rate
        return stream

    def play_audio_data(data, sr):
        nonlocal stream, prev_tail
        if not stream or current_stream_rate != sr: stream = open_stream(sr)
        fade_len = min(len(prev_tail), len(data))
        if fade_len > 0:
            fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            blend = (prev_tail[:fade_len] * fade_out + data[:fade_len] * fade_in).astype(np.int16)
            data_to_play = np.concatenate((blend, data[fade_len:]))
        else:
            data_to_play = data
        stream.write(data_to_play.tobytes())
        tail_size = int(sr * CROSSFADE_MS / 1000)
        prev_tail = data[-tail_size:]

    while not program_is_shutting_down.is_set():
        try:
            gen_id, data, text = audio_q.get(timeout=1.0)
            if data is None: break # Shutdown signal

            if text == "[PAUSE]":
                if not interruption_requested.is_set():
                    time.sleep(data / 1000.0)
                audio_q.task_done()
                continue

            p_path_obj = data

            try:
                if gen_id != active_gen_id:
                    active_gen_id = gen_id; prev_tail = np.array([], dtype=np.int16)
                if interruption_requested.is_set(): continue

                tts_actively_playing.set()
                was_filler = text == "[FILLER]"

                if was_filler:
                    logging.info(f"⏳ Filler: {p_path_obj.name}")
                else:
                    logging.info(f"🎵 TTS: '{text}'")
                    last_tts_text = text

                current_samples, sr = sf.read(str(p_path_obj), dtype='int16')
                play_audio_data(current_samples, sr)

                while not interruption_requested.is_set():
                    try:
                        next_gen_id, next_data, next_text = audio_q.get(timeout=0.65)
                        if next_gen_id != gen_id:
                            audio_q.put((next_gen_id, next_data, next_text)); break

                        if next_text == "[PAUSE]":
                            if not interruption_requested.is_set():
                                time.sleep(next_data / 1000.0)
                            audio_q.task_done()
                            continue

                        next_path = next_data

                        if next_text == "[FILLER]":
                             logging.info(f"⏳ Filler: {next_path.name}")
                        else:
                             logging.info(f"🎵 TTS: '{next_text}'")
                             last_tts_text = next_text

                        tts_samples, tts_sr = sf.read(str(next_path), dtype='int16')
                        if tts_sr == sr:
                            play_audio_data(tts_samples, sr)
                        else:
                            play_audio_data(tts_samples, tts_sr)
                            sr = tts_sr

                        if next_path.exists() and not next_path.is_relative_to(CACHE_DIR):
                            next_path.unlink(missing_ok=True)
                        audio_q.task_done()
                    except queue.Empty: break
            finally:
                audio_q.task_done()

        except queue.Empty:
            if tts_actively_playing.is_set():
                active_gen_id = None; tts_actively_playing.clear(); last_tts_text = ""
        except Exception as e:
            print(f"‼️ Audio Player Error: {e}")

    if stream: stream.close()
    pa.terminate()


def tts_requester_thread(text_chunk, generation_id):
    try:
        if program_is_shutting_down.is_set() or (current_generation_id and generation_id != current_generation_id): return
        raw_audio_path = tts_request(text_chunk, generation_id)
        if not raw_audio_path: return

        raw_segment = AudioSegment.from_file(raw_audio_path, "wav")
        processed_segment = preprocess_clip(
            seg=raw_segment, ref_f0=REFERENCE_F0, silence_thresh_db=SILENCE_THRES_DB,
            safe_pause_ms=SAFE_PAUSE_MS, edge_fade_ms=EDGE_FADE_MS,
            fade_in_ms=FADE_IN_MS, target_lufs=TARGET_LUFS
        )
        processed_segment.export(raw_audio_path, format="wav")
        audio_q.put((generation_id, raw_audio_path, text_chunk))
    except Exception as e:
        print(f"‼️ TTS Requester Error for '{text_chunk[:30]}...': {e}")

def prefetch_worker():
    while not program_is_shutting_down.is_set():
        try:
            generation_id, chunks_to_prefetch = prefetch_q.get(timeout=0.2)
            if chunks_to_prefetch is None: break
            if generation_id != current_generation_id:
                prefetch_q.task_done(); continue

            fetch_threads = []
            for text_chunk in chunks_to_prefetch:
                if program_is_shutting_down.is_set() or interruption_requested.is_set(): break
                if not text_chunk.strip() or re.fullmatch(r"<[^>]+>", text_chunk.strip()): continue
                thread = threading.Thread(target=tts_requester_thread, args=(text_chunk, generation_id))
                thread.daemon = True; fetch_threads.append(thread); thread.start()
            for thread in fetch_threads: thread.join()
            prefetch_q.task_done()
        except queue.Empty: continue

def tts_request(txt, generation_id):
    if not txt or (current_generation_id and generation_id != current_generation_id): return None
    try:
        payload = {"input": txt, "model": "orpheus", "voice": VOICE, "response_format": "wav", "speed": 1.0}
        wav_data = requests.post(ORPHEUS_API_URL, json=payload, timeout=TTS_REQUEST_TIMEOUT).content
        if len(wav_data) < 1000: return None
        file_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUT_DIR / file_name
        output_path.write_bytes(wav_data)
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"‼️ TTS API failed for '{txt[:30]}...': {e}"); return None

def cleanup_old_audio():
    for folder in set(CLEANUP_FOLDERS):
        if not folder.is_dir(): continue
        for f in folder.glob("*.wav"):
            try:
                resolved_f = f.resolve()
                if (CACHE_DIR.is_dir() and CACHE_DIR.resolve() in resolved_f.parents):
                    continue
                f.unlink(missing_ok=True)
            except Exception as e: print(f"🧹 Could not delete {f}: {e}")

def hermes_chat(msgs):
    try:
        r = requests.post(LM_STUDIO_CHAT_URL, json={"model":"hermes","messages":msgs,"temperature":0.9,"stream":True}, stream=True,timeout=(10,60))
        r.raise_for_status()
        r.encoding = 'utf-8'
        buffer = ""
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if interruption_requested.is_set() or program_is_shutting_down.is_set(): break
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
                        if 'content' in delta: yield delta['content']
                except (json.JSONDecodeError, IndexError): pass
    except Exception as e:
        if not program_is_shutting_down.is_set(): print(f"‼️ Hermes Error: {e}")

def summarise(prev_summary, new_transcript):
    try:
        content = f"PREVIOUS SUMMARY:\n{prev_summary or 'None.'}\n\nNEW TRANSCRIPT TO ADD:\n{new_transcript}"
        r = requests.post(LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}], "temperature": 0.6, "max_tokens": 1024}, timeout=200)
        r.raise_for_status()
        r.encoding = 'utf-8'
        summary_text = r.json()['choices'][0]['message']['content'].strip()
        return summary_text.replace("Updated Summary:", "").strip()
    except Exception: return prev_summary or "[summary failed]"

def play_startup_greeting_thread(audio_path_obj):
    try:
        data_audio, sr_audio = sf.read(audio_path_obj, dtype='float32')
        sd.play(data_audio, sr_audio)
        sd.wait()
    except Exception as e: print(f"⚠️ Error playing greeting '{audio_path_obj.name}': {e}", file=sys.stderr)

def main():
    global current_generation_id, REFERENCE_F0, PERSONA_TEMPLATE, SUMMARY_PROMPT

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    cleanup_old_audio()
    interruption_requested.clear(); tts_actively_playing.clear(); program_is_shutting_down.clear()
    chat_history=[{"role":"system","content":""}]; current_memories = ""

    if CACHE_DIR.is_dir():
        interrupt_audio_files = list(CACHE_DIR.glob("*_interrupt_cache.wav"))
        greeting_files = list(CACHE_DIR.glob("*_greeting_cache.wav"))
    else:
        interrupt_audio_files, greeting_files = [], []

    if greeting_files:
        chosen_greeting_path = random.choice(greeting_files)
        logging.info(f"👋 Greeting: {chosen_greeting_path.name}")
        greeting_thread = threading.Thread(target=play_startup_greeting_thread, args=(chosen_greeting_path,), daemon=True)
        greeting_thread.start()

    def memory_manager_worker():
        nonlocal current_memories
        while not program_is_shutting_down.is_set():
            try:
                prev_mems, transcript_to_add = memory_q.get(timeout=0.2)
                if prev_mems is None: break
                current_memories = summarise(prev_mems, transcript_to_add)
                memory_q.task_done()
            except queue.Empty: continue

    thread_targets = [
        (audio_player_thread, "AudioPlayerThread"),
        (prefetch_worker, "PrefetchThread"), (asr_listener, "ASRListenerThread"),
        (kb_listener, "KBListenerThread"), (memory_manager_worker, "MemoryManagerThread")
    ]
    threads = [threading.Thread(target=t, name=n, daemon=True) for t, n in thread_targets]
    for thread in threads: thread.start()

    print("🎙️ System ready. Talk or type anytime.")

    try:
        while not program_is_shutting_down.is_set():
            try:
                current_user_input, _ = user_q.get(timeout=0.2)
                user_q.task_done()
            except queue.Empty:
                if program_is_shutting_down.is_set(): break
                continue

            if interruption_requested.is_set():
                sd.stop()
                while not audio_q.empty():
                    try: audio_q.get_nowait(); audio_q.task_done()
                    except queue.Empty: break
                while not prefetch_q.empty():
                    try: prefetch_q.get_nowait(); prefetch_q.task_done()
                    except queue.Empty: break

                if interrupt_audio_files:
                    try:
                        data_int, sr_int = sf.read(random.choice(interrupt_audio_files), dtype='float32')
                        sd.play(data_int, sr_int); sd.wait()
                    except Exception as e: print(f"‼️ Could not play interrupt clip: {e}", file=sys.stderr)
                interruption_requested.clear()

                if any(keyword in current_user_input.lower().strip(".?!, ") for keyword in INTERRUPT_KEYWORDS):
                    current_generation_id = None; continue

            current_generation_id = uuid.uuid4()

            for item in filler_builder.build_queue_items():
                if isinstance(item, Path):
                    audio_q.put((current_generation_id, item, "[FILLER]"))
                elif isinstance(item, int):
                    audio_q.put((current_generation_id, item, "[PAUSE]"))

            if current_user_input.lower() in {"quit","exit"}: program_is_shutting_down.set(); break
            if current_user_input.lower() == "mem": print(f"\n--- MEMORY ---\n{current_memories or '[none]'}\n---"); continue
            if current_user_input.lower() == "dump": [print(f"[{m['role']}] {m['content']}") for m in chat_history]; print("---"); continue

            system_prompt = PERSONA_TEMPLATE + (f"\n\n--- CONVERSATION MEMORIES ---\n{current_memories}" if current_memories else "")
            messages_for_llm = list(chat_history)
            messages_for_llm[0] = {"role": "system", "content": system_prompt}
            messages_for_llm.append({"role": "user", "content": current_user_input})

            llm_full_response = ""
            try:
                chunker_instance = StreamingChunker(min_words=MIN_WORDS_FOR_CHUNK)
                token_stream = (token for token in hermes_chat(messages_for_llm))

                for sanitized_chunk in chunker_instance.chunker(token_stream):
                    if interruption_requested.is_set(): break
                    prefetch_q.put((current_generation_id, [sanitized_chunk]))
                    llm_full_response += sanitized_chunk + " "

                if interruption_requested.is_set():
                    if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                    continue

            except Exception as e:
                print(f"\n‼️ Error during LLM streaming/chunking: {e}", file=sys.stderr)
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                continue

            assistant_response_for_chat_history = strip_action_and_emoji(llm_full_response)
            if assistant_response_for_chat_history.strip():
                print(f"🤖 Assistant: {assistant_response_for_chat_history}")

            if not assistant_response_for_chat_history.strip():
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                continue

            chat_history.append({"role": "user", "content": current_user_input})
            chat_history.append({"role": "assistant", "content": assistant_response_for_chat_history})

            total_tokens_in_history = sum(rough_tokens(m['content']) for m in chat_history)
            if total_tokens_in_history > TOKEN_SOFT_LIMIT:
                messages_to_retain = KEEP_RECENT * 2
                if len(chat_history) > messages_to_retain + 1:
                    messages_to_summarize = chat_history[1:-(messages_to_retain)]
                    if messages_to_summarize:
                        transcript_to_summarize = "\n".join(f"{m['role']}: {m['content']}" for m in messages_to_summarize)
                        memory_q.put((current_memories, transcript_to_summarize))
                        chat_history = [chat_history[0]] + chat_history[-(messages_to_retain):]

            while sum(rough_tokens(m['content']) for m in chat_history) > TOKEN_HARD_LIMIT:
                if len(chat_history) > 2: del chat_history[1]
                else: break

    except KeyboardInterrupt: print("\n🚨 Shutting down...");
    except Exception as e: print(f"‼️ UNHANDLED EXCEPTION IN MAIN: {e}"); traceback.print_exc()
    finally:
        program_is_shutting_down.set()
        audio_q.put((None, None, None)); prefetch_q.put((None, None)); memory_q.put((None,None))
        for t in threads: t.join(timeout=2.0)
        cleanup_old_audio()
        print("👋 Goodbye!")

if __name__ == "__main__":
    try:
        with open(PERSONA_PROMPT_TEMPLATE, "r", encoding="utf-8") as f: PERSONA_TEMPLATE = f.read()
        with open(SUMMARY_BOT_TEMPLATE, "r", encoding="utf-8") as f: SUMMARY_PROMPT = f.read()
        main()
    except FileNotFoundError as e:
        print(f"‼️ FATAL: A required persona file is missing: {e}. Please check your paths.")
        sys.exit(1)
    except Exception as e:
        print(f"‼️ An unexpected startup error occurred: {e}"); traceback.print_exc()