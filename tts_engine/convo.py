#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, pathlib, queue, re, sys, threading, traceback, uuid, random
from datetime import datetime
from difflib import SequenceMatcher
import time

import numpy as np, requests, sounddevice as sd, soundfile as sf
import torch
import whisper

os.environ["NEMO_DISABLE_TQDM"] = "1"

# --- VAD & ASR Configuration (from Message #12 "correct chunking" script) ---
VAD_INTERRUPT_TIMEOUT_S           = 0.2
VAD_DEFAULT_SILENCE_TIMEOUT_S     = 1.2
VAD_LONG_PROMPT_TIMEOUT_S         = 2.0
VAD_LONG_PROMPT_TRIGGER_S         = 3.0
INTERRUPTION_DURATION_THRESHOLD_S = 4

MIN_WORDS_ASR                     = 1
VAD_MIN_SPEECH_S                  = 0.25
VAD_SPEECH_CONFIDENCE_THRESHOLD   = 0.3
ECHO_SIMILARITY_THRESHOLD         = 0.7

WHISPER_BEAM_SIZE                 = 5
WHISPER_LOGPROB_THRESHOLD         = -1.0
WHISPER_NO_SPEECH_THRESHOLD       = 0.6

# --- endpoints / constants (unchanged) ---
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
ORPHEUS_API_URL    = "http://127.0.0.1:5005/v1/audio/speech"
VOICE              = "tara"
MIN_FIRST_WORDS    = 8 # Key for the main loop streaming logic
CHUNK_WORD_MAX     = 10
TOKEN_SOFT_LIMIT   = 2500
TOKEN_HARD_LIMIT   = 7700
KEEP_RECENT        = 2

# --- USER PATHS ---
OUTPUT_DIR = pathlib.Path("outputs")
CACHE_DIR  = pathlib.Path("../cached_clips")
OUTPUT_DIR.mkdir(exist_ok=True)

CLEANUP_FOLDERS = [
    OUTPUT_DIR,
    pathlib.Path("../outputs"),
]

SUMMARY_BOT_TEMPLATE    = pathlib.Path("../personas/chat_summary_bot.txt")
PERSONA_PROMPT_TEMPLATE = pathlib.Path("../personas/tts_default.txt")

# --- Constants for prefetch_worker filtering & TTS timeouts ---
TTS_REQUEST_TIMEOUT = (10, 30)
WORD_COUNT_THRESHOLD_FOR_TAGGED_SENTENCE_DISCARD = 2
STANDALONE_TAG_RE = re.compile(r"^\s*<[^>]+>\s*$")
ANY_BRACKETED_TAG_RE = re.compile(r"<[^>]+>")

try:
    with open(PERSONA_PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
        PERSONA_TEMPLATE = f.read()
    print(f"✅ Loaded default persona from: {PERSONA_PROMPT_TEMPLATE}")
except FileNotFoundError:
    print(f"‼️ FATAL: Persona file not found at '{PERSONA_PROMPT_TEMPLATE}'. Please ensure it exists.")
    sys.exit(1)

try:
    with open(SUMMARY_BOT_TEMPLATE, "r", encoding="utf-8") as f:
        SUMMARY_PROMPT = f.read()
    print(f"✅ Loaded summarizer persona from: {SUMMARY_BOT_TEMPLATE}")
except FileNotFoundError:
    print(f"⚠️ WARNING: Summarizer persona file not found at '{SUMMARY_BOT_TEMPLATE}'. Using a default.")
    SUMMARY_PROMPT = "You are a summarization bot. Your task is to update a running summary of a conversation."

audio_q, prefetch_q, user_q, memory_q = (queue.Queue() for _ in range(4))

interruption_requested = threading.Event()
tts_actively_playing = threading.Event()
program_is_shutting_down = threading.Event()
INTERRUPT_KEYWORDS = {"stop", "shut up", "hold on", "wait", "enough", "nevermind", "cancel", "that's enough"}

last_tts_text = ""
current_generation_id = None

# --- Text Cleaning Functions ---
ACTION_RE = re.compile(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]') # Original ACTION_RE
LLM_SPECIAL_TOKENS_RE = re.compile(r'<\|.*?\|>')

strip_stage = lambda txt: ACTION_RE.sub('', txt).strip() # Original strip_stage lambda

def strip_action_and_emoji(text_input: str) -> str: # For chat history (more aggressive)
    if not isinstance(text_input, str): return ""
    text = text_input
    text = ANY_BRACKETED_TAG_RE.sub('', text)
    text = ACTION_RE.sub('', text)
    text = LLM_SPECIAL_TOKENS_RE.sub('',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_for_tts_payload(text_input: str) -> str: # For sending to Orpheus TTS (less aggressive)
    if not isinstance(text_input, str): return ""
    text = text_input
    text = LLM_SPECIAL_TOKENS_RE.sub('', text) # Remove only LLM control tokens
    text = ACTION_RE.sub('', text) # Remove *...*, (...), emojis
    # General <expressive_tag> are KEPT at this stage.
    text = text.replace("<|", "").replace("|>", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Worker Threads (ASR, KB, AudioPlayer, TTS Requester, Prefetch, Summarizer) ---
# These are taken from a combination of your "correct chunking script" (msg #12)
# and the script from msg #10 where inter-segment delays were good and tag filtering was added.

def asr_listener(): # Exactly as in Message #12
    print("🎤 Loading Whisper ASR & Silero VAD…")
    try:
        asr_model = whisper.load_model("small.en")
        print("🎤 Whisper 'small.en' model loaded.")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        print("🎤 Silero VAD model loaded.")
    except Exception as e:
        print(f"‼️ FATAL: Could not load ASR/VAD models: {e}"); traceback.print_exc()
        program_is_shutting_down.set(); return

    sr = 16000
    vad_chunk_size = 512
    min_speech_frames = int(VAD_MIN_SPEECH_S * 1000 / (vad_chunk_size / sr * 1000))
    is_speaking = False; audio_buffer = []; silence_counter = 0

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=vad_chunk_size, callback=None) as stream:
            print("🎤 Continuous ASR listening (Literal Transcription Mode)…")
            while not program_is_shutting_down.is_set():
                try:
                    frame_float32, overflowed = stream.read(vad_chunk_size)
                    if overflowed: print("‼️ Mic overflow!", file=sys.stderr); continue
                    audio_tensor = torch.from_numpy(frame_float32.flatten())
                    speech_confidence = vad_model(audio_tensor, sr).item()
                    is_speech_in_frame = speech_confidence > VAD_SPEECH_CONFIDENCE_THRESHOLD
                    if is_speaking:
                        audio_buffer.append(frame_float32)
                        if is_speech_in_frame:
                            silence_counter = 0
                        else:
                            silence_counter += 1
                            active_timeout_s = VAD_DEFAULT_SILENCE_TIMEOUT_S
                            if tts_actively_playing.is_set():
                                active_timeout_s = VAD_INTERRUPT_TIMEOUT_S
                            else:
                                current_audio_data = np.concatenate(audio_buffer).squeeze()
                                if len(current_audio_data) > 0:
                                    speech_timestamps = get_speech_timestamps(torch.from_numpy(current_audio_data), vad_model, sampling_rate=sr)
                                    if speech_timestamps:
                                        precise_speech_duration_s = (speech_timestamps[-1]['end'] - speech_timestamps[0]['start']) / sr
                                        if precise_speech_duration_s > VAD_LONG_PROMPT_TRIGGER_S:
                                            active_timeout_s = VAD_LONG_PROMPT_TIMEOUT_S
                            silence_frames_needed = int(active_timeout_s / (vad_chunk_size / sr))
                            if silence_counter >= silence_frames_needed:
                                is_speaking = False
                                end_of_speech_time = time.monotonic()
                                if len(audio_buffer) >= min_speech_frames:
                                    full_audio = np.concatenate(audio_buffer).squeeze()
                                    final_speech_timestamps = get_speech_timestamps(torch.from_numpy(full_audio), vad_model, sampling_rate=sr)
                                    speech_duration_s = 0.0
                                    if final_speech_timestamps:
                                        speech_duration_s = (final_speech_timestamps[-1]['end'] - final_speech_timestamps[0]['start']) / sr
                                    result = asr_model.transcribe(full_audio, fp16=torch.cuda.is_available(), beam_size=WHISPER_BEAM_SIZE, logprob_threshold=WHISPER_LOGPROB_THRESHOLD, no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD, condition_on_previous_text=False)
                                    txt = result['text'].strip()
                                    processing_lag_s = time.monotonic() - end_of_speech_time
                                    if txt:
                                        is_echo = False
                                        if tts_actively_playing.is_set() and last_tts_text:
                                            similarity = SequenceMatcher(None, txt.lower(), last_tts_text.lower()).ratio()
                                            if similarity > ECHO_SIMILARITY_THRESHOLD:
                                                print(f"🎤 Echo detected (Similarity: {similarity:.2f}), discarding.", file=sys.stderr); is_echo = True
                                        if not is_echo and len(txt.split()) >= MIN_WORDS_ASR:
                                            print(f"\n🗣️ User (ASR): '{txt}' (Duration: {speech_duration_s:.2f}s, VAD Lag: {processing_lag_s:.2f}s)")
                                            if tts_actively_playing.is_set() and not interruption_requested.is_set():
                                                print("🎤 ASR: User spoke while TTS active -> Setting INTERRUPT_REQUESTED"); interruption_requested.set()
                                            user_q.put((txt, speech_duration_s))
                                audio_buffer = []
                    elif is_speech_in_frame:
                        is_speaking = True
                        silence_counter = 0
                        audio_buffer = [frame_float32]
                        if not tts_actively_playing.is_set(): print("\n🎤 Speech detected…", end='', flush=True)
                except sd.PortAudioError as pae:
                    if "Input overflowed" in str(pae): print("‼️ Mic overflow (PortAudioError)!", file=sys.stderr); continue
                    print(f"‼️ ASR PortAudioError: {pae}", file=sys.stderr); sd.sleep(1)
                except Exception as e:
                    if program_is_shutting_down.is_set(): break
                    print(f"‼️ ASR loop error: {e}", file=sys.stderr); traceback.print_exc(); is_speaking = False; audio_buffer = []; silence_counter = 0; sd.sleep(1)
    except Exception as e: print(f"‼️ ASR: Could not open InputStream: {e}"); traceback.print_exc()
    print("🎤 ASR listener thread finished.")

def kb_listener(): # Exactly as in Message #12
    print("⌨️ Keyboard listener started.")
    while not program_is_shutting_down.is_set():
        try:
            line = input("\n⌨️  ").strip()
            if program_is_shutting_down.is_set(): break
            if line:
                if tts_actively_playing.is_set() and line.lower() in INTERRUPT_KEYWORDS:
                    print(f"⌨️ Interrupt command: '{line}' -> Setting INTERRUPT_REQUESTED")
                    if not interruption_requested.is_set(): interruption_requested.set()
                user_q.put((line, 0.0))
        except EOFError: print("⌨️ EOF received, exiting keyboard listener."); break
        except KeyboardInterrupt: print("\n⌨️ Keyboard interrupt in listener. Exiting."); break
        except Exception as e:
            if program_is_shutting_down.is_set(): break
            print(f"‼️ KB Listener error: {e}"); traceback.print_exc(); break
    print("⌨️ Keyboard listener thread finished.")

def audio_player(): # From Message #10 (good inter-segment delay) with slight merge from #12
    global last_tts_text
    print("🎵 Audio player thread started.")
    player_active_gen_id = None
    while not program_is_shutting_down.is_set():
        try:
            gen_id, p_path_obj, text_that_was_spoken = audio_q.get(timeout=0.1) # Timeout from #12
        except queue.Empty:
            if not tts_actively_playing.is_set(): player_active_gen_id = None
            continue
        if p_path_obj is None: print("🎵 AudioPlayer: Shutdown sentinel received."); break # My addition for clarity
        if interruption_requested.is_set():
            # Logic from #10 to handle file unlinking for non-cached
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                 p_path_obj.unlink(missing_ok=True)
            player_active_gen_id = None; audio_q.task_done(); continue
        if player_active_gen_id is None:
            if gen_id != current_generation_id:
                print(f"🎵 AudioPlayer: Discarding stale audio chunk for an old generation. File: {p_path_obj.name if p_path_obj else 'N/A'}", file=sys.stderr) # Log from #12
                if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                    p_path_obj.unlink(missing_ok=True)
                audio_q.task_done(); continue
            player_active_gen_id = gen_id
        elif gen_id != player_active_gen_id:
            print(f"🎵 AudioPlayer: Gen ID mismatch. Discarding. File: {p_path_obj.name if p_path_obj else 'N/A'}", file=sys.stderr) # Log from #12
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                p_path_obj.unlink(missing_ok=True)
            audio_q.task_done(); continue
        is_filler_audio = (text_that_was_spoken == "[FILLER_AUDIO]")
        try:
            if not p_path_obj or not p_path_obj.exists():
                print(f"🎵 AudioPlayer: Audio file path is invalid or file does not exist: {p_path_obj}", file=sys.stderr)
                audio_q.task_done(); continue
            data, sr_audio = sf.read(str(p_path_obj), dtype="float32")
            last_tts_text_candidate = text_that_was_spoken # Store before play
            tts_actively_playing.set()
            if is_filler_audio: print(f"🎵 AudioPlayer: Playing FILLER audio: {p_path_obj.name}")
            else: print(f"🎵 AudioPlayer: Playing TTS audio: '{text_that_was_spoken}' (File: {p_path_obj.name})")
            sd.play(data, sr_audio); sd.wait()
            if not is_filler_audio: last_tts_text = last_tts_text_candidate
        except Exception as e: print(f"‼️ Audio playback error for '{p_path_obj}': {e}", file=sys.stderr); traceback.print_exc()
        finally:
            tts_actively_playing.clear()
            if not is_filler_audio and last_tts_text == text_that_was_spoken: # Clear only if it was the one played
                 last_tts_text = "" # Original from #12 cleared unconditionally, this is safer
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                try: p_path_obj.unlink(missing_ok=True)
                except Exception: pass
            audio_q.task_done()
    print("🎵 Audio player thread finished.")

def tts_requester_thread(text_chunk, generation_id): # text_chunk is filtered by prefetch_worker
    try:
        if program_is_shutting_down.is_set() or generation_id != current_generation_id: return
        audio_file = tts_request(text_chunk, generation_id)
        if audio_file:
            # Use the original text_chunk (that passed prefetch_worker's filter) for description
            # This ensures audio_player logs what was intended for speech, tags and all.
            audio_q.put((generation_id, audio_file, text_chunk))
    except Exception as e: print(f"‼️ TTS Requester Thread error for chunk '{text_chunk[:30]}...': {e}"); traceback.print_exc()

def prefetch_worker(): # Incorporates filtering logic
    print("⏳ Prefetch worker thread started.")
    while not program_is_shutting_down.is_set():
        try:
            generation_id, chunks_to_prefetch_from_main = prefetch_q.get(timeout=0.2) # Original timeout from #12
        except queue.Empty: continue
        if chunks_to_prefetch_from_main is None: print("⏳ Prefetch worker: Shutdown sentinel received."); break
        if generation_id != current_generation_id: prefetch_q.task_done(); continue

        fetch_threads = []
        for text_chunk_candidate in chunks_to_prefetch_from_main:
            if program_is_shutting_down.is_set() or interruption_requested.is_set() or generation_id != current_generation_id:
                break

            if STANDALONE_TAG_RE.fullmatch(text_chunk_candidate.strip()):
                print(f"PREFETCH_FILTER: DISCARDING standalone tag: '{text_chunk_candidate[:50]}' GenID: {generation_id}", file=sys.stderr)
                continue

            text_for_word_count = ANY_BRACKETED_TAG_RE.sub('', text_chunk_candidate).strip()
            word_count = len(text_for_word_count.split()) if text_for_word_count else 0
            contains_any_bracketed_tag = bool(ANY_BRACKETED_TAG_RE.search(text_chunk_candidate))

            if contains_any_bracketed_tag and word_count <= WORD_COUNT_THRESHOLD_FOR_TAGGED_SENTENCE_DISCARD:
                print(f"PREFETCH_FILTER: DISCARDING short tagged chunk: '{text_chunk_candidate[:50]}' (Word count of text: {word_count}) GenID: {generation_id}", file=sys.stderr)
                continue

            thread = threading.Thread(target=tts_requester_thread, args=(text_chunk_candidate, generation_id))
            thread.daemon = True; fetch_threads.append(thread); thread.start()

        if fetch_threads: # Only join if threads were started
            for i, thread in enumerate(fetch_threads):
                thread.join(timeout=TTS_REQUEST_TIMEOUT[1] + 10)
                if thread.is_alive():
                    print(f"⚠️ PREFETCH_WORKER_LOG: TTS requester thread {i} for GenID {generation_id} DID NOT JOIN IN TIME!", file=sys.stderr)

        prefetch_q.task_done()
    print("⏳ Prefetch worker thread finished.")

# Text processing utilities from Message #12 (your "correct chunking" script)
# ACTION_RE = re.compile(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]') # Already defined above
# strip_stage = lambda txt: ACTION_RE.sub('', txt).strip() # Already defined above
rough_tokens = lambda txt: max(1, len(txt)//4)
def split_sentences(txt):
    pat=re.compile(r'(<[^>]+>|[^.!?]+[.!?]?)')
    return [m.strip() for m in pat.findall(txt) if m.strip()]
def batch_sentences(sents,mw):
    out,buf,words=[],[],0
    for s in sents:
        if re.fullmatch(r"<.*?>",s):
            if buf: out.append(" ".join(buf)); buf=[]; words=0
            out.append(s); continue
        w=len(s.split())
        if words+w>mw and buf: out.append(" ".join(buf)); buf,[words]=[s],[w]
        else: buf.append(s); words+=w
    if buf: out.append(" ".join(buf))
    return out

def summarise(prev_summary, new_transcript): # From Message #12
    if program_is_shutting_down.is_set(): return prev_summary or "[summary skipped due to shutdown]"
    try:
        content = f"PREVIOUS SUMMARY:\n{prev_summary or 'None.'}\n\nNEW TRANSCRIPT TO ADD:\n{new_transcript}"
        r = requests.post(LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}], "temperature": 0.6, "max_tokens": 1024}, timeout=200)
        r.raise_for_status(); summary_text = r.json()['choices'][0]['message']['content'].strip()
        if summary_text.lower().startswith("updated summary:"): summary_text = summary_text.split(":", 1)[1].strip()
        elif summary_text.lower().startswith("here is an updated summary:"): summary_text = summary_text.split(":", 1)[1].strip()
        return summary_text
    except Exception as e: print(f"‼️ Summary failed: {e}", file=sys.stderr); traceback.print_exc(); return prev_summary or "[summary generation failed]"

def hermes_chat(msgs): # Using iter_content version for robustness (from message #10 pattern)
    llm_response_stream = None
    try:
        llm_response_stream = requests.post(LM_STUDIO_CHAT_URL, json={"model":"hermes","messages":msgs,"temperature":0.9,"stream":True}, stream=True,timeout=(10,60)) # Timeout from msg #12
        llm_response_stream.raise_for_status()
        buffer = ""
        for chunk in llm_response_stream.iter_content(chunk_size=None, decode_unicode=True):
            if program_is_shutting_down.is_set() or interruption_requested.is_set():
                print("\n🚫 LLM stream: INTERRUPT_REQUESTED. Stopping generation.", file=sys.stderr); break
            buffer += chunk
            while 'data: ' in buffer and '\n' in buffer:
                try:
                    event_start_index = buffer.find('data: ')
                    if event_start_index == -1: break
                    line_end_index = buffer.find('\n', event_start_index)
                    if line_end_index == -1: break
                    line_with_data_prefix = buffer[event_start_index : line_end_index]
                    if line_with_data_prefix.startswith('data: '):
                        data_segment = line_with_data_prefix[6:].strip()
                        if data_segment == '[DONE]':
                            buffer = buffer[line_end_index+1:]; return
                        if data_segment:
                            delta = json.loads(data_segment)['choices'][0]['delta']
                            content_token = delta.get('content')
                            if content_token: yield content_token
                    buffer = buffer[line_end_index+1:]
                except json.JSONDecodeError: # Bare except from #12 was `except: continue`
                    next_data_index = buffer.find('data: ', event_start_index + 1 if 'event_start_index' in locals() else 0)
                    if next_data_index != -1: buffer = buffer[next_data_index:]
                    else:
                        if len(buffer) > 1024: return
                        break
                except Exception: # Bare except from #12 was `except: continue`
                    next_data_index = buffer.find('data: ', event_start_index + 1 if 'event_start_index' in locals() else 0)
                    if next_data_index != -1: buffer = buffer[next_data_index:]
                    else:
                        if len(buffer) > 1024: return
                        break
    except requests.exceptions.RequestException as e: print(f"‼️ Hermes network error: {e}", file=sys.stderr); traceback.print_exc()
    except Exception as e_hermes: print(f"‼️ Unhandled Hermes error: {e_hermes}", file=sys.stderr); traceback.print_exc()
    finally:
        if llm_response_stream:
            try: llm_response_stream.close()
            except Exception : pass # Bare except from #12

def tts_request(txt, generation_id): # txt is pre-filtered by prefetch_worker
    if program_is_shutting_down.is_set() or (generation_id is not None and generation_id != current_generation_id):
        print(f"   -> 📡 TTS Request CANCELLED for stale generation.")
        return None

    payload_text = clean_text_for_tts_payload(txt)
    if not payload_text: return None

    try:
        wav_data = requests.post(ORPHEUS_API_URL,json={"input": payload_text,"model":"orpheus","voice":VOICE,"response_format":"wav","speed":1.0},timeout=TTS_REQUEST_TIMEOUT).content
        if len(wav_data) < 1000 and "RIFF" not in str(wav_data[:4]).upper(): # Original check was "RIFF" in str(wav_data[:4])
             print(f"🎤 TTS: Short WAV ({len(wav_data)}b) for '{payload_text[:30]}...'. Skip."); return None
        file_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+".wav"; output_path = OUTPUT_DIR/file_name
        output_path.write_bytes(wav_data); return output_path
    except requests.exceptions.Timeout as e_timeout: # Specific timeout handling
        print(f"‼️ TTS_REQUEST_LOG: TIMEOUT for '{payload_text[:30]}' GenID: {generation_id} Error: {e_timeout}", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"‼️ TTS API failed for '{payload_text[:30]}...': {e}"); traceback.print_exc(); return None
    except Exception as e:
        print(f"‼️ TTS error for '{payload_text[:30]}...': {e}"); traceback.print_exc(); return None

def cleanup_old_audio():
    print("🧹 Cleaning up old audio files...")
    count = 0
    for folder_path in CLEANUP_FOLDERS:
        if not folder_path.is_dir(): continue
        for audio_file in folder_path.glob("*.wav"):
            try:
                if CACHE_DIR.resolve() in audio_file.resolve().parents:
                    continue
                audio_file.unlink(); count += 1
            except Exception as e: print(f"🧹 Could not delete {audio_file}: {e}", file=sys.stderr)
    if count > 0: print(f"🧹 Removed {count} old .wav file(s).") # Original had no else

def main():
    global current_generation_id
    cleanup_old_audio()
    interruption_requested.clear(); tts_actively_playing.clear(); program_is_shutting_down.clear()
    chat_history=[{"role":"system","content":PERSONA_TEMPLATE}]; current_memories = ""

    interrupt_audio_files, pending_audio_files, long_query_audio_files = [], [], []
    if CACHE_DIR.is_dir():
        interrupt_audio_files = list(CACHE_DIR.glob("*_interrupt_cache.wav"))
        pending_audio_files = list(CACHE_DIR.glob("*_pending_cache.wav"))
        long_query_audio_files = list(CACHE_DIR.glob("*_long_query_cache.wav"))
        if interrupt_audio_files: print(f"🎤 Found {len(interrupt_audio_files)} interrupt acknowledgement clips.")
        if pending_audio_files: print(f"🎤 Found {len(pending_audio_files)} pending clips.")
        if long_query_audio_files: print(f"🎤 Found {len(long_query_audio_files)} long query clips.")
    else: print(f"🎤 WARNING: Cache directory '{CACHE_DIR}' not found, filler audio will be disabled.")

    def memory_manager_worker():
        nonlocal current_memories; print("🧠 Memory manager thread started.")
        while not program_is_shutting_down.is_set():
            try: prev_mems, transcript_to_add = memory_q.get(timeout=0.2)
            except queue.Empty: continue
            if prev_mems is None and transcript_to_add is None: print("🧠 Memory manager: Shutdown sentinel.");break # My addition for clarity
            current_memories = summarise(prev_mems, transcript_to_add)
            print(f"🔹 Memories updated to: '{current_memories[:100].replace(os.linesep, ' ')}...'")
            memory_q.task_done()
        print("🧠 Memory manager thread finished.")

    threads = [(threading.Thread(target=f,daemon=d, name=n)) for f,d,n in [
            (audio_player,True,"AudioPlayerThread"), (prefetch_worker,True,"PrefetchThread"),
            (asr_listener,True,"ASRListenerThread"), (kb_listener,True,"KBListenerThread"),
            (memory_manager_worker,True,"MemoryManagerThread")]]
    for t in threads: t.start()
    print("🎙️  Talk or type anytime — 'quit' to exit, 'mem'/'dump' commands, or interrupt (e.g., 'stop').")

    try:
        while not program_is_shutting_down.is_set():
            try:
                item = user_q.get(timeout=0.2)
                current_user_input, speech_duration = item if isinstance(item, tuple) and len(item) == 2 else (str(item), 0.0)
                user_q.task_done()
            except queue.Empty:
                if program_is_shutting_down.is_set(): break
                continue
            was_an_interruption = interruption_requested.is_set()

            # Non-blocking filler and interruption handling:
            if was_an_interruption:
                print("MAIN: Interruption detected. Cleaning up...")
                # current_generation_id = None # Set this *after* filler is queued with new ID
                sd.stop(); tts_actively_playing.clear()
                for q in [audio_q, prefetch_q]:
                    while not q.empty():
                        try:
                            item_q = q.get_nowait()
                            if isinstance(item_q, tuple) and len(item_q) > 1 and isinstance(item_q[1], pathlib.Path):
                                if not item_q[1].is_relative_to(CACHE_DIR):
                                    item_q[1].unlink(missing_ok=True)
                            q.task_done()
                        except Exception: pass

                if interrupt_audio_files: # Short blocking acknowledgement
                    try:
                        clip_to_play = random.choice(interrupt_audio_files)
                        data_int, sr_int = sf.read(str(clip_to_play), dtype='float32')
                        sd.play(data_int, sr_int); sd.wait()
                    except Exception as e: print(f"‼️ Could not play interrupt clip: {e}", file=sys.stderr)

                interruption_requested.clear() # Clear flag *after* ack, *before* processing input
                print("MAIN: Interruption processed.")

                is_stop_command = any(keyword in current_user_input.lower().strip(".?!, ") for keyword in INTERRUPT_KEYWORDS)
                if is_stop_command:
                    print("INFO: Interrupt was a stop command. Waiting for next user input.")
                    current_generation_id = None # No active generation now
                    continue
                else: # Interrupt was new content
                    print("INFO: Interrupt was content. Proceeding with it as a new prompt.")
                    # A new current_generation_id will be set for this new content.
                    # Filler audio should be associated with that *new* upcoming generation.
                    # So, we set the new gen_id *before* queuing the filler.
                    current_generation_id = uuid.uuid4() # NEW Gen ID for the interrupting content
                    filler_clip_to_play = None
                    if speech_duration > INTERRUPTION_DURATION_THRESHOLD_S and long_query_audio_files:
                        filler_clip_to_play = random.choice(long_query_audio_files); print(f"INFO: User interruption was long ({speech_duration:.2f}s). Queueing long_query filler.")
                    elif pending_audio_files:
                        filler_clip_to_play = random.choice(pending_audio_files); print(f"INFO: User interruption was short ({speech_duration:.2f}s). Queueing pending filler.")

                    if filler_clip_to_play:
                        print(f"INFO: Queueing interrupt filler clip '{filler_clip_to_play.name}' to audio_q for new GenID: {current_generation_id}", file=sys.stderr)
                        audio_q.put((current_generation_id, filler_clip_to_play, "[FILLER_AUDIO]"))
                    # Fall through to process current_user_input with the new current_generation_id

            if current_user_input.lower() in {"quit","exit"}: print("MAIN: Quit."); program_is_shutting_down.set(); break
            if current_user_input.lower() == "mem": print(f"--- MEMORY ---\n{current_memories or '[none]'}\n---"); continue
            if current_user_input.lower() == "dump": [print(f"[{m['role']}] {m['content']}") for m in chat_history]; print("---"); continue

            if not was_an_interruption: # Only generate a new ID if it wasn't an interruption that already set one
                current_generation_id = uuid.uuid4()

            messages_for_llm = list(chat_history)
            system_prompt_content = PERSONA_TEMPLATE
            if current_memories:
                messages_for_llm[0] = {"role": "system", "content": f"{PERSONA_TEMPLATE}\n\n--- CONVERSATION MEMORIES ---\n{current_memories}"}
            elif not (messages_for_llm and messages_for_llm[0]['role'] == 'system'):
                messages_for_llm.insert(0, {"role": "system", "content": system_prompt_content})
            messages_for_llm.append({"role": "user", "content": current_user_input})

            # Removed the blocking filler play from here if it was a regular turn.
            # Filler logic is now consolidated within the `if was_an_interruption` block and is non-blocking.

            print(f"🤖 Assistant thinking... (Gen: {current_generation_id})", end='', flush=True, file=sys.stderr)

            # --- TTS STREAMING LOOP - EXACTLY FROM SCRIPT IN MESSAGE #12 (CHUNK_SOURCE_TRUTH) ---
            llm_full_response = ""
            text_processed_for_tts = ""
            first_sentence_sent = False
            sentence_ender = re.compile(r"(?<!\b[A-Z][a-z]\.)(?<=[.?!])\s")

            try:
                # print(f"MAIN_TTS_STREAM_LOG: [{time.monotonic():.2f}] Starting hermes_chat loop GenID: {current_generation_id}", file=sys.stderr)
                for token_chunk in hermes_chat(messages_for_llm):
                    print(token_chunk, end='', flush=True, file=sys.stderr)
                    llm_full_response += token_chunk
                    unprocessed_text = llm_full_response[len(text_processed_for_tts):]
                    while True:
                        ready_to_process = (
                            (not first_sentence_sent and len(unprocessed_text.split()) >= MIN_FIRST_WORDS) or
                            first_sentence_sent
                        )
                        if not ready_to_process: break
                        sentence_match = sentence_ender.search(unprocessed_text)
                        if not sentence_match:
                            break

                        sentence_to_process = unprocessed_text[:sentence_match.end()]
                        text_processed_for_tts += sentence_to_process
                        batched = batch_sentences(split_sentences(sentence_to_process.strip()), CHUNK_WORD_MAX)
                        if batched:
                            # print(f"MAIN_TTS_STREAM_LOG: [{time.monotonic():.2f}] Putting on prefetch_q: '{sentence_to_process.strip()[:30]}' ({len(batched)} sub-chunks) GenID: {current_generation_id}", file=sys.stderr)
                            prefetch_q.put((current_generation_id, batched))
                        if not first_sentence_sent: first_sentence_sent = True
                        unprocessed_text = llm_full_response[len(text_processed_for_tts):]
                        if not unprocessed_text.strip(): break

                print(file=sys.stderr)
                # print(f"MAIN_TTS_STREAM_LOG: [{time.monotonic():.2f}] Finished hermes_chat loop. Full response len: {len(llm_full_response)} GenID: {current_generation_id}", file=sys.stderr)

                if interruption_requested.is_set():
                    if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                    continue
                remaining_text = llm_full_response[len(text_processed_for_tts):].strip()
                if remaining_text:
                    batched = batch_sentences(split_sentences(remaining_text), CHUNK_WORD_MAX)
                    if batched:
                        # print(f"MAIN_TTS_STREAM_LOG: [{time.monotonic():.2f}] Putting remaining on prefetch_q: '{remaining_text[:30]}' ({len(batched)} sub-chunks) GenID: {current_generation_id}", file=sys.stderr)
                        prefetch_q.put((current_generation_id, batched))
            # --- END OF TTS STREAMING LOGIC FROM SCRIPT IN MESSAGE #12 ---
            except Exception as e:
                print(f"\n‼️ Error during LLM response streaming or TTS batching: {e}", file=sys.stderr); traceback.print_exc()
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                continue

            assistant_response_for_chat_history = strip_action_and_emoji(llm_full_response)
            if not assistant_response_for_chat_history.strip():
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                continue

            if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
            chat_history.append({"role": "assistant", "content": assistant_response_for_chat_history})

            total_tokens_in_history = sum(rough_tokens(m['content']) for m in chat_history)
            if total_tokens_in_history > TOKEN_SOFT_LIMIT:
                messages_to_retain_at_end = KEEP_RECENT * 2
                if len(chat_history) > messages_to_retain_at_end + 1:
                    messages_to_summarize = chat_history[1 : -(messages_to_retain_at_end)]
                    if messages_to_summarize:
                        transcript_text_for_summary = "\n".join(f"{m['role']}: {m['content']}" for m in messages_to_summarize)
                        print(f"🧠 Pruning chat. Summarizing {len(messages_to_summarize)} messages.")
                        memory_q.put((current_memories, transcript_text_for_summary))
                        chat_history = [chat_history[0]] + chat_history[-(messages_to_retain_at_end):]
            while sum(rough_tokens(m['content']) for m in chat_history) > TOKEN_HARD_LIMIT:
                if len(chat_history) > 2 : print(f"‼️ HARD TOKEN PRUNING: Removing '{chat_history[1]['content'][:30]}...'"); del chat_history[1]
                else: break
    except KeyboardInterrupt: print("\n🚨 Main loop: KeyboardInterrupt."); program_is_shutting_down.set()
    except Exception as e: print(f"‼️ UNHANDLED EXCEPTION IN MAIN: {e}"); traceback.print_exc(); program_is_shutting_down.set()
    finally:
        print("🛑 Main loop finished. Finalizing shutdown...")
        if not program_is_shutting_down.is_set(): program_is_shutting_down.set()
        print("🛑 Sending sentinels to worker queues..."); audio_q.put((None, None, None)); prefetch_q.put((None, None)); memory_q.put((None,None))
        active_threads = [t for t in threads if t.is_alive()]
        for t in active_threads:
            print(f"🛑 Waiting for {t.name} to join (2s)..."); t.join(timeout=2.0)
            if t.is_alive(): print(f"⚠️ {t.name} did not join cleanly.")
            else: print(f"✅ {t.name} joined.")
        print("🧹 Performing final audio cleanup..."); cleanup_old_audio(); print("👋 Goodbye!")

if __name__=="__main__":
    main()