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

VAD_INTERRUPT_TIMEOUT_S           = 0.5
VAD_DEFAULT_SILENCE_TIMEOUT_S     = 1.2
VAD_LONG_PROMPT_TIMEOUT_S         = 2
VAD_LONG_PROMPT_TRIGGER_S         = 3
INTERRUPTION_DURATION_THRESHOLD_S = 4

MIN_WORDS_ASR                     = 1
VAD_MIN_SPEECH_S                  = 0.25
VAD_SPEECH_CONFIDENCE_THRESHOLD   = 0.1
ECHO_SIMILARITY_THRESHOLD         = 0.9

WHISPER_BEAM_SIZE                 = 5
WHISPER_LOGPROB_THRESHOLD         = -1.0
WHISPER_NO_SPEECH_THRESHOLD       = 0.6

LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
ORPHEUS_API_URL    = "http://127.0.0.1:5005/v1/audio/speech"
VOICE              = "tara"
MIN_FIRST_WORDS    = 15
CHUNK_WORD_MAX     = 25
MIN_SUBSEQUENT_WORDS = MIN_FIRST_WORDS
TOKEN_SOFT_LIMIT   = 2500
TOKEN_HARD_LIMIT   = 7700
KEEP_RECENT        = 2

LLM_CHUNK_LOOKAHEAD_S = 0.35

TTS_REQUEST_TIMEOUT = (10, 30)
WORD_COUNT_THRESHOLD_FOR_TAGGED_SENTENCE_DISCARD = 2
STANDALONE_TAG_RE = re.compile(r"^\s*<[^>]+>\s*$")
ANY_BRACKETED_TAG_RE = re.compile(r"<[^>]+>")
UNWANTED_INTERJECTIONS_RE = re.compile(
    r"""\b(hehe|haha|hihi|huhu|hoho)(?:\.{2,3})?(?!\w)\s*""",
    re.IGNORECASE | re.VERBOSE
)
ACTION_RE_FOR_COUNT = re.compile(r'\*[^\*]+\*')


OUTPUT_DIR = pathlib.Path("outputs")
CACHE_DIR  = pathlib.Path("../cached_clips")
OUTPUT_DIR.mkdir(exist_ok=True)
CLEANUP_FOLDERS = [ OUTPUT_DIR, pathlib.Path("../outputs"),]

SUMMARY_BOT_TEMPLATE    = pathlib.Path("../personas/chat_summary_bot.txt")
PERSONA_PROMPT_TEMPLATE = pathlib.Path("../personas/tts_default.txt")

try:
    with open(PERSONA_PROMPT_TEMPLATE, "r", encoding="utf-8") as f: PERSONA_TEMPLATE = f.read()
    print(f"[SYSTEM] ✅ Loaded default persona from: {PERSONA_PROMPT_TEMPLATE}")
except FileNotFoundError: print(f"[SYSTEM] ‼️ FATAL: Persona file not found at '{PERSONA_PROMPT_TEMPLATE}'."); sys.exit(1)
try:
    with open(SUMMARY_BOT_TEMPLATE, "r", encoding="utf-8") as f: SUMMARY_PROMPT = f.read()
    print(f"[SYSTEM] ✅ Loaded summarizer persona from: {SUMMARY_BOT_TEMPLATE}")
except FileNotFoundError: print(f"[SYSTEM] ⚠️ WARNING: Summarizer persona not found at '{SUMMARY_BOT_TEMPLATE}'."); SUMMARY_PROMPT = "You are a summarization bot..."

audio_q, prefetch_q, user_q, memory_q = (queue.Queue() for _ in range(4))
interruption_requested = threading.Event(); tts_actively_playing = threading.Event(); program_is_shutting_down = threading.Event()
INTERRUPT_KEYWORDS = {"stop", "shut up", "hold on", "wait", "enough", "nevermind", "cancel", "that's enough"}
last_tts_text = ""; current_generation_id = None

ACTION_RE_CLEAN = re.compile(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]') # Used for cleaning TTS payload and history
LLM_SPECIAL_TOKENS_RE = re.compile(r'<\|.*?\|>')

def strip_action_and_emoji(text_input: str) -> str:
    if not isinstance(text_input, str): return ""
    text = text_input; text = UNWANTED_INTERJECTIONS_RE.sub('', text); text = ANY_BRACKETED_TAG_RE.sub('', text)
    text = ACTION_RE_CLEAN.sub('', text); text = LLM_SPECIAL_TOKENS_RE.sub('',text)
    text = re.sub(r'\s+', ' ', text).strip(); return text

def clean_text_for_tts_payload(text_input: str) -> str:
    if not isinstance(text_input, str): return ""
    text = text_input; text = UNWANTED_INTERJECTIONS_RE.sub('', text); text = LLM_SPECIAL_TOKENS_RE.sub('', text)
    text = ACTION_RE_CLEAN.sub('', text); text = text.replace("<|", "").replace("|>", "")
    text = re.sub(r'\s+', ' ', text).strip(); return text

def asr_listener():
    print("[SYSTEM] 🎤 Loading Whisper ASR & Silero VAD…")
    try:
        asr_model = whisper.load_model("small.en"); print("[SYSTEM] 🎤 Whisper 'small.en' model loaded.")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (get_speech_timestamps, _, _, _, _) = utils; print("[SYSTEM] 🎤 Silero VAD model loaded.")
    except Exception as e: print(f"[ERROR] ‼️ FATAL: Could not load ASR/VAD models: {e}"); traceback.print_exc(); program_is_shutting_down.set(); return
    sr = 16000; vad_chunk_size = 512; vad_chunk_duration_s = vad_chunk_size / sr
    min_speech_frames = int(VAD_MIN_SPEECH_S / vad_chunk_duration_s)
    is_speaking = False; audio_buffer_list = []; silence_counter_frames = 0; is_long_prompt_criteria_met = False
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=vad_chunk_size, callback=None) as stream:
            print("[ASR] 🎤 Continuous ASR listening (Literal Transcription Mode)…")
            while not program_is_shutting_down.is_set():
                try:
                    frame_float32, overflowed = stream.read(vad_chunk_size)
                    if overflowed: print("[ASR_ERROR] ‼️ Mic overflow!", file=sys.stderr); continue
                    audio_tensor = torch.from_numpy(frame_float32.flatten())
                    speech_confidence = vad_model(audio_tensor, sr).item()
                    is_speech_in_current_frame = speech_confidence > VAD_SPEECH_CONFIDENCE_THRESHOLD
                    if is_speaking:
                        audio_buffer_list.append(frame_float32)
                        if is_speech_in_current_frame: silence_counter_frames = 0
                        else:
                            silence_counter_frames += 1; active_timeout_s = VAD_DEFAULT_SILENCE_TIMEOUT_S
                            if tts_actively_playing.is_set(): active_timeout_s = VAD_INTERRUPT_TIMEOUT_S
                            else:
                                if not is_long_prompt_criteria_met:
                                    current_audio_data_np = np.concatenate(audio_buffer_list).squeeze()
                                    if current_audio_data_np.ndim == 0: current_audio_data_np = np.array([current_audio_data_np])
                                    if current_audio_data_np.size > 0:
                                        current_audio_data = torch.from_numpy(current_audio_data_np)
                                        speech_ts = get_speech_timestamps(current_audio_data, vad_model, sampling_rate=sr, min_speech_duration_ms=int(VAD_MIN_SPEECH_S * 1000))
                                        if speech_ts:
                                            if (speech_ts[-1]['end'] - speech_ts[0]['start']) / sr > VAD_LONG_PROMPT_TRIGGER_S: is_long_prompt_criteria_met = True
                                if is_long_prompt_criteria_met: active_timeout_s = VAD_LONG_PROMPT_TRIGGER_S
                                else: active_timeout_s = VAD_DEFAULT_SILENCE_TIMEOUT_S
                            if silence_counter_frames * vad_chunk_duration_s >= active_timeout_s:
                                is_speaking = False; is_long_prompt_criteria_met = False; end_of_speech_time = time.monotonic()
                                if len(audio_buffer_list) >= min_speech_frames:
                                    full_audio_np = np.concatenate(audio_buffer_list).squeeze()
                                    if full_audio_np.ndim == 0: full_audio_np = np.array([full_audio_np])
                                    if full_audio_np.size > 0:
                                        final_speech_ts = get_speech_timestamps(torch.from_numpy(full_audio_np), vad_model, sampling_rate=sr, min_speech_duration_ms=int(VAD_MIN_SPEECH_S * 1000))
                                        s_dur = (final_speech_ts[-1]['end'] - final_speech_ts[0]['start']) / sr if final_speech_ts else 0.0
                                        txt = asr_model.transcribe(full_audio_np, fp16=torch.cuda.is_available(), beam_size=WHISPER_BEAM_SIZE, logprob_threshold=WHISPER_LOGPROB_THRESHOLD, no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD)['text'].strip()
                                        lag = time.monotonic() - end_of_speech_time
                                        if txt and (SequenceMatcher(None, txt.lower(), last_tts_text.lower()).ratio() < ECHO_SIMILARITY_THRESHOLD if tts_actively_playing.is_set() and last_tts_text else True) and len(txt.split()) >= MIN_WORDS_ASR:
                                            print(f"\n[USER_ASR] '{txt}' (Duration: {s_dur:.2f}s, VAD Lag: {lag:.2f}s)")
                                            if tts_actively_playing.is_set() and not interruption_requested.is_set(): print("[ASR_DEBUG] User spoke while TTS active -> Setting INTERRUPT_REQUESTED"); interruption_requested.set()
                                            user_q.put((txt, s_dur))
                                audio_buffer_list = []; silence_counter_frames = 0
                    elif is_speech_in_current_frame:
                        is_speaking = True; silence_counter_frames = 0; audio_buffer_list = [frame_float32]; is_long_prompt_criteria_met = False
                        if not tts_actively_playing.is_set(): print(f"\n[ASR] Speech detected…", end='', flush=True)
                except sd.PortAudioError as pae:
                    if "Input overflowed" in str(pae): print("[ASR_ERROR] ‼️ Mic overflow (PortAudioError)!", file=sys.stderr); continue
                    print(f"[ASR_ERROR] ‼️ PortAudioError: {pae}", file=sys.stderr); sd.sleep(1) # type: ignore
                except Exception as e:
                    if program_is_shutting_down.is_set(): break
                    print(f"[ASR_ERROR] ‼️ Loop error: {e}", file=sys.stderr); traceback.print_exc(); is_speaking=False; audio_buffer_list=[]; silence_counter_frames=0; is_long_prompt_criteria_met=False; sd.sleep(1) # type: ignore
    except Exception as e: print(f"[ASR_ERROR] ‼️ Could not open InputStream: {e}"); traceback.print_exc()
    print("[ASR] Listener thread finished.")

def kb_listener():
    print("[SYSTEM] ⌨️ Keyboard listener started.")
    while not program_is_shutting_down.is_set():
        try:
            line = input("\n[USER_KB_PROMPT] ⌨️  ").strip()
            if program_is_shutting_down.is_set(): break
            if line:
                if tts_actively_playing.is_set() and line.lower() in INTERRUPT_KEYWORDS and not interruption_requested.is_set():
                    print(f"[USER_KB] Interrupt command: '{line}' -> Setting INTERRUPT_REQUESTED"); interruption_requested.set()
                user_q.put((line, 0.0))
        except EOFError: print("[USER_KB] EOF received, exiting."); break
        except KeyboardInterrupt: print("\n[USER_KB] Keyboard interrupt. Exiting."); break
        except Exception as e:
            if program_is_shutting_down.is_set(): break
            print(f"[USER_KB_ERROR] ‼️ {e}"); traceback.print_exc(); break
    print("[SYSTEM] ⌨️ Keyboard listener thread finished.")

def audio_player():
    global last_tts_text
    print("[SYSTEM] 🎵 Audio player thread started.")
    while not program_is_shutting_down.is_set():
        try: gen_id, p_path_obj, text_that_was_spoken = audio_q.get(timeout=0.1)
        except queue.Empty: continue
        if p_path_obj is None and text_that_was_spoken is None : print("[AUDIO_PLAYER] Shutdown sentinel."); break
        if interruption_requested.is_set() and gen_id != current_generation_id:
            if tts_actively_playing.is_set(): sd.stop(ignore_errors=True); tts_actively_playing.clear()
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR): p_path_obj.unlink(missing_ok=True)
            audio_q.task_done(); continue
        is_filler = (text_that_was_spoken == "[FILLER_AUDIO]")
        if gen_id == current_generation_id or is_filler:
            try:
                if not p_path_obj or not p_path_obj.exists(): print(f"[AUDIO_PLAYER_ERROR] File DNE: {p_path_obj}", file=sys.stderr); audio_q.task_done(); continue
                if tts_actively_playing.is_set(): print(f"[AUDIO_PLAYER_WARN] TTS active. Stopping previous for '{p_path_obj.name}'.", file=sys.stderr); sd.stop(ignore_errors=True); tts_actively_playing.clear()
                data, sr_audio = sf.read(str(p_path_obj), dtype="float32"); tts_actively_playing.set()
                print(f"\n[TTS_PLAY{'_FILLER' if is_filler else ''}] Playing: '{text_that_was_spoken}' (File: {p_path_obj.name}, Gen: {gen_id_short(gen_id)})")
                sd.play(data, sr_audio); sd.wait()
                if not is_filler: last_tts_text = text_that_was_spoken
            except Exception as e: print(f"[AUDIO_PLAYER_ERROR] ‼️ Playback error for '{p_path_obj}': {e}", file=sys.stderr); traceback.print_exc(); tts_actively_playing.clear()
            finally:
                tts_actively_playing.clear()
                if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                    try: p_path_obj.unlink(missing_ok=True)
                    except Exception as del_e: print(f"[AUDIO_PLAYER_ERROR] Error deleting {p_path_obj.name}: {del_e}", file=sys.stderr)
                audio_q.task_done()
        else:
            print(f"[AUDIO_PLAYER_DEBUG] Discarding stale audio '{p_path_obj.name if p_path_obj else 'N/A'}' GenID {gen_id_short(gen_id)} (current: {gen_id_short(current_generation_id)}).", file=sys.stderr)
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR): p_path_obj.unlink(missing_ok=True)
            audio_q.task_done()
    print("[SYSTEM] 🎵 Audio player thread finished.")

def gen_id_short(gen_id): return str(gen_id)[-6:] if gen_id else "None"

def tts_request(txt, gen_id):
    if program_is_shutting_down.is_set() or (gen_id is not None and gen_id != current_generation_id): return None
    payload_text = clean_text_for_tts_payload(txt)
    if not payload_text: return None
    try:
        r = requests.post(ORPHEUS_API_URL, json={"input": payload_text, "model":"orpheus", "voice":VOICE, "response_format":"wav", "speed":1.0}, timeout=TTS_REQUEST_TIMEOUT)
        r.raise_for_status(); wav_data = r.content
        if len(wav_data) < 1000 and not (wav_data.startswith(b"RIFF") and len(wav_data) > 44): print(f"[TTS_API_WARN] Short/invalid WAV ({len(wav_data)}b) for '{payload_text[:30]}...'. Skip.", file=sys.stderr); return None
        f_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+".wav"; o_path = OUTPUT_DIR/f_name
        o_path.write_bytes(wav_data); return o_path
    except requests.exceptions.Timeout as et: print(f"[TTS_API_ERROR] TIMEOUT for '{payload_text[:30]}' GenID: {gen_id_short(gen_id)} Error: {et}", file=sys.stderr); return None
    except requests.exceptions.RequestException as er: print(f"[TTS_API_ERROR] FAILED for '{payload_text[:30]}' GenID: {gen_id_short(gen_id)} Error: {er}", file=sys.stderr); return None
    except Exception as e: print(f"[TTS_API_ERROR] UNKNOWN ERR for '{payload_text[:30]}' GenID: {gen_id_short(gen_id)} Error: {e}", file=sys.stderr); return None

def tts_requester_thread(text_chunk, gen_id):
    try:
        if program_is_shutting_down.is_set() or gen_id != current_generation_id: return
        audio_file = tts_request(text_chunk, gen_id)
        if audio_file: audio_q.put((gen_id, audio_file, text_chunk))
    except Exception as e: print(f"[TTS_REQUESTER_ERROR] ‼️ Thread error for chunk '{text_chunk[:30]}...': {e}"); traceback.print_exc()

def prefetch_worker():
    print("[SYSTEM] ⏳ Prefetch worker thread started (Sequential for Pipelining).")
    while not program_is_shutting_down.is_set():
        try: gen_id, chunks_to_prefetch = prefetch_q.get(timeout=0.5)
        except queue.Empty: continue
        if chunks_to_prefetch is None: print("[PREFETCH_WORKER] Shutdown sentinel."); prefetch_q.task_done(); break
        if gen_id != current_generation_id: prefetch_q.task_done(); continue
        for i, text_chunk_candidate in enumerate(chunks_to_prefetch):
            if program_is_shutting_down.is_set() or interruption_requested.is_set() or gen_id != current_generation_id:
                print("[PREFETCH_WORKER_INFO] Interruption/stale GenID during sequential batch. Stopping."); break
            if STANDALONE_TAG_RE.fullmatch(text_chunk_candidate.strip()):
                print(f"[PREFETCH_FILTER] DISCARD standalone tag: '{text_chunk_candidate[:50]}'", file=sys.stderr); continue
            text_for_word_count_filter = ACTION_RE_FOR_COUNT.sub('', text_chunk_candidate)
            text_for_word_count_filter = ANY_BRACKETED_TAG_RE.sub('', text_for_word_count_filter).strip()
            word_count_filter = len(text_for_word_count_filter.split()) if text_for_word_count_filter else 0
            if bool(ANY_BRACKETED_TAG_RE.search(text_chunk_candidate)) and word_count_filter <= WORD_COUNT_THRESHOLD_FOR_TAGGED_SENTENCE_DISCARD:
                print(f"[PREFETCH_FILTER] DISCARD short tagged chunk: '{text_chunk_candidate[:50]}'", file=sys.stderr); continue
            tts_processing_thread = threading.Thread(target=tts_requester_thread, args=(text_chunk_candidate, gen_id))
            tts_processing_thread.daemon = True; tts_processing_thread.start()
            tts_processing_thread.join(timeout=TTS_REQUEST_TIMEOUT[0] + TTS_REQUEST_TIMEOUT[1] + 15)
            if tts_processing_thread.is_alive():
                print(f"[PREFETCH_WORKER_WARN] ⚠️ TTS thread for chunk '{text_chunk_candidate[:30]}...' GenID {gen_id_short(gen_id)} timed out! Skipping rest of batch.", file=sys.stderr)
                break
        prefetch_q.task_done()
    print("[SYSTEM] ⏳ Prefetch worker thread finished.")

rough_tokens = lambda txt: max(1, len(txt)//4)
def split_sentences(txt_input: str) -> list[str]:
    pat = re.compile(r"""(<[^>]+>)|((?:[^.!?<>]| \.(?!\s*\.) )+(?:(?:\.{2,3}|[.!?])(?=\s|$) | (?=\s+[A-Z<]) | $ )?)""", re.VERBOSE)
    sents = []; [(sents.append(c.strip())) for t, s in pat.findall(txt_input) if (c := t or s)]; return sents
def batch_sentences(sents: list[str], max_words_per_chunk: int) -> list[str]:
    o,b,w = [],[],0
    for s_item in sents:
        t = ANY_BRACKETED_TAG_RE.sub('',s_item).strip(); w_s = len(t.split()) if t else 0
        if STANDALONE_TAG_RE.fullmatch(s_item.strip()): (o.append(" ".join(b)) if b else None); b,w=[],0; o.append(s_item); continue
        if w_s==0 and s_item.strip(): (b.append(s_item) if b else None); continue
        if w+w_s > max_words_per_chunk and b: o.append(" ".join(b)); b,w=[s_item],w_s
        else: b.append(s_item); w+=w_s
    if b: o.append(" ".join(b)); return o

def summarise(prev_summary, new_transcript):
    if program_is_shutting_down.is_set(): return prev_summary or "[summary skipped due to shutdown]"
    try:
        content = f"PREVIOUS SUMMARY:\n{prev_summary or 'None.'}\n\nNEW TRANSCRIPT TO ADD:\n{new_transcript}"
        r = requests.post(LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}], "temperature": 0.7, "max_tokens": 2048, "stop": ["\nUser:", "\nAssistant:"]}, timeout=200)
        r.raise_for_status(); summary_text = r.json()['choices'][0]['message']['content'].strip()
        for prefix in ["updated summary:", "here is an updated summary:", "here's the updated summary:"]:
            if summary_text.lower().startswith(prefix): summary_text = summary_text[len(prefix):].strip(); break
        return summary_text
    except Exception as e: print(f"[SUMMARIZER_ERROR] ‼️ Summary failed: {e}", file=sys.stderr); traceback.print_exc(); return prev_summary or "[summary generation failed]"

def hermes_chat(msgs):
    llm_response_stream = None
    try:
        llm_response_stream = requests.post( LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": msgs, "temperature": 0.7, "stream": True}, stream=True, timeout=(10, 200))
        llm_response_stream.raise_for_status(); buffer = ""
        for chunk in llm_response_stream.iter_content(chunk_size=None, decode_unicode=True):
            if program_is_shutting_down.is_set() or interruption_requested.is_set(): print("\n[LLM_STREAM_INFO] 🚫 INTERRUPT/SHUTDOWN.", file=sys.stderr); break
            buffer += chunk
            while 'data: ' in buffer and '\n' in buffer:
                try:
                    event_start_index = buffer.find('data: '); line_end_index = buffer.find('\n', event_start_index)
                    if event_start_index == -1 or line_end_index == -1: break
                    line_with_data_prefix = buffer[event_start_index : line_end_index]
                    if line_with_data_prefix.startswith('data: '):
                        data_segment = line_with_data_prefix[6:].strip()
                        if data_segment == '[DONE]': buffer = buffer[line_end_index+1:]; return
                        if data_segment:
                            delta = json.loads(data_segment)['choices'][0]['delta']
                            if content_token := delta.get('content'): yield content_token
                    buffer = buffer[line_end_index+1:]
                except Exception:
                    next_data_idx = buffer.find('data: ', event_start_index + 1 if 'event_start_index' in locals() else 0)
                    if next_data_idx != -1: buffer = buffer[next_data_idx:]
                    else:
                        if len(buffer) > 4096: print(f"[LLM_STREAM_ERROR] ‼️ Hermes buffer grew too large.", file=sys.stderr); return
                        break
    except requests.exceptions.RequestException as e: print(f"[LLM_STREAM_ERROR] ‼️ Hermes network error: {e}", file=sys.stderr); traceback.print_exc()
    except Exception as e_hermes: print(f"[LLM_STREAM_ERROR] ‼️ Unhandled Hermes error: {e_hermes}", file=sys.stderr); traceback.print_exc()
    finally:
        if llm_response_stream:
            try: llm_response_stream.close()
            except Exception: pass

def cleanup_old_audio():
    print("[SYSTEM] 🧹 Cleaning up old audio files...")
    count = 0
    for folder_path in CLEANUP_FOLDERS:
        if not folder_path.is_dir(): continue
        for audio_file in folder_path.glob("*.wav"):
            try:
                if CACHE_DIR.resolve() in audio_file.resolve().parents or CACHE_DIR.resolve() == audio_file.resolve().parent: continue
                audio_file.unlink(); count += 1
            except Exception as e: print(f"[CLEANUP_ERROR] 🧹 Could not delete {audio_file}: {e}", file=sys.stderr)
    if count > 0: print(f"[SYSTEM] 🧹 Removed {count} old .wav file(s).")
    else: print("[SYSTEM] 🧹 No old .wav files to remove.")

def main():
    global current_generation_id
    cleanup_old_audio(); interruption_requested.clear(); tts_actively_playing.clear(); program_is_shutting_down.clear()
    chat_history=[{"role":"system","content":PERSONA_TEMPLATE}]; current_memories = ""
    interrupt_audio_files, pending_audio_files, long_query_audio_files = [], [], []
    if CACHE_DIR.is_dir():
        interrupt_audio_files = list(CACHE_DIR.glob("*_interrupt_cache.wav")); pending_audio_files = list(CACHE_DIR.glob("*_pending_cache.wav")); long_query_audio_files = list(CACHE_DIR.glob("*_long_query_cache.wav"))
        if interrupt_audio_files: print(f"[SYSTEM_INIT] Found {len(interrupt_audio_files)} interrupt clips.")
        if pending_audio_files: print(f"[SYSTEM_INIT] Found {len(pending_audio_files)} pending clips.")
        if long_query_audio_files: print(f"[SYSTEM_INIT] Found {len(long_query_audio_files)} long query clips.")
    else: print(f"[SYSTEM_INIT_WARN] ⚠️ Cache directory '{CACHE_DIR}' not found, filler audio disabled.")

    def memory_manager_worker():
        nonlocal current_memories; print("[SYSTEM] 🧠 Memory manager thread started.")
        while not program_is_shutting_down.is_set():
            try: prev_mems, transcript_to_add = memory_q.get(timeout=0.5)
            except queue.Empty: continue
            if prev_mems is None and transcript_to_add is None: print("[MEMORY_MANAGER] Shutdown sentinel."); break
            current_memories = summarise(prev_mems, transcript_to_add)
            print(f"[MEMORY_MANAGER] 🔹 Memories updated. Length: {len(current_memories)}. Preview: '{current_memories[:100].replace(os.linesep, ' ')}...'")
            memory_q.task_done()
        print("[SYSTEM] 🧠 Memory manager thread finished.")

    threads = [(threading.Thread(target=f,daemon=True, name=n)) for f,d,n in
               [(audio_player,True,"AudioPlayerThread"), (prefetch_worker,True,"PrefetchThread"),
                (asr_listener,True,"ASRListenerThread"), (kb_listener,True,"KBListenerThread"),
                (memory_manager_worker,True,"MemoryManagerThread")]]
    for t in threads: t.start()
    print("🎙️  Talk or type anytime — 'quit' to exit, 'mem'/'dump' commands, or interrupt (e.g., 'stop').")

    sentence_ender = re.compile(r"""(?<!\b[A-Z][a-z]\.)(?:(?<=[.?!])|(?<=\.{2})|(?<=\.{3}))\s""", re.VERBOSE)

    try:
        while not program_is_shutting_down.is_set():
            current_user_input = ""; speech_duration = 0.0
            try:
                item = user_q.get(timeout=0.5)
                current_user_input, speech_duration = item if isinstance(item, tuple) and len(item) == 2 else (str(item), 0.0)
                user_q.task_done()
            except queue.Empty:
                if program_is_shutting_down.is_set(): break
                continue

            was_interruption_flag_originally_set = interruption_requested.is_set()
            filler_audio_queued_this_turn = False
            if was_interruption_flag_originally_set:
                print("[SYSTEM_INTERRUPT] Interruption flag was set. Processing cleanup...")
                current_generation_id = None
                if tts_actively_playing.is_set(): sd.stop(ignore_errors=True); tts_actively_playing.clear()
                for q_to_clear in [audio_q, prefetch_q]:
                    try:
                        while True:
                            item_in_q = q_to_clear.get_nowait()
                            if isinstance(item_in_q, tuple) and len(item_in_q) > 1 and isinstance(item_in_q[1], pathlib.Path):
                                p = item_in_q[1]
                                if p and p.exists() and not p.is_relative_to(CACHE_DIR): p.unlink(missing_ok=True)
                            q_to_clear.task_done()
                    except queue.Empty: pass
                if interrupt_audio_files:
                    try: data, sr_clip = sf.read(str(random.choice(interrupt_audio_files)), dtype='float32'); sd.play(data, sr_clip); sd.wait()
                    except Exception as e: print(f"[SYSTEM_INTERRUPT_ERROR] ‼️ Could not play interrupt clip: {e}", file=sys.stderr)
                interruption_requested.clear(); print("[SYSTEM_INTERRUPT] Interruption flag processed and cleared.")
                if any(k in current_user_input.lower().strip(".?!, ") for k in INTERRUPT_KEYWORDS):
                    print("[SYSTEM_INTERRUPT_INFO] Interrupt was a 'stop' command."); current_user_input = ""; continue
                else: print("[SYSTEM_INTERRUPT_INFO] Interrupt was new content.")

            if not current_user_input.strip(): continue

            if current_user_input.lower() in {"quit","exit"}: print("[SYSTEM] Quit command received."); program_is_shutting_down.set(); break
            if current_user_input.lower() == "mem": print(f"\n[SYSTEM_COMMAND]\n--- MEMORY ---\n{current_memories or '[none]'}\n--------------"); continue
            if current_user_input.lower() == "dump": print("\n[SYSTEM_COMMAND]\n--- CHAT HISTORY DUMP ---"); [print(f"[{i}] [{m['role']}] {m['content']}") for i, m in enumerate(chat_history)]; print("-------------------------"); continue

            current_generation_id = uuid.uuid4()
            print(f"\n[SYSTEM] New turn. User: '{current_user_input[:60]}...'. GenID: {gen_id_short(current_generation_id)}")
            messages_for_llm = list(chat_history)
            system_prompt_content = PERSONA_TEMPLATE + (f"\n\n--- CONVERSATION MEMORIES ---\n{current_memories}" if current_memories else "")
            if messages_for_llm and messages_for_llm[0]['role'] == 'system': messages_for_llm[0]['content'] = system_prompt_content
            else: messages_for_llm.insert(0, {"role": "system", "content": system_prompt_content})
            messages_for_llm.append({"role": "user", "content": current_user_input})

            if was_interruption_flag_originally_set and not any(k in current_user_input.lower().strip(".?!, ") for k in INTERRUPT_KEYWORDS):
                filler_clip_path = None
                if speech_duration >= INTERRUPTION_DURATION_THRESHOLD_S and long_query_audio_files: filler_clip_path = random.choice(long_query_audio_files)
                elif pending_audio_files: filler_clip_path = random.choice(pending_audio_files)
                if filler_clip_path: audio_q.put((current_generation_id, filler_clip_path, "[FILLER_AUDIO]")); filler_audio_queued_this_turn = True

            print(f"[ASSISTANT_THINKING] Gen: {gen_id_short(current_generation_id)}")

            llm_full_response = ""
            all_chunks_for_this_response = []
            current_segment_buffer_raw = ""
            first_chunk_for_tts_identified = False
            in_lookahead_for_current_segment = False
            lookahead_expiry_time = 0
            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Initializing chunking logic.") # DEBUG

            try:
                for token_chunk in hermes_chat(messages_for_llm):
                    if interruption_requested.is_set():
                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Interruption during token stream."); # DEBUG
                        break

                    llm_full_response += token_chunk
                    current_segment_buffer_raw += token_chunk
                    # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Token added. Buffer now: '{current_segment_buffer_raw[:80]}...'") # DEBUG


                    # Inner loop to make as many chunks as possible
                    while True:
                        if interruption_requested.is_set():
                            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Interruption in inner chunking loop."); # DEBUG
                            break

                        if not current_segment_buffer_raw.strip():
                            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Inner loop: Buffer empty. Breaking to get more tokens.") # DEBUG
                            break

                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Inner loop evaluating buffer: '{current_segment_buffer_raw[:80]}...'") # DEBUG
                        text_for_word_counting = ACTION_RE_FOR_COUNT.sub('', current_segment_buffer_raw)
                        text_for_word_counting = ANY_BRACKETED_TAG_RE.sub('', text_for_word_counting).strip()
                        speakable_words_in_buffer = len(text_for_word_counting.split()) if text_for_word_counting else 0
                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Speakable words in buffer: {speakable_words_in_buffer}") # DEBUG

                        is_first = not first_chunk_for_tts_identified
                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Is first: {is_first}") # DEBUG

                        chunk_to_add_raw = None

                        if in_lookahead_for_current_segment:
                            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Currently in lookahead. Expiry: {lookahead_expiry_time:.2f}, Now: {time.monotonic():.2f}") # DEBUG
                            if time.monotonic() < lookahead_expiry_time:
                                # Still in lookahead. If current buffer now ends in sentence, FIRE.
                                match = sentence_ender.search(current_segment_buffer_raw)
                                if match:
                                    # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Lookahead active & sentence ended. Firing.") # DEBUG
                                    chunk_to_add_raw = current_segment_buffer_raw[:match.end()]
                                else: # Still in lookahead, no sentence end yet, break inner to get more tokens from LLM.
                                    # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Lookahead active, no sentence end. Breaking for more tokens.") # DEBUG
                                    break
                            else: # Lookahead expired
                                # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Lookahead expired.") # DEBUG
                                in_lookahead_for_current_segment = False
                                # Force evaluate the current buffer as if not in lookahead
                                # If it now has a sentence end, it will be caught by logic below.
                                # If not, and it met a milestone, it might start a new lookahead or be cut.
                                # This ensures we re-evaluate after lookahead completes.
                                match = sentence_ender.search(current_segment_buffer_raw)
                                if match: # If sentence ended during or by end of lookahead
                                     # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Lookahead expired, sentence end found. Firing.") # DEBUG
                                     chunk_to_add_raw = current_segment_buffer_raw[:match.end()]
                                else: # Lookahead expired, still no sentence end.
                                      # If it's a subsequent chunk that hit CHUNK_WORD_MAX, it will be handled by hard cap.
                                      # If it was first chunk, it means MIN_FIRST_WORDS was met, lookahead expired, still no sentence end.
                                      # This chunk will be force-cut below if it meets CHUNK_WORD_MAX as a subsequent,
                                      # or kept accumulating if it doesn't meet any other criteria.
                                      # For a first chunk that started lookahead, this is the "force cut or keep accumulating" point.
                                      # Let's assume for now that if lookahead expires and no sentence, we keep accumulating unless a hard cap is hit.
                                      # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Lookahead expired, no sentence end. Will re-evaluate or break for more tokens.") # DEBUG
                                      pass # Fall through to normal evaluation without lookahead active

                        if not in_lookahead_for_current_segment and chunk_to_add_raw is None: # Normal evaluation or post-expired-lookahead re-evaluation
                            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Normal/Post-Lookahead Evaluation. Buffer: '{current_segment_buffer_raw[:70]}...'") # DEBUG
                            if is_first:
                                milestone_to_check = MIN_FIRST_WORDS
                                # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] First chunk. Milestone: {milestone_to_check}, Speakable: {speakable_words_in_buffer}") # DEBUG
                                if speakable_words_in_buffer >= milestone_to_check:
                                    match = sentence_ender.search(current_segment_buffer_raw)
                                    if match:
                                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] First chunk: Milestone & sentence end. Firing.") # DEBUG
                                        chunk_to_add_raw = current_segment_buffer_raw[:match.end()]
                                    else:
                                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] First chunk: Milestone, no sentence end. Activating lookahead.") # DEBUG
                                        in_lookahead_for_current_segment = True
                                        lookahead_expiry_time = time.monotonic() + LLM_CHUNK_LOOKAHEAD_S
                                        break
                                else: break # Not enough for first chunk milestone
                            else: # Subsequent chunk logic
                                # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Subsequent chunk. Speakable: {speakable_words_in_buffer}, MinSub: {MIN_SUBSEQUENT_WORDS}, ChunkMax: {CHUNK_WORD_MAX}") # DEBUG
                                if speakable_words_in_buffer >= CHUNK_WORD_MAX: # Hard cap
                                    # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Subsequent: Met CHUNK_WORD_MAX.") # DEBUG
                                    match = sentence_ender.search(current_segment_buffer_raw)
                                    if match:
                                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Subsequent: Hard cap & sentence end. Firing.") # DEBUG
                                        chunk_to_add_raw = current_segment_buffer_raw[:match.end()]
                                    else:
                                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Subsequent: Hard cap, no sentence. Taking whole buffer. (Consider cutting at word approx)") # DEBUG
                                        chunk_to_add_raw = current_segment_buffer_raw
                                elif speakable_words_in_buffer >= MIN_SUBSEQUENT_WORDS:
                                    match = sentence_ender.search(current_segment_buffer_raw)
                                    if match:
                                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Subsequent: Natural sentence end. Firing.") # DEBUG
                                        chunk_to_add_raw = current_segment_buffer_raw[:match.end()]
                                    else:
                                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Subsequent: Met min words, no sentence end. Activating lookahead.") # DEBUG
                                        in_lookahead_for_current_segment = True
                                        lookahead_expiry_time = time.monotonic() + LLM_CHUNK_LOOKAHEAD_S
                                        break
                                else: break # Not enough for subsequent chunk natural sentence

                        if chunk_to_add_raw:
                            stripped_chunk = chunk_to_add_raw.strip()
                            if stripped_chunk:
                                print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] FIRING CHUNK: '{stripped_chunk[:70]}...'") # DEBUG
                                all_chunks_for_this_response.append(stripped_chunk)
                                current_segment_buffer_raw = current_segment_buffer_raw[len(chunk_to_add_raw):]
                                if is_first: first_chunk_for_tts_identified = True
                                in_lookahead_for_current_segment = False
                                # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Buffer remaining after fire: '{current_segment_buffer_raw[:70]}...'") # DEBUG
                                # After firing a chunk, immediately re-evaluate the new buffer (continue inner while loop)
                            else:
                                # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Fired chunk was empty after strip. Consuming raw part.") # DEBUG
                                current_segment_buffer_raw = current_segment_buffer_raw[len(chunk_to_add_raw):]
                                if not current_segment_buffer_raw.strip(): break
                        elif not in_lookahead_for_current_segment:
                            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] No chunk made and not starting lookahead. Breaking for more LLM tokens.") # DEBUG
                            break
                        elif in_lookahead_for_current_segment :
                             # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] In lookahead, didn't fire yet. Breaking for more LLM tokens or timer.") # DEBUG
                             break

                    if interruption_requested.is_set():
                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Interruption after inner chunking loop."); # DEBUG
                        break
                # --- End of LLM Streaming Loop ---
                # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] LLM Stream ended. Final buffer: '{current_segment_buffer_raw[:80]}...'") # DEBUG

                if current_segment_buffer_raw.strip():
                    final_text_raw = current_segment_buffer_raw.strip()
                    # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Processing final remaining buffer: '{final_text_raw[:70]}...'") # DEBUG
                    if not all_chunks_for_this_response and final_text_raw:
                        # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Final buffer is the ONLY response. Splitting by sentences.") # DEBUG
                        for sentence in split_sentences(final_text_raw):
                            if (clean_s := sentence.strip()): all_chunks_for_this_response.append(clean_s)
                    elif final_text_raw:
                        cleaned_final_for_count = ACTION_RE_FOR_COUNT.sub('', final_text_raw)
                        cleaned_final_for_count = ANY_BRACKETED_TAG_RE.sub('', cleaned_final_for_count).strip()
                        if len(cleaned_final_for_count.split()) > CHUNK_WORD_MAX * 1.2 and \
                           any(s_end in final_text_raw for s_end in ['.', '?', '!']):
                            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Final remaining text is long tail, splitting.") # DEBUG
                            for sentence in split_sentences(final_text_raw):
                                if (clean_s := sentence.strip()): all_chunks_for_this_response.append(clean_s)
                        else:
                            # print(f"[CHUNK_LOG|{gen_id_short(current_generation_id)}] Adding final remaining text as single chunk.") # DEBUG
                            all_chunks_for_this_response.append(final_text_raw)

                if llm_full_response.strip(): print(f"\n[ASSISTANT_RESPONSE] Gen: {gen_id_short(current_generation_id)}\n{llm_full_response.strip()}")
                else: print(f"\n[ASSISTANT_RESPONSE] Gen: {gen_id_short(current_generation_id)} (No textual response)")

                if interruption_requested.is_set():
                    if current_user_input: chat_history.append({"role": "user", "content": current_user_input}); continue

                if all_chunks_for_this_response:
                    print(f"[TTS_QUEUE_BATCH] Queuing {len(all_chunks_for_this_response)} chunks (Gen: {gen_id_short(current_generation_id)}):")
                    for i, chunk_text in enumerate(all_chunks_for_this_response): print(f"  [TTS_QUEUE_ITEM {i+1}] '{chunk_text[:70]}...'")
                    prefetch_q.put((current_generation_id, all_chunks_for_this_response))
                elif llm_full_response.strip():
                    print(f"[TTS_QUEUE_INFO] LLM response present but no chunks for TTS (Gen: {gen_id_short(current_generation_id)}).")

            except Exception as e:
                print(f"\n[LLM_ERROR] ‼️ Error during LLM streaming/batching: {e}", file=sys.stderr); traceback.print_exc()
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input}); continue

            assistant_response_for_chat_history = strip_action_and_emoji(llm_full_response)
            if assistant_response_for_chat_history:
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                chat_history.append({"role": "assistant", "content": assistant_response_for_chat_history})
            elif current_user_input :
                if not chat_history or chat_history[-1].get('content') != current_user_input or chat_history[-1].get('role') != 'user':
                     chat_history.append({"role": "user", "content": current_user_input})
                print(f"[SYSTEM_INFO] Assistant response empty/actions only for GenID {gen_id_short(current_generation_id)}. Continuing."); continue
            total_tokens = sum(rough_tokens(m['content']) for m in chat_history)
            if total_tokens > TOKEN_SOFT_LIMIT and len(chat_history) > (KEEP_RECENT * 2 + 1):
                to_summarize_count = len(chat_history) - (KEEP_RECENT * 2 +1)
                if to_summarize_count > 0:
                    transcript_for_summary = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history[1:-(KEEP_RECENT*2)])
                    if transcript_for_summary: memory_q.put((current_memories, transcript_for_summary)); chat_history = [chat_history[0]] + chat_history[-(KEEP_RECENT*2):]
            current_tokens_pruned = sum(rough_tokens(m['content']) for m in chat_history)
            while current_tokens_pruned > TOKEN_HARD_LIMIT and len(chat_history) > 2:
                removed = chat_history.pop(1); print(f"[CHAT_HISTORY_WARN] ‼️ HARD TOKEN PRUNING: Removing '{removed['content'][:30]}...'"); current_tokens_pruned = sum(rough_tokens(m['content']) for m in chat_history)

    except KeyboardInterrupt: print("\n[SYSTEM_ERROR] 🚨 Main loop: KeyboardInterrupt."); program_is_shutting_down.set()
    except Exception as e: print(f"\n[SYSTEM_ERROR] ‼️ UNHANDLED EXCEPTION IN MAIN: {e}"); traceback.print_exc(); program_is_shutting_down.set()
    finally:
        print("[SYSTEM] 🛑 Main loop finished. Finalizing shutdown...")
        if not program_is_shutting_down.is_set(): program_is_shutting_down.set()
        global_current_gen_id_at_shutdown = current_generation_id; current_generation_id = None
        print(f"[SYSTEM] 🛑 Active GenID at shutdown start: {gen_id_short(global_current_gen_id_at_shutdown)}. Now set to None.")
        print("[SYSTEM] 🛑 Sending sentinels to worker queues...");
        audio_q.put((None, None, None)); prefetch_q.put((None, None)); memory_q.put((None,None))
        active_threads = [t for t in threading.enumerate() if t is not threading.main_thread() and t.name in [th.name for th in threads]]
        for t in active_threads:
            print(f"[SYSTEM] 🛑 Waiting for {t.name} to join (5s)..."); t.join(timeout=5.0)
            if t.is_alive(): print(f"[SYSTEM_WARN] ⚠️ {t.name} did not join cleanly.")
            else: print(f"[SYSTEM] ✅ {t.name} joined.")
        print("[SYSTEM] 🛑 Stopping any remaining audio playback..."); sd.stop(ignore_errors=True)
        print("[SYSTEM] 🧹 Performing final audio cleanup..."); cleanup_old_audio();
        print("👋 Goodbye!")

if __name__=="__main__":
    main()