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

LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
ORPHEUS_API_URL    = "http://127.0.0.1:5005/v1/audio/speech"
VOICE              = "tara"
MIN_FIRST_WORDS    = 15
CHUNK_WORD_MAX     = 25
TOKEN_SOFT_LIMIT   = 2500
TOKEN_HARD_LIMIT   = 7700
KEEP_RECENT        = 2

TTS_REQUEST_TIMEOUT = (10, 30)
WORD_COUNT_THRESHOLD_FOR_TAGGED_SENTENCE_DISCARD = 2
STANDALONE_TAG_RE = re.compile(r"^\s*<[^>]+>\s*$")
ANY_BRACKETED_TAG_RE = re.compile(r"<[^>]+>")
UNWANTED_INTERJECTIONS_RE = re.compile(
    r"""\b(hehe|haha|hihi|huhu|hoho)(?:\.{2,3})?(?!\w)\s*""",
    re.IGNORECASE | re.VERBOSE
)

OUTPUT_DIR = pathlib.Path("outputs")
CACHE_DIR  = pathlib.Path("../cached_clips")
OUTPUT_DIR.mkdir(exist_ok=True)

CLEANUP_FOLDERS = [ OUTPUT_DIR, pathlib.Path("../outputs"),]

SUMMARY_BOT_TEMPLATE    = pathlib.Path("../personas/chat_summary_bot.txt")
PERSONA_PROMPT_TEMPLATE = pathlib.Path("../personas/tts_default.txt")

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

ACTION_RE_CLEAN = re.compile(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]')
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
    print("🎤 Loading Whisper ASR & Silero VAD…")
    try:
        asr_model = whisper.load_model("small.en"); print("🎤 Whisper 'small.en' model loaded.")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (get_speech_timestamps, _, _, _, _) = utils; print("🎤 Silero VAD model loaded.")
    except Exception as e: print(f"‼️ FATAL: Could not load ASR/VAD models: {e}"); traceback.print_exc(); program_is_shutting_down.set(); return
    sr = 16000; vad_chunk_size = 512; vad_chunk_duration_s = vad_chunk_size / sr; min_speech_frames = int(VAD_MIN_SPEECH_S / vad_chunk_duration_s)
    is_speaking = False; audio_buffer_list = []; silence_counter_frames = 0
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=vad_chunk_size, callback=None) as stream:
            print("🎤 Continuous ASR listening (Literal Transcription Mode)…")
            while not program_is_shutting_down.is_set():
                try:
                    frame_float32, overflowed = stream.read(vad_chunk_size)
                    if overflowed: print("‼️ Mic overflow!", file=sys.stderr); continue
                    audio_tensor = torch.from_numpy(frame_float32.flatten()); speech_confidence = vad_model(audio_tensor, sr).item()
                    is_speech_in_current_frame = speech_confidence > VAD_SPEECH_CONFIDENCE_THRESHOLD
                    if is_speaking:
                        audio_buffer_list.append(frame_float32)
                        if is_speech_in_current_frame: silence_counter_frames = 0
                        else:
                            silence_counter_frames += 1; active_timeout_s = VAD_DEFAULT_SILENCE_TIMEOUT_S
                            if tts_actively_playing.is_set(): active_timeout_s = VAD_INTERRUPT_TIMEOUT_S
                            else:
                                if audio_buffer_list:
                                    current_audio_data_np = np.concatenate(audio_buffer_list).squeeze()
                                    if current_audio_data_np.ndim == 0: current_audio_data_np = np.array([current_audio_data_np])
                                    current_audio_data = torch.from_numpy(current_audio_data_np)
                                    speech_ts = get_speech_timestamps(current_audio_data, vad_model, sampling_rate=sr, min_speech_duration_ms=int(VAD_MIN_SPEECH_S * 1000))
                                    if speech_ts:
                                        precise_speech_duration_s = (speech_ts[-1]['end'] - speech_ts[0]['start']) / sr
                                        if precise_speech_duration_s > VAD_LONG_PROMPT_TRIGGER_S: active_timeout_s = VAD_LONG_PROMPT_TIMEOUT_S
                            silence_duration_s_calculated = silence_counter_frames * vad_chunk_duration_s
                            if silence_duration_s_calculated >= active_timeout_s:
                                is_speaking = False; end_of_speech_time = time.monotonic()
                                if len(audio_buffer_list) >= min_speech_frames:
                                    full_audio_np = np.concatenate(audio_buffer_list).squeeze()
                                    if full_audio_np.ndim == 0: full_audio_np = np.array([full_audio_np])
                                    final_speech_ts = get_speech_timestamps(torch.from_numpy(full_audio_np), vad_model, sampling_rate=sr, min_speech_duration_ms=int(VAD_MIN_SPEECH_S * 1000))
                                    speech_duration_s_for_userq = (final_speech_ts[-1]['end'] - final_speech_ts[0]['start']) / sr if final_speech_ts else 0.0
                                    result = asr_model.transcribe(full_audio_np, fp16=torch.cuda.is_available(), beam_size=WHISPER_BEAM_SIZE, logprob_threshold=WHISPER_LOGPROB_THRESHOLD, no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD, condition_on_previous_text=False)
                                    txt = result['text'].strip(); processing_lag_s = time.monotonic() - end_of_speech_time
                                    if txt:
                                        is_echo = False
                                        if tts_actively_playing.is_set() and last_tts_text:
                                            similarity = SequenceMatcher(None, txt.lower(), last_tts_text.lower()).ratio()
                                            if similarity > ECHO_SIMILARITY_THRESHOLD: print(f"🎤 Echo detected (Similarity: {similarity:.2f}), discarding.", file=sys.stderr); is_echo = True
                                        if not is_echo and len(txt.split()) >= MIN_WORDS_ASR:
                                            print(f"\n🗣️ User (ASR): '{txt}' (Duration: {speech_duration_s_for_userq:.2f}s, VAD Lag: {processing_lag_s:.2f}s)")
                                            if tts_actively_playing.is_set() and not interruption_requested.is_set(): print("🎤 ASR: User spoke while TTS active -> Setting INTERRUPT_REQUESTED"); interruption_requested.set()
                                            user_q.put((txt, speech_duration_s_for_userq))
                                audio_buffer_list = []; silence_counter_frames = 0
                    elif is_speech_in_current_frame:
                        is_speaking = True; silence_counter_frames = 0; audio_buffer_list = [frame_float32]
                        if not tts_actively_playing.is_set(): print("\n🎤 Speech detected…", end='', flush=True)
                except sd.PortAudioError as pae:
                    if "Input overflowed" in str(pae): print("‼️ Mic overflow (PortAudioError)!", file=sys.stderr); continue
                    print(f"‼️ ASR PortAudioError: {pae}", file=sys.stderr); sd.sleep(1) # type: ignore
                except Exception as e:
                    if program_is_shutting_down.is_set(): break
                    print(f"‼️ ASR loop error: {e}", file=sys.stderr); traceback.print_exc(); is_speaking = False; audio_buffer_list = []; silence_counter_frames = 0; sd.sleep(1) # type: ignore
    except Exception as e: print(f"‼️ ASR: Could not open InputStream: {e}"); traceback.print_exc()
    print("🎤 ASR listener thread finished.")

def kb_listener():
    print("⌨️ Keyboard listener started.")
    while not program_is_shutting_down.is_set():
        try:
            line = input("\n⌨️  ").strip()
            if program_is_shutting_down.is_set(): break
            if line:
                if tts_actively_playing.is_set() and line.lower() in INTERRUPT_KEYWORDS:
                    print(f"⌨️ Interrupt command: '{line}' -> Setting INTERRUPT_REQUESTED");
                    if not interruption_requested.is_set(): interruption_requested.set()
                user_q.put((line, 0.0))
        except EOFError: print("⌨️ EOF received, exiting keyboard listener."); break
        except KeyboardInterrupt: print("\n⌨️ Keyboard interrupt in listener. Exiting."); break
        except Exception as e:
            if program_is_shutting_down.is_set(): break
            print(f"‼️ KB Listener error: {e}"); traceback.print_exc(); break
    print("⌨️ Keyboard listener thread finished.")

def audio_player():
    global last_tts_text
    print("🎵 Audio player thread started.")
    while not program_is_shutting_down.is_set():
        try:
            gen_id, p_path_obj, text_that_was_spoken = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if p_path_obj is None:
            print("🎵 AudioPlayer: Shutdown sentinel received.\n"); break

        if interruption_requested.is_set():
            if tts_actively_playing.is_set():
                sd.stop(); tts_actively_playing.clear() # type: ignore
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                p_path_obj.unlink(missing_ok=True)
            audio_q.task_done(); continue

        is_filler_audio = (text_that_was_spoken == "[FILLER_AUDIO]")

        if gen_id == current_generation_id:
            try:
                if not p_path_obj or not p_path_obj.exists():
                    print(f"🎵 AudioPlayer: Audio file path invalid or DNE: {p_path_obj}\n", file=sys.stderr)
                    audio_q.task_done(); continue

                if tts_actively_playing.is_set(): # Should not happen if sd.wait() is effective and interruptions are handled.
                    print(f"🎵 AudioPlayer: TTS was already active. Stopping previous before playing '{p_path_obj.name}'.\n", file=sys.stderr)
                    sd.stop() # type: ignore
                    tts_actively_playing.clear()

                data, sr_audio = sf.read(str(p_path_obj), dtype="float32")
                tts_actively_playing.set()
                if is_filler_audio: print(f"🎵 AudioPlayer: Playing FILLER audio: {p_path_obj.name}\n")
                else: print(f"🎵 AudioPlayer:\n Playing TTS audio: '{text_that_was_spoken}' (File: {p_path_obj.name}, Gen: {gen_id_short(gen_id)})\n")

                sd.play(data, sr_audio)
                sd.wait()

                if not is_filler_audio: last_tts_text = text_that_was_spoken
            except Exception as e:
                print(f"‼️ Audio playback error for '{p_path_obj}': {e}", file=sys.stderr); traceback.print_exc()
                if tts_actively_playing.is_set(): tts_actively_playing.clear()
            finally:
                tts_actively_playing.clear()
                if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                    try: p_path_obj.unlink(missing_ok=True)
                    except Exception as del_e: print(f"🎵 AudioPlayer: Error deleting file {p_path_obj.name}: {del_e}\n", file=sys.stderr)
                audio_q.task_done()
        else: # Stale audio
            print(f"🎵 AudioPlayer: Discarding stale audio chunk '{p_path_obj.name if p_path_obj else 'N/A'}' from GenID {gen_id_short(gen_id)} (current global GenID: {gen_id_short(current_generation_id)}).\n", file=sys.stderr)
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                p_path_obj.unlink(missing_ok=True)
            audio_q.task_done()

    print("🎵 Audio player thread finished.")

def gen_id_short(gen_id): return str(gen_id)[-6:] if gen_id else "None"

def tts_request(txt, gen_id):
    if program_is_shutting_down.is_set() or (gen_id is not None and gen_id != current_generation_id): return None
    payload_text = clean_text_for_tts_payload(txt)
    if not payload_text: return None
    try:
        response = requests.post(ORPHEUS_API_URL, json={"input": payload_text, "model":"orpheus", "voice":VOICE, "response_format":"wav", "speed":1.0}, timeout=TTS_REQUEST_TIMEOUT)
        response.raise_for_status(); wav_data = response.content
        if len(wav_data) < 1000 and not (wav_data.startswith(b"RIFF") and len(wav_data) > 44): print(f"   -> 📡 TTS API: Short/invalid WAV ({len(wav_data)}b) for '{payload_text[:30]}...'. Skip."); return None
        file_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+".wav"; output_path = OUTPUT_DIR/file_name
        output_path.write_bytes(wav_data); return output_path
    except requests.exceptions.Timeout as et: print(f"‼️ TTS_REQUEST_LOG: TIMEOUT for '{payload_text[:30]}' GenID: {gen_id_short(gen_id)} Error: {et}", file=sys.stderr); return None
    except requests.exceptions.RequestException as er: print(f"‼️ TTS_REQUEST_LOG: FAILED for '{payload_text[:30]}' GenID: {gen_id_short(gen_id)} Error: {er}", file=sys.stderr); return None
    except Exception as e: print(f"‼️ TTS_REQUEST_LOG: UNKNOWN ERR for '{payload_text[:30]}' GenID: {gen_id_short(gen_id)} Error: {e}", file=sys.stderr); return None

def tts_requester_thread(text_chunk, gen_id):
    try:
        if program_is_shutting_down.is_set() or gen_id != current_generation_id: return
        audio_file = tts_request(text_chunk, gen_id)
        if audio_file: audio_q.put((gen_id, audio_file, text_chunk))
    except Exception as e: print(f"‼️ TTS Requester Thread error for chunk '{text_chunk[:30]}...': {e}"); traceback.print_exc()

def prefetch_worker():
    print("⏳ Prefetch worker thread started.")
    while not program_is_shutting_down.is_set():
        try: gen_id, chunks_to_prefetch = prefetch_q.get(timeout=0.5)
        except queue.Empty: continue
        if chunks_to_prefetch is None: print("⏳ Prefetch worker: Shutdown sentinel received."); break
        if gen_id != current_generation_id: prefetch_q.task_done(); continue
        fetch_threads = []
        for text_chunk_candidate in chunks_to_prefetch:
            if program_is_shutting_down.is_set() or interruption_requested.is_set() or gen_id != current_generation_id: break
            if STANDALONE_TAG_RE.fullmatch(text_chunk_candidate.strip()): print(f"PREFETCH_FILTER: DISCARD standalone tag: '{text_chunk_candidate[:50]}'", file=sys.stderr); continue
            text_for_word_count = ANY_BRACKETED_TAG_RE.sub('', text_chunk_candidate).strip()
            word_count = len(text_for_word_count.split()) if text_for_word_count else 0
            if bool(ANY_BRACKETED_TAG_RE.search(text_chunk_candidate)) and word_count <= WORD_COUNT_THRESHOLD_FOR_TAGGED_SENTENCE_DISCARD: print(f"PREFETCH_FILTER: DISCARD short tagged chunk: '{text_chunk_candidate[:50]}'", file=sys.stderr); continue
            thread = threading.Thread(target=tts_requester_thread, args=(text_chunk_candidate, gen_id)); thread.daemon = True; fetch_threads.append(thread); thread.start()
        if fetch_threads:
            for i, thread in enumerate(fetch_threads):
                thread.join(timeout=TTS_REQUEST_TIMEOUT[1] + 10)
                if thread.is_alive(): print(f"⚠️ PREFETCH_WORKER: TTS requester thread {i} for GenID {gen_id_short(gen_id)} DID NOT JOIN!", file=sys.stderr)
        prefetch_q.task_done()
    print("⏳ Prefetch worker thread finished.")

rough_tokens = lambda txt: max(1, len(txt)//4)

def split_sentences(txt_input: str) -> list[str]:
    pat = re.compile(
        r"""
        (<[^>]+>)
        |
        (
            (?:[^.!?<>]| \.(?!\s*\.) )+
            (?:
                (?:\.{2,3}|[.!?])
                 (?=\s|$)
              | (?=\s+[A-Z<])
              | $
            )?
        )
        """, re.VERBOSE
    )
    sentences = []
    for match_tuple in pat.findall(txt_input):
        tag_match, sentence_match = match_tuple[0], match_tuple[1]
        content = (tag_match or sentence_match or "").strip()
        if content: sentences.append(content)
    return sentences

def batch_sentences(sents: list[str], max_words_per_chunk: int) -> list[str]:
    output_batches, current_batch_sentences, current_word_count = [], [], 0
    for s_item in sents:
        text_content_of_s_item = ANY_BRACKETED_TAG_RE.sub('', s_item).strip()
        words_in_s_item = len(text_content_of_s_item.split()) if text_content_of_s_item else 0
        if STANDALONE_TAG_RE.fullmatch(s_item.strip()):
            if current_batch_sentences: output_batches.append(" ".join(current_batch_sentences)); current_batch_sentences, current_word_count = [], 0
            output_batches.append(s_item); continue
        if words_in_s_item == 0 and s_item.strip():
            if current_batch_sentences: current_batch_sentences.append(s_item)
            continue
        if current_word_count + words_in_s_item > max_words_per_chunk and current_batch_sentences:
            output_batches.append(" ".join(current_batch_sentences)); current_batch_sentences, current_word_count = [s_item], words_in_s_item
        else:
            current_batch_sentences.append(s_item); current_word_count += words_in_s_item
    if current_batch_sentences: output_batches.append(" ".join(current_batch_sentences))
    return output_batches

def summarise(prev_summary, new_transcript):
    if program_is_shutting_down.is_set(): return prev_summary or "[summary skipped due to shutdown]"
    try:
        content = f"PREVIOUS SUMMARY:\n{prev_summary or 'None.'}\n\nNEW TRANSCRIPT TO ADD (do not repeat content already in previous summary):\n{new_transcript}"
        r = requests.post(LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}], "temperature": 0.6, "max_tokens": 1024, "stop": ["\nUser:", "\nAssistant:"]}, timeout=200)
        r.raise_for_status(); summary_text = r.json()['choices'][0]['message']['content'].strip()
        prefixes_to_remove = ["updated summary:", "here is an updated summary:", "here's the updated summary:"]
        summary_text_lower = summary_text.lower()
        for prefix in prefixes_to_remove:
            if summary_text_lower.startswith(prefix): summary_text = summary_text[len(prefix):].strip(); break
        return summary_text
    except Exception as e: print(f"‼️ Summary failed: {e}", file=sys.stderr); traceback.print_exc(); return prev_summary or "[summary generation failed]"

def hermes_chat(msgs):
    llm_response_stream = None
    try:
        llm_response_stream = requests.post( LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": msgs, "temperature": 0.6, "stream": True}, stream=True, timeout=(10, 200))
        llm_response_stream.raise_for_status(); buffer = ""
        for chunk in llm_response_stream.iter_content(chunk_size=None, decode_unicode=True):
            if program_is_shutting_down.is_set() or interruption_requested.is_set(): print("\n🚫 LLM stream: INTERRUPT/SHUTDOWN.", file=sys.stderr); break
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
                        if len(buffer) > 4096: print(f"‼️ Hermes buffer grew too large with unparsable data.", file=sys.stderr); return
                        break
    except requests.exceptions.RequestException as e: print(f"‼️ Hermes network error: {e}", file=sys.stderr); traceback.print_exc()
    except Exception as e_hermes: print(f"‼️ Unhandled Hermes error: {e_hermes}", file=sys.stderr); traceback.print_exc()
    finally:
        if llm_response_stream:
            try: llm_response_stream.close()
            except Exception: pass

def cleanup_old_audio():
    print("🧹 Cleaning up old audio files...")
    count = 0
    for folder_path in CLEANUP_FOLDERS:
        if not folder_path.is_dir(): continue
        for audio_file in folder_path.glob("*.wav"):
            try:
                if CACHE_DIR.resolve() in audio_file.resolve().parents: continue
                audio_file.unlink(); count += 1
            except Exception as e: print(f"🧹 Could not delete {audio_file}: {e}", file=sys.stderr)
    if count > 0: print(f"🧹 Removed {count} old .wav file(s).")
    else: print("🧹 No old .wav files to remove from output directories.")

def main():
    global current_generation_id
    cleanup_old_audio(); interruption_requested.clear(); tts_actively_playing.clear(); program_is_shutting_down.clear()
    chat_history=[{"role":"system","content":PERSONA_TEMPLATE}]; current_memories = ""
    interrupt_audio_files, pending_audio_files, long_query_audio_files = [], [], []
    if CACHE_DIR.is_dir():
        interrupt_audio_files = list(CACHE_DIR.glob("*_interrupt_cache.wav")); pending_audio_files = list(CACHE_DIR.glob("*_pending_cache.wav")); long_query_audio_files = list(CACHE_DIR.glob("*_long_query_cache.wav"))
        if interrupt_audio_files: print(f"🎤 Found {len(interrupt_audio_files)} interrupt clips.")
        if pending_audio_files: print(f"🎤 Found {len(pending_audio_files)} pending clips.")
        if long_query_audio_files: print(f"🎤 Found {len(long_query_audio_files)} long query clips.")
    else: print(f"🎤 WARNING: Cache directory '{CACHE_DIR}' not found, filler audio disabled.")

    def memory_manager_worker():
        nonlocal current_memories; print("🧠 Memory manager thread started.")
        while not program_is_shutting_down.is_set():
            try: prev_mems, transcript_to_add = memory_q.get(timeout=0.5)
            except queue.Empty: continue
            if prev_mems is None and transcript_to_add is None: print("🧠 Memory manager: Shutdown sentinel."); break
            current_memories = summarise(prev_mems, transcript_to_add)
            print(f"🔹 Memories updated. Length: {len(current_memories)}. Preview: '{current_memories[:100].replace(os.linesep, ' ')}...'")
            memory_q.task_done()
        print("🧠 Memory manager thread finished.")

    threads = [(threading.Thread(target=f,daemon=d, name=n)) for f,d,n in [(audio_player,True,"AudioPlayerThread"), (prefetch_worker,True,"PrefetchThread"), (asr_listener,True,"ASRListenerThread"), (kb_listener,True,"KBListenerThread"), (memory_manager_worker,True,"MemoryManagerThread")]]
    for t in threads: t.start()
    print("🎙️  Talk or type anytime — 'quit' to exit, 'mem'/'dump' commands, or interrupt (e.g., 'stop').")

    try:
        while not program_is_shutting_down.is_set():
            try:
                item = user_q.get(timeout=0.5)
                current_user_input, speech_duration = item if isinstance(item, tuple) and len(item) == 2 else (str(item), 0.0)
                user_q.task_done()
            except queue.Empty:
                if program_is_shutting_down.is_set(): break
                continue

            was_interruption_flag_originally_set = interruption_requested.is_set()
            filler_audio_queued_this_turn = False # Reset for each new turn

            if was_interruption_flag_originally_set:
                print("MAIN: Interruption flag was set. Processing cleanup...")
                previous_gen_id_before_interrupt = current_generation_id # Store for logging if needed
                current_generation_id = None

                if tts_actively_playing.is_set():
                    # print(f"MAIN: TTS was active (for gen {gen_id_short(previous_gen_id_before_interrupt)}), stopping playback.", file=sys.stderr)
                    sd.stop(); tts_actively_playing.clear() # type: ignore

                # print("MAIN: Clearing audio and prefetch queues of stale items.", file=sys.stderr)
                for q_to_clear in [audio_q, prefetch_q]:
                    try:
                        while True:
                            item_in_q = q_to_clear.get_nowait()
                            item_gen_id = item_in_q[0] if isinstance(item_in_q, tuple) else None
                            if isinstance(item_in_q, tuple) and len(item_in_q) > 1 and isinstance(item_in_q[1], pathlib.Path):
                                p = item_in_q[1]
                                if p and p.exists() and not p.is_relative_to(CACHE_DIR):
                                    # print(f"MAIN_INT_CLEANUP: Unlinking {p} from GenID {gen_id_short(item_gen_id)}", file=sys.stderr)
                                    p.unlink(missing_ok=True)
                            q_to_clear.task_done()
                    except queue.Empty: pass

                if interrupt_audio_files:
                    try: sd.play(*sf.read(str(random.choice(interrupt_audio_files)), dtype='float32')); sd.wait() # type: ignore
                    except Exception as e: print(f"‼️ Could not play interrupt clip: {e}", file=sys.stderr)

                interruption_requested.clear(); print("MAIN: Interruption flag processed and cleared.")
                if any(k in current_user_input.lower().strip(".?!, ") for k in INTERRUPT_KEYWORDS):
                    print("INFO: Interrupt was a 'stop' command."); current_generation_id = None;
                    continue
                else: print("INFO: Interrupt was new content. Proceeding.")

            if current_user_input.lower() in {"quit","exit"}: print("MAIN: Quit."); program_is_shutting_down.set(); break
            if current_user_input.lower() == "mem": print(f"\n--- MEMORY ---\n{current_memories or '[none]'}\n--------------"); continue
            if current_user_input.lower() == "dump": print("\n--- CHAT HISTORY DUMP ---"); [print(f"[{i}] [{m['role']}] {m['content']}") for i, m in enumerate(chat_history)]; print("-------------------------"); continue

            current_generation_id = uuid.uuid4()
            print(f"MAIN: New turn. User: '{current_user_input[:50]}...'. GenID: {gen_id_short(current_generation_id)}")
            messages_for_llm = list(chat_history)
            system_prompt_content = PERSONA_TEMPLATE + (f"\n\n--- CONVERSATION MEMORIES ---\n{current_memories}" if current_memories else "")
            if messages_for_llm and messages_for_llm[0]['role'] == 'system': messages_for_llm[0]['content'] = system_prompt_content
            else: messages_for_llm.insert(0, {"role": "system", "content": system_prompt_content})
            messages_for_llm.append({"role": "user", "content": current_user_input})

            if was_interruption_flag_originally_set and not any(k in current_user_input.lower().strip(".?!, ") for k in INTERRUPT_KEYWORDS):
                filler_clip = None
                if speech_duration >= INTERRUPTION_DURATION_THRESHOLD_S and long_query_audio_files: filler_clip = random.choice(long_query_audio_files)
                elif pending_audio_files: filler_clip = random.choice(pending_audio_files)
                if filler_clip:
                    audio_q.put((current_generation_id, filler_clip, "[FILLER_AUDIO]"))
                    filler_audio_queued_this_turn = True # Set flag

            print(f"🤖 Assistant thinking... (Gen: {gen_id_short(current_generation_id)})\n", end='', flush=True, file=sys.stderr)
            llm_full_response = ""; text_processed_for_tts = ""; first_sentence_sent_flag = False

            sentence_ender = re.compile(
                r"""(?<!\b[A-Z][a-z]\.)(?:(?<=[.?!])|(?<=\.{2})|(?<=\.{3}))\s""", re.VERBOSE
            )

            try:
                for token_chunk in hermes_chat(messages_for_llm):
                    if interruption_requested.is_set(): print("MAIN: Interruption during LLM stream.", file=sys.stderr); break
                    sys.stderr.write(token_chunk); sys.stderr.flush(); llm_full_response += token_chunk

                    while True:
                        unprocessed_text = llm_full_response[len(text_processed_for_tts):]
                        if not unprocessed_text.strip(): break
                        words_in_unprocessed = len(unprocessed_text.split())
                        segment_to_send_to_pipeline = ""

                        if not first_sentence_sent_flag:
                            if words_in_unprocessed >= MIN_FIRST_WORDS:
                                match = sentence_ender.search(unprocessed_text)
                                if match: segment_to_send_to_pipeline = unprocessed_text[:match.end()]
                                else: break
                            else: break
                        else:
                            if words_in_unprocessed >= CHUNK_WORD_MAX:
                                last_break_position = -1
                                for m_iter in sentence_ender.finditer(unprocessed_text):
                                    last_break_position = m_iter.end()
                                if last_break_position > 0: segment_to_send_to_pipeline = unprocessed_text[:last_break_position]
                                else: break
                            else: break

                        if segment_to_send_to_pipeline:
                            text_processed_for_tts += segment_to_send_to_pipeline
                            stripped_segment = segment_to_send_to_pipeline.strip()
                            if stripped_segment:
                                individual_sentences = split_sentences(stripped_segment)
                                if individual_sentences:
                                    batched_chunks = batch_sentences(individual_sentences, CHUNK_WORD_MAX)
                                    if batched_chunks: prefetch_q.put((current_generation_id, batched_chunks))
                            if not first_sentence_sent_flag and stripped_segment: first_sentence_sent_flag = True
                        else: break

                sys.stderr.write('\n')
                if interruption_requested.is_set():
                    if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                    continue

                remaining_text = llm_full_response[len(text_processed_for_tts):].strip()
                if remaining_text:
                    should_process_remaining = True
                    if not first_sentence_sent_flag:
                        if len(remaining_text.split()) < MIN_FIRST_WORDS:
                            if filler_audio_queued_this_turn: # Check the flag for this turn
                                print(f"MAIN: Entire LLM response ('{remaining_text}') is short, AND filler audio was played. Discarding.", file=sys.stderr)
                                should_process_remaining = False
                            # else: # It's short, but NO filler was played, so we should process it.
                            #    print(f"MAIN: Entire LLM response ('{remaining_text}') is short, but NO filler played. Processing.", file=sys.stderr)

                    if should_process_remaining:
                        individual_sentences = split_sentences(remaining_text)
                        if individual_sentences:
                            batched_chunks = batch_sentences(individual_sentences, CHUNK_WORD_MAX)
                            if batched_chunks:
                                final_chunks_to_send = [chunk for chunk in batched_chunks if chunk.strip()]
                                if final_chunks_to_send:
                                    prefetch_q.put((current_generation_id, final_chunks_to_send))
            except Exception as e:
                print(f"\n‼️ Error during LLM streaming/batching: {e}", file=sys.stderr); traceback.print_exc()
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                continue

            assistant_response_for_chat_history = strip_action_and_emoji(llm_full_response)
            if assistant_response_for_chat_history:
                if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                chat_history.append({"role": "assistant", "content": assistant_response_for_chat_history})
            elif current_user_input :
                if not chat_history or chat_history[-1].get('content') != current_user_input or chat_history[-1].get('role') != 'user':
                     chat_history.append({"role": "user", "content": current_user_input})
                continue

            total_tokens = sum(rough_tokens(m['content']) for m in chat_history)
            if total_tokens > TOKEN_SOFT_LIMIT and len(chat_history) > (KEEP_RECENT * 2 + 1):
                to_summarize = chat_history[1 : -(KEEP_RECENT * 2)]
                if to_summarize:
                    transcript_for_summary = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
                    memory_q.put((current_memories, transcript_for_summary))
                    chat_history = [chat_history[0]] + chat_history[-(KEEP_RECENT * 2):]
            current_tokens_pruned = sum(rough_tokens(m['content']) for m in chat_history)
            while current_tokens_pruned > TOKEN_HARD_LIMIT and len(chat_history) > 2:
                removed = chat_history.pop(1); print(f"‼️ HARD TOKEN PRUNING: Removing '{removed['content'][:30]}...'")
                current_tokens_pruned = sum(rough_tokens(m['content']) for m in chat_history)

    except KeyboardInterrupt: print("\n🚨 Main loop: KeyboardInterrupt."); program_is_shutting_down.set()
    except Exception as e: print(f"‼️ UNHANDLED EXCEPTION IN MAIN: {e}"); traceback.print_exc(); program_is_shutting_down.set()
    finally:
        print("🛑 Main loop finished. Finalizing shutdown...")
        if not program_is_shutting_down.is_set(): program_is_shutting_down.set()
        current_generation_id = None
        print("🛑 Sending sentinels to worker queues..."); audio_q.put((None, None, None)); prefetch_q.put((None, None)); memory_q.put((None,None))
        active_threads = [t for t in threads if t.is_alive()]
        for t in active_threads:
            print(f"🛑 Waiting for {t.name} to join (5s)..."); t.join(timeout=5.0)
            if t.is_alive(): print(f"⚠️ {t.name} did not join cleanly.")
            else: print(f"✅ {t.name} joined.")
        print("🛑 Stopping any remaining audio playback..."); sd.stop() # type: ignore
        print("🧹 Performing final audio cleanup..."); cleanup_old_audio(); print("👋 Goodbye!")

if __name__=="__main__":
    main()