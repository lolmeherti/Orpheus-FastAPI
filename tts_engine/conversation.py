#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# convo.py

import json, os, pathlib, queue, re, sys, threading, traceback, uuid, random
from datetime import datetime
from difflib import SequenceMatcher
import time

import numpy as np, requests, sounddevice as sd, soundfile as sf
import torch
import whisper

# --- NEW: Import the chunker library ---
from streaming_chunker import StreamingChunker

os.environ["NEMO_DISABLE_TQDM"] = "1"

# --- VAD & ASR Configuration ---
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

# --- Core Application Settings ---
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
ORPHEUS_API_URL    = "http://127.0.0.1:5005/v1/audio/speech"
VOICE              = "tara"
MIN_WORDS_FOR_CHUNK= 8 # Parameter for the new StreamingChunker
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

TTS_REQUEST_TIMEOUT = (10, 30)
# --- Regex for filtering/history (minimal set) ---
WORD_COUNT_THRESHOLD_FOR_TAGGED_SENTENCE_DISCARD = 2
STANDALONE_TAG_RE = re.compile(r"^\s*<[^>]+>\s*$")
ANY_BRACKETED_TAG_RE = re.compile(r"<[^>]+>")
ACTION_RE = re.compile(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]')
LLM_SPECIAL_TOKENS_RE = re.compile(r'<\|.*?\|>')


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

# --- HISTORY SANITATION: The only sanitation function left in this file ---
def strip_action_and_emoji(text_input: str) -> str:
    """Cleans the full text response before it's saved to chat history."""
    if not isinstance(text_input, str): return ""
    text = text_input
    text = ANY_BRACKETED_TAG_RE.sub('', text)
    text = ACTION_RE.sub('', text)
    text = LLM_SPECIAL_TOKENS_RE.sub('',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Worker Threads ---

def asr_listener():
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
                                    txt = asr_model.transcribe(full_audio, fp16=torch.cuda.is_available())['text'].strip()
                                    processing_lag_s = time.monotonic() - end_of_speech_time
                                    if txt:
                                        is_echo = False
                                        if tts_actively_playing.is_set() and last_tts_text:
                                            similarity = SequenceMatcher(None, txt.lower(), last_tts_text.lower()).ratio()
                                            if similarity > ECHO_SIMILARITY_THRESHOLD:
                                                print(f"🎤 Echo detected (Similarity: {similarity:.2f}), discarding.", file=sys.stderr); is_echo = True
                                        if not is_echo and len(txt.split()) >= MIN_WORDS_ASR:
                                            speech_duration_s = len(full_audio) / sr
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

def kb_listener():
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

def audio_player():
    global last_tts_text
    print("🎵 Audio player thread started.")
    player_active_gen_id = None
    while not program_is_shutting_down.is_set():
        try:
            gen_id, p_path_obj, text_that_was_spoken = audio_q.get(timeout=0.1)
        except queue.Empty:
            if not tts_actively_playing.is_set(): player_active_gen_id = None
            continue
        if p_path_obj is None: print("🎵 AudioPlayer: Shutdown sentinel received."); break
        if interruption_requested.is_set():
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                 p_path_obj.unlink(missing_ok=True)
            player_active_gen_id = None; audio_q.task_done(); continue
        if player_active_gen_id is None:
            if gen_id != current_generation_id:
                print(f"🎵 AudioPlayer: Discarding stale audio chunk. File: {p_path_obj.name if p_path_obj else 'N/A'}", file=sys.stderr)
                if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                    p_path_obj.unlink(missing_ok=True)
                audio_q.task_done(); continue
            player_active_gen_id = gen_id
        elif gen_id != player_active_gen_id:
            print(f"🎵 AudioPlayer: Gen ID mismatch. Discarding. File: {p_path_obj.name if p_path_obj else 'N/A'}", file=sys.stderr)
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                p_path_obj.unlink(missing_ok=True)
            audio_q.task_done(); continue
        is_filler_audio = (text_that_was_spoken == "[FILLER_AUDIO]")
        try:
            if not p_path_obj or not p_path_obj.exists():
                print(f"🎵 AudioPlayer: Audio file path is invalid: {p_path_obj}", file=sys.stderr)
                audio_q.task_done(); continue
            data, sr_audio = sf.read(str(p_path_obj), dtype="float32")
            last_tts_text_candidate = text_that_was_spoken
            tts_actively_playing.set()
            if is_filler_audio: print(f"🎵 Playing FILLER audio: {p_path_obj.name}")
            else: print(f"🎵 Playing TTS: '{text_that_was_spoken}' ({p_path_obj.name})")
            sd.play(data, sr_audio); sd.wait()
            if not is_filler_audio: last_tts_text = last_tts_text_candidate
        except Exception as e: print(f"‼️ Audio playback error for '{p_path_obj}': {e}", file=sys.stderr); traceback.print_exc()
        finally:
            tts_actively_playing.clear()
            if not is_filler_audio and last_tts_text == text_that_was_spoken:
                 last_tts_text = ""
            if p_path_obj and p_path_obj.exists() and not p_path_obj.is_relative_to(CACHE_DIR):
                try: p_path_obj.unlink(missing_ok=True)
                except Exception: pass
            audio_q.task_done()
    print("🎵 Audio player thread finished.")

def tts_requester_thread(text_chunk, generation_id):
    try:
        if program_is_shutting_down.is_set() or generation_id != current_generation_id: return
        audio_file = tts_request(text_chunk, generation_id)
        if audio_file:
            audio_q.put((generation_id, audio_file, text_chunk))
    except Exception as e: print(f"‼️ TTS Requester Thread error for chunk '{text_chunk[:30]}...': {e}"); traceback.print_exc()

def prefetch_worker():
    print("⏳ Prefetch worker thread started.")
    while not program_is_shutting_down.is_set():
        try:
            generation_id, chunks_to_prefetch = prefetch_q.get(timeout=0.2)
        except queue.Empty: continue
        if chunks_to_prefetch is None: print("⏳ Prefetch worker: Shutdown sentinel received."); break
        if generation_id != current_generation_id: prefetch_q.task_done(); continue

        fetch_threads = []
        for text_chunk in chunks_to_prefetch:
            if program_is_shutting_down.is_set() or interruption_requested.is_set() or generation_id != current_generation_id:
                break

            # This secondary filter remains as a simple safeguard.
            if STANDALONE_TAG_RE.fullmatch(text_chunk.strip()):
                print(f"PREFETCH_FILTER: DISCARDING standalone tag: '{text_chunk[:50]}'", file=sys.stderr)
                continue

            thread = threading.Thread(target=tts_requester_thread, args=(text_chunk, generation_id))
            thread.daemon = True; fetch_threads.append(thread); thread.start()

        for thread in fetch_threads:
            thread.join()

        prefetch_q.task_done()
    print("⏳ Prefetch worker thread finished.")

rough_tokens = lambda txt: max(1, len(txt)//4)

def summarise(prev_summary, new_transcript):
    if program_is_shutting_down.is_set(): return prev_summary or "[summary skipped]"
    try:
        content = f"PREVIOUS SUMMARY:\n{prev_summary or 'None.'}\n\nNEW TRANSCRIPT TO ADD:\n{new_transcript}"
        r = requests.post(LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}], "temperature": 0.6, "max_tokens": 1024}, timeout=200)
        r.raise_for_status(); summary_text = r.json()['choices'][0]['message']['content'].strip()
        if summary_text.lower().startswith("updated summary:"): summary_text = summary_text.split(":", 1)[1].strip()
        return summary_text
    except Exception as e: print(f"‼️ Summary failed: {e}", file=sys.stderr); return prev_summary or "[summary failed]"

def hermes_chat(msgs):
    llm_response_stream = None
    try:
        llm_response_stream = requests.post(LM_STUDIO_CHAT_URL, json={"model":"hermes","messages":msgs,"temperature":0.9,"stream":True}, stream=True,timeout=(10,60))
        llm_response_stream.raise_for_status()
        buffer = ""
        for chunk in llm_response_stream.iter_content(chunk_size=None, decode_unicode=True):
            if program_is_shutting_down.is_set() or interruption_requested.is_set():
                print("\n🚫 LLM stream interrupted.", file=sys.stderr); break
            buffer += chunk
            while 'data: ' in buffer and '\n' in buffer:
                event_start = buffer.find('data: ')
                line_end = buffer.find('\n', event_start)
                if line_end == -1: break
                line = buffer[event_start:line_end]
                try:
                    data = line.split('data: ', 1)[1].strip()
                    if data == '[DONE]':
                        buffer = buffer[line_end+1:]; return
                    if data:
                        delta = json.loads(data)['choices'][0]['delta']
                        if 'content' in delta:
                            yield delta['content']
                except (json.JSONDecodeError, IndexError):
                    pass # Incomplete data, wait for more
                buffer = buffer[line_end+1:]
    except requests.exceptions.RequestException as e: print(f"‼️ Hermes network error: {e}", file=sys.stderr)
    except Exception as e: print(f"‼️ Unhandled Hermes error: {e}", file=sys.stderr); traceback.print_exc()
    finally:
        if llm_response_stream:
            try: llm_response_stream.close()
            except Exception: pass

def tts_request(txt, generation_id):
    if program_is_shutting_down.is_set() or (generation_id is not None and generation_id != current_generation_id):
        return None

    # The 'txt' payload is now received pre-sanitized from the chunker.
    if not txt:
        return None

    try:
        wav_data = requests.post(ORPHEUS_API_URL,json={"input": txt,"model":"orpheus","voice":VOICE,"response_format":"wav","speed":1.0},timeout=TTS_REQUEST_TIMEOUT).content
        if len(wav_data) < 1000:
             print(f"🎤 TTS: Short/Invalid WAV received for '{txt[:30]}...'. Skipping.", file=sys.stderr); return None
        file_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+".wav"; output_path = OUTPUT_DIR/file_name
        output_path.write_bytes(wav_data); return output_path
    except requests.exceptions.RequestException as e:
        print(f"‼️ TTS API failed for '{txt[:30]}...': {e}", file=sys.stderr); return None
    except Exception as e:
        print(f"‼️ TTS error for '{txt[:30]}...': {e}", file=sys.stderr); traceback.print_exc(); return None

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
    if count > 0: print(f"🧹 Removed {count} old .wav file(s).")

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
            if prev_mems is None and transcript_to_add is None: print("🧠 Memory manager shutdown.");break
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
                current_user_input, speech_duration = user_q.get(timeout=0.2)
                user_q.task_done()
            except queue.Empty:
                if program_is_shutting_down.is_set(): break
                continue

            was_an_interruption = interruption_requested.is_set()

            if was_an_interruption:
                print("MAIN: Interruption detected. Cleaning up...")
                sd.stop(); tts_actively_playing.clear()
                for q in [audio_q, prefetch_q]:
                    while not q.empty():
                        try: q.get_nowait(); q.task_done()
                        except Exception: pass

                if interrupt_audio_files:
                    try:
                        clip_to_play = random.choice(interrupt_audio_files)
                        data_int, sr_int = sf.read(str(clip_to_play), dtype='float32')
                        sd.play(data_int, sr_int); sd.wait()
                    except Exception as e: print(f"‼️ Could not play interrupt clip: {e}", file=sys.stderr)

                interruption_requested.clear()
                print("MAIN: Interruption processed.")

                is_stop_command = any(keyword in current_user_input.lower().strip(".?!, ") for keyword in INTERRUPT_KEYWORDS)
                if is_stop_command:
                    print("INFO: Stop command received. Awaiting next input.")
                    current_generation_id = None
                    continue
                else:
                    print("INFO: Barge-in with new prompt. Proceeding.")

            current_generation_id = uuid.uuid4()

            if not was_an_interruption and speech_duration > 0.1:
                filler_clip_to_play = None
                if speech_duration > INTERRUPTION_DURATION_THRESHOLD_S and long_query_audio_files:
                    filler_clip_to_play = random.choice(long_query_audio_files)
                elif pending_audio_files:
                    filler_clip_to_play = random.choice(pending_audio_files)
                if filler_clip_to_play:
                    audio_q.put((current_generation_id, filler_clip_to_play, "[FILLER_AUDIO]"))

            if current_user_input.lower() in {"quit","exit"}: program_is_shutting_down.set(); break
            if current_user_input.lower() == "mem": print(f"\n--- MEMORY ---\n{current_memories or '[none]'}\n---"); continue
            if current_user_input.lower() == "dump": [print(f"[{m['role']}] {m['content']}") for m in chat_history]; print("---"); continue

            messages_for_llm = list(chat_history)
            system_prompt = PERSONA_TEMPLATE
            if current_memories:
                system_prompt += f"\n\n--- CONVERSATION MEMORIES ---\n{current_memories}"
            messages_for_llm[0] = {"role": "system", "content": system_prompt}
            messages_for_llm.append({"role": "user", "content": current_user_input})

            print(f"🤖 Assistant thinking... (Gen: {current_generation_id})", end='', flush=True, file=sys.stderr)

            llm_full_response = ""
            try:
                chunker_instance = StreamingChunker(min_words=MIN_WORDS_FOR_CHUNK)

                def token_tee(token_iterator):
                    nonlocal llm_full_response
                    for token in token_iterator:
                        llm_full_response += token
                        yield token
                        print(token, end='', flush=True, file=sys.stderr)

                token_stream = token_tee(hermes_chat(messages_for_llm))

                for i, sanitized_chunk in enumerate(chunker_instance.chunker(token_stream)):
                    if i == 0: print(file=sys.stderr) # Newline after "thinking..."
                    if interruption_requested.is_set(): break

                    prefetch_q.put((current_generation_id, [sanitized_chunk]))

                if interruption_requested.is_set():
                    if current_user_input: chat_history.append({"role": "user", "content": current_user_input})
                    continue

            except Exception as e:
                print(f"\n‼️ Error during LLM streaming/chunking: {e}", file=sys.stderr); traceback.print_exc()
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
                messages_to_retain = KEEP_RECENT * 2
                if len(chat_history) > messages_to_retain + 1:
                    messages_to_summarize = chat_history[1:-(messages_to_retain)]
                    if messages_to_summarize:
                        transcript_to_summarize = "\n".join(f"{m['role']}: {m['content']}" for m in messages_to_summarize)
                        print(f"🧠 Pruning chat. Summarizing {len(messages_to_summarize)} messages.")
                        memory_q.put((current_memories, transcript_to_summarize))
                        chat_history = [chat_history[0]] + chat_history[-(messages_to_retain):]

            while sum(rough_tokens(m['content']) for m in chat_history) > TOKEN_HARD_LIMIT:
                if len(chat_history) > 2:
                    print(f"‼️ HARD TOKEN PRUNING: Removing '{chat_history[1]['content'][:30]}...'")
                    del chat_history[1]
                else: break

    except KeyboardInterrupt: print("\n🚨 Main loop interrupted by user.");
    except Exception as e: print(f"‼️ UNHANDLED EXCEPTION IN MAIN: {e}"); traceback.print_exc()
    finally:
        print("🛑 Shutting down...")
        if not program_is_shutting_down.is_set(): program_is_shutting_down.set()
        audio_q.put((None, None, None)); prefetch_q.put((None, None)); memory_q.put((None,None))
        active_threads = [t for t in threads if t.is_alive()]
        for t in active_threads:
            t.join(timeout=2.0)
            if t.is_alive(): print(f"⚠️ {t.name} did not join cleanly.")
        print("🧹 Final cleanup..."); cleanup_old_audio(); print("👋 Goodbye!")

if __name__=="__main__":
    main()