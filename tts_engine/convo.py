#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# convo.py

import json, os, queue, re, sys, threading, traceback, uuid, random, time, logging, pathlib
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import requests
import torch
import httpx

import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

import config

import web_tools

import search_logger

from streaming_chunker import StreamingChunker
from llm_classifier import LLMStyleClassifier
from audio_player import AudioPlayer
from voice_listener import VoiceListener
from tts_service import TTSService

audio_q, prefetch_q, user_q, memory_q = (queue.Queue() for _ in range(4))
interruption_requested, tts_actively_playing, program_is_shutting_down = threading.Event(), threading.Event(), threading.Event()
last_tts_text_ref = [""]
current_generation_id = None

pending_search = {"query": None, "original_input": None}
CONFIRMATION_KEYWORDS = {"yes", "yep", "yeah", "that's right", "correct", "indeed", "go ahead", "do it", "sure"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)-15s] %(message)s', datefmt='%H:%M:%S')
logging.getLogger("http.client").setLevel(logging.WARNING)
llm_classifier_instance = None

def strip_action_and_emoji(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"<[^>]+>", '', text)
    text = re.sub(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]', '', text)
    text = re.sub(r'<\|.*?\|>', '', text)
    return " ".join(text.split())

def rough_tokens(txt: str) -> int: return max(1, len(txt) // 4)

def kb_listener():
    logging.info("KB_LISTENER_READY: Keyboard listener started.")
    while not program_is_shutting_down.is_set():
        try:
            line = input("\n⌨️  ").strip()
            if program_is_shutting_down.is_set(): break
            if line:
                logging.info(f"USER_KB_INPUT: '{line}'")
                if tts_actively_playing.is_set() and line.lower() in config.INTERRUPT_KEYWORDS:
                    interruption_requested.set()
                user_q.put((line, 0.0))
        except (EOFError, KeyboardInterrupt): break
        except Exception as e:
            if not program_is_shutting_down.is_set(): logging.error(f"KB_LISTENER_ERROR: {e}", exc_info=True)
    logging.info("KB_LISTENER_SHUTDOWN")

def cleanup_old_audio():
    logging.info("CLEANUP_AUDIO_START: Cleaning up old audio files...")
    cleaned_count = 0
    for folder_to_clean in set([config.OUTPUT_DIR]):
        if not folder_to_clean.is_dir(): continue
        for f_path in folder_to_clean.glob("*.wav"):
            try:
                if config.CACHE_DIR.resolve() in f_path.resolve().parents: continue
                f_path.unlink(missing_ok=True)
                cleaned_count +=1
            except Exception as e: logging.warning(f"CLEANUP_AUDIO_FAIL: Could not delete {f_path}: {e}")
    logging.info(f"CLEANUP_AUDIO_DONE: Deleted {cleaned_count} file(s).")

def hermes_chat(msgs):
    try:
        buffer = ""
        with httpx.stream("POST",
                         config.LM_STUDIO_CHAT_URL,
                         json={"model": "grok-3-reasoning-gemma3-12b-distilled", "messages": msgs, "temperature": 0.6, "stream": True},
                         timeout=httpx.Timeout(10.0, read=60.0)) as r:
            r.raise_for_status()
            for chunk in r.iter_text():
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
                            if 'content' in delta and delta['content'] is not None: yield delta['content']
                    except (json.JSONDecodeError, IndexError, KeyError): pass
    except Exception as e:
        if not program_is_shutting_down.is_set(): logging.error(f"LLM_STREAM_ERROR: {e}", exc_info=True)
def summarise(prev_summary, new_transcript):
    try:
        content = f"PREVIOUS SUMMARY:\n{prev_summary or 'None.'}\n\nNEW TRANSCRIPT TO ADD:\n{new_transcript}"
        if not SUMMARY_PROMPT: return prev_summary or "[summary failed: no prompt]"
        r = requests.post(config.LM_STUDIO_CHAT_URL, json={"model": "hermes", "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}], "temperature": 0.6, "max_tokens": 1024}, timeout=200)
        r.raise_for_status()
        summary_text = r.json()['choices'][0]['message']['content'].strip()
        return summary_text.replace("Updated Summary:", "").strip()
    except Exception as e:
        logging.error(f"SUMMARY_FAIL: {e}", exc_info=True)
        return prev_summary or "[summary failed: exception]"

def play_startup_greeting_thread(audio_path_obj):
    try:
        data_audio, sr_audio = sf.read(audio_path_obj, dtype='float32')
        sd.play(data_audio, sr_audio)
        sd.wait()
    except Exception as e: logging.warning(f"GREETING_PLAY_ERROR: '{audio_path_obj.name}': {e}", exc_info=True)

def warmup_tts():
    try:
        payload = {"input": "The system is ready.", "model": "orpheus", "voice": config.VOICE, "response_format": "wav", "speed": 1.0}
        requests.post(config.ORPHEUS_API_URL, json=payload, timeout=(5,10)).raise_for_status()
        logging.info("TTS_WARMUP: Successful.")
    except Exception as e: logging.warning(f"TTS_WARMUP_FAIL: {e}")

def main():
    global current_generation_id, personas, SUMMARY_PROMPT, llm_classifier_instance
    logging.info("MAIN_INIT: Application starting...")
    cleanup_old_audio()
    interruption_requested.clear(); tts_actively_playing.clear(); program_is_shutting_down.clear()
    chat_history=[{"role":"system","content":""}]; current_memories = ""

    def get_current_generation_id():
        return current_generation_id

    try:
        personas = {
            'default': Path(config.PERSONA_PROMPT_TEMPLATE).read_text(encoding="utf-8"),
            'summarizer': Path(config.SCRAPE_SUMMARY_BOT_TEMPLATE).read_text(encoding="utf-8"),
        }
        SUMMARY_PROMPT = Path(config.SUMMARY_BOT_TEMPLATE).read_text(encoding="utf-8")
        llm_classifier_instance = None
    except Exception as e:
        logging.critical(f"MAIN_FATAL_INIT: Could not load persona files: {e}", exc_info=True)
        sys.exit(1)

    chat_history[0]["content"] = personas['default']

    if config.CACHE_DIR.is_dir():
        interrupt_audio_files = list(config.CACHE_DIR.glob("*_interrupt_cache.wav"))
        greeting_files = list(config.CACHE_DIR.glob("*_greeting_cache.wav"))
        if greeting_files:
            greeting_thread = threading.Thread(target=play_startup_greeting_thread, args=(random.choice(greeting_files),), daemon=True, name="GreetingPlayer")
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
            except Exception as e: logging.error(f"MEMORY_WORKER_ERROR: {e}", exc_info=True)
    audio_player = AudioPlayer(audio_q, tts_actively_playing, interruption_requested, program_is_shutting_down, last_tts_text_ref)
    voice_listener = VoiceListener(user_q, tts_actively_playing, interruption_requested, last_tts_text_ref, program_is_shutting_down)
    tts_service = TTSService(prefetch_q, audio_q, interruption_requested, program_is_shutting_down, get_current_generation_id)
    threads = {
        "AudioPlayer": threading.Thread(target=audio_player.run, daemon=True),
        "TTSService": threading.Thread(target=tts_service.run, daemon=True),
        "VoiceListener": threading.Thread(target=voice_listener.run, daemon=True),
        "KBListener": threading.Thread(target=kb_listener, daemon=True),
        "MemoryManager": threading.Thread(target=memory_manager_worker, daemon=True),
    }
    for name, thread in threads.items():
        thread.name = name
        thread.start()
        logging.info(f"MAIN_THREAD_START: Started {name}")
    warmup_tts()
    print("🎙️ System ready. Talk or type anytime. Check logs for detailed info.")

    try:
        while not program_is_shutting_down.is_set():
            if interruption_requested.is_set():
                logging.info("MAIN_INTERRUPT_HANDLING_START")
                interruption_requested.clear()

                tts_actively_playing.clear()

                for q in [audio_q, prefetch_q]:
                    while not q.empty():
                        try:
                            q.get_nowait()
                            q.task_done()
                        except queue.Empty:
                            break
                logging.info("MAIN_INTERRUPT: Audio queues cleared.")

                if 'interrupt_audio_files' in locals() and interrupt_audio_files:
                    try:
                        data_int, sr_int = sf.read(random.choice(interrupt_audio_files), dtype='float32')
                        sd.play(data_int, sr_int)
                        sd.wait()
                    except Exception as e:
                        logging.warning(f"MAIN_INTERRUPT_SOUND_FAIL: {e}")

                if not user_q.empty():
                    try:
                        interruption_input, _ = user_q.get_nowait()
                        user_q.task_done()
                        logging.info(f"MAIN_INTERRUPT: Discarding user input that caused interruption: '{interruption_input}'")
                    except queue.Empty:
                        pass

                logging.info("MAIN_INTERRUPT_HANDLING_DONE: Ready for new input.")
                continue

            try:
                current_user_input, _ = user_q.get(timeout=0.2)
                user_q.task_done()
            except queue.Empty:
                if program_is_shutting_down.is_set(): break
                continue

            turn_start_time = time.monotonic()
            logging.info(f"MAIN_TURN_START: Processing: '{current_user_input[:100]}...'")

            if current_user_input.lower() in {"quit", "exit"}: break
            if current_user_input.lower() == "mem": print(f"\n--- MEMORY ---\n{current_memories or '[none]'}\n---"); continue
            if current_user_input.lower() == "dump": print("\n--- CHAT DUMP ---"); [print(f"[{m['role']}] {m['content']}") for m in chat_history]; print("---"); continue

            messages_for_llm = []
            tool_was_used = False

            if pending_search["query"]:
                user_response_lower = current_user_input.lower().strip().rstrip('.!')
                if user_response_lower in CONFIRMATION_KEYWORDS:
                    logging.info(f"MAIN_CONFIRMATION: User confirmed search. Executing.")
                    messages_for_llm = web_tools.execute_search_and_summarize(
                        user_input=pending_search["original_input"],
                        search_query=pending_search["query"],
                        personas=personas
                    )
                    tool_was_used = True
                    # We only save the user part of the history here
                    chat_history.append({"role": "user", "content": pending_search["original_input"]})
                else:
                    logging.info(f"MAIN_CONFIRMATION: User denied search. Cancelling.")
                    messages_for_llm = [{"role": "system", "content": personas['default']}, {"role": "user", "content": "Acknowledge that you have cancelled the requested action and ask what they would like to do instead."}]
                    chat_history.append({"role": "user", "content": pending_search["original_input"]})
                    # Since the search was denied, reset the state immediately
                    pending_search["query"] = None
                    pending_search["original_input"] = None

            # STATE 2: If not waiting, did the user just ask to start a new search?
            else:
                proposed_query = web_tools.check_for_search_keyword(current_user_input)
                if proposed_query:
                    # A keyword was found. Set the state and ask for confirmation.
                    pending_search["query"] = proposed_query
                    pending_search["original_input"] = current_user_input

                    confirmation_question = f"I think you want me to search for: \"{proposed_query}\". Is that correct?"

                    # Manually generate the confirmation response without calling the main LLM
                    current_generation_id = uuid.uuid4()
                    prefetch_q.put((current_generation_id, [(confirmation_question, 0)]))
                    print(f"🤖 Assistant: {confirmation_question}")
                    logging.info(f"MAIN_AWAIT_CONFIRM: Waiting for user confirmation for query: '{proposed_query}'")
                    continue # End the turn here and wait for the user's "yes" or "no"

                # STATE 3: If no search is pending or proposed, it's a normal conversation.
                else:
                    system_prompt = personas['default'] + (f"\n\n--- SUMMARY ---\n{current_memories}" if current_memories else "")
                    messages_for_llm = [{"role": "system", "content": system_prompt}] + chat_history[1:] + [{"role": "user", "content": current_user_input}]
                    # Save user's turn to history for normal chat
                    chat_history.append({"role": "user", "content": current_user_input})

            # --- LLM Processing and Response Generation ---

            # This block now only runs if there is something for the LLM to say
            current_generation_id = uuid.uuid4()
            logging.info(f"MAIN_NEW_GEN_ID: {current_generation_id.hex[:8]}")

            llm_full_response = ""
            try:
                chunker = StreamingChunker(min_words=config.MIN_WORDS_FOR_CHUNK)
                token_stream = hermes_chat(messages_for_llm)
                chunks_with_sequence = [(chunk, i) for i, chunk in enumerate(chunker.chunker(token_stream))]
                if chunks_with_sequence:
                    prefetch_q.put((current_generation_id, chunks_with_sequence))
                    llm_full_response = " ".join(c[0] for c in chunks_with_sequence)
            except Exception as e:
                logging.error(f"MAIN_LLM_STREAM_ERROR: {e}", exc_info=True)
                continue

            assistant_response = strip_action_and_emoji(llm_full_response).strip()
            if not assistant_response:
                continue

            log_prefix = "ASSISTANT (from tool):" if tool_was_used else "ASSISTANT:"
            logging.info(f"{log_prefix} '{assistant_response[:100]}...'")
            print(f"🤖 Assistant: {assistant_response}")

            # Save the final assistant response to history
            chat_history.append({"role": "assistant", "content": assistant_response})

            if tool_was_used:
                search_logger.log_search(
                    original_question=pending_search["original_input"],
                    executed_query=pending_search["query"],
                    summary_answer=assistant_response
                )
                pending_search["query"] = None
                pending_search["original_input"] = None

            total_tokens = sum(rough_tokens(m['content']) for m in chat_history)
            if total_tokens > config.TOKEN_SOFT_LIMIT:
                to_retain_count = config.KEEP_RECENT * 2
                if len(chat_history) > to_retain_count + 1:
                    to_summarize = chat_history[1:-to_retain_count]
                    if to_summarize:
                        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
                        memory_q.put((current_memories, transcript))
                        chat_history = [chat_history[0]] + chat_history[-to_retain_count:]
            while sum(rough_tokens(m['content']) for m in chat_history) > config.TOKEN_HARD_LIMIT and len(chat_history) > 2:
                del chat_history[1]

            logging.info(f"MAIN_TURN_END: Duration: {(time.monotonic() - turn_start_time) * 1000:.0f}ms.")

    except (KeyboardInterrupt, EOFError):
        logging.info("\nMAIN_INTERRUPT: Shutting down...")
    finally:
        logging.info("MAIN_FINALIZING: Initiating shutdown...")
        program_is_shutting_down.set()
        audio_q.put((None, None, None, None))
        prefetch_q.put((None, None))
        memory_q.put((None, None))
        for name, t in threads.items():
            t.join(timeout=2.0)
            if t.is_alive(): logging.warning(f"MAIN_FINALIZING_THREAD_ALIVE: {name} did not shut down cleanly.")
        cleanup_old_audio()
        logging.info("MAIN_SHUTDOWN_COMPLETE: Application has shut down.")
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    personas = {}
    SUMMARY_PROMPT = ""
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"MAIN_FATAL_PROMPT_MISSING: {e}. Please check paths in config.py.", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logging.critical(f"MAIN_FATAL_STARTUP_ERROR: {e}", exc_info=True)
        sys.exit(1)