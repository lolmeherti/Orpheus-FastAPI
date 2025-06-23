# voice_listener.py

import logging
import queue
import threading
from difflib import SequenceMatcher
import time

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

import config

class VoiceListener:
    def __init__(self, user_q: queue.Queue, tts_actively_playing: threading.Event, interruption_requested: threading.Event, last_tts_text_ref: list, program_is_shutting_down: threading.Event):
        self.user_q = user_q
        self.tts_actively_playing = tts_actively_playing
        self.interruption_requested = interruption_requested
        self.last_tts_text_ref = last_tts_text_ref
        self.program_is_shutting_down = program_is_shutting_down
        self.asr_model = None
        self.vad_model = None
        self.sr = 16000
        self.vad_chunk_size = 512

    def _load_models(self):
        try:
            if torch.cuda.is_available(): device = "cuda"; compute_type = "float16"
            else: device = "cpu"; compute_type = "int8"
            logging.info(f"ASR_LOAD_START: Loading ASR model: small.en ({device}, {compute_type})")
            self.asr_model = WhisperModel("small.en", device=device, compute_type=compute_type)
            logging.info("ASR_LOAD_DONE: ASR model loaded.")
            logging.info("VAD_LOAD_START: Loading VAD model...")
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
            logging.info("VAD_LOAD_DONE: VAD model loaded.")
            return True
        except Exception as e:
            logging.critical(f"ASR_VAD_LOAD_FATAL: {e}", exc_info=True)
            self.program_is_shutting_down.set()
            return False

    def run(self):
        if not self._load_models(): return

        is_speaking, audio_buffer, silence_counter, speech_start_time = False, [], 0, 0
        logging.info("ASR_LISTENER_READY: Waiting for speech...")

        try:
            with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', blocksize=self.vad_chunk_size) as stream:
                while not self.program_is_shutting_down.is_set():
                    frame, overflowed = stream.read(self.vad_chunk_size)
                    if overflowed: logging.warning("ASR_OVERFLOW: Input stream overflowed.")

                    is_speech = self.vad_model(torch.from_numpy(frame.flatten()), self.sr).item() > config.VAD_SPEECH_CONFIDENCE_THRESHOLD

                    if is_speaking:
                        audio_buffer.append(frame)
                        if not is_speech:
                            silence_counter += 1

                            active_timeout_s = config.VAD_DEFAULT_SILENCE_TIMEOUT_S
                            if self.tts_actively_playing.is_set():
                                current_speech_duration_s = time.monotonic() - speech_start_time

                                if current_speech_duration_s < config.VAD_LONG_PROMPT_TRIGGER_S:
                                    active_timeout_s = config.VAD_INTERRUPT_TIMEOUT_S
                                else:
                                    active_timeout_s = config.VAD_LONG_PROMPT_TIMEOUT_S

                            if silence_counter * (self.vad_chunk_size / self.sr) >= active_timeout_s:
                                was_interruption = self.tts_actively_playing.is_set()
                                is_speaking = False

                                if was_interruption:
                                    logging.info("VAD_INTERRUPT: User spoke while TTS was active.")
                                    self.interruption_requested.set()
                                    time.sleep(0.2)

                                self._process_audio_buffer(audio_buffer)
                                audio_buffer = []
                                silence_counter = 0
                        else:
                            silence_counter = 0

                    elif is_speech:
                        is_speaking = True
                        silence_counter = 0
                        speech_start_time = time.monotonic()
                        if self.tts_actively_playing.is_set():
                             self.interruption_requested.set()

                        audio_buffer = [frame]
        except Exception as e:
            if not self.program_is_shutting_down.is_set():
                logging.error(f"ASR_LISTENER_ERROR: {e}", exc_info=True)
        finally:
            logging.info("ASR_LISTENER_SHUTDOWN")

    def _process_audio_buffer(self, audio_buffer):
        full_audio_np = np.concatenate(audio_buffer).squeeze()
        effective_audio_duration_s = len(full_audio_np) / self.sr
        if effective_audio_duration_s < config.VAD_MIN_SPEECH_S: return

        segments, _ = self.asr_model.transcribe(full_audio_np, language="en", beam_size=1)
        txt = " ".join(segment.text for segment in segments).strip()
        if not txt: return

        is_echo = SequenceMatcher(None, txt.lower(), self.last_tts_text_ref[0].lower()).ratio() > config.ECHO_SIMILARITY_THRESHOLD
        if is_echo:
            logging.info(f"ASR_ECHO_SUPPRESSED: '{txt}'")
            return

        if len(txt.split()) < config.MIN_WORDS_ASR:
            logging.info(f"ASR_TOO_SHORT: Discarded: '{txt}'")
            return

        logging.info(f"USER_SPEECH: '{txt}' (Audio: {effective_audio_duration_s:.2f}s)")
        print(f"\n🗣️ User: '{txt}' (Audio duration: {effective_audio_duration_s:.2f}s)")

        normalized_input = txt.lower().strip(".?!, ")
        if any(keyword in normalized_input for keyword in config.INTERRUPT_KEYWORDS):
            if not self.interruption_requested.is_set():
                logging.info("VAD_INTERRUPT: Keyword detected.")
                self.interruption_requested.set()

        self.user_q.put((txt, effective_audio_duration_s))