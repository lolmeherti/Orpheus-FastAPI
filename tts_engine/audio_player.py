# audio_player.py

import logging
import queue
import time
import threading
import traceback

import numpy as np
import pyaudio
import soundfile as sf

import config

class AudioPlayer:
    def __init__(self, audio_q: queue.Queue, tts_actively_playing: threading.Event, interruption_requested: threading.Event, program_is_shutting_down: threading.Event, last_tts_text_ref: list):
        self.audio_q = audio_q
        self.tts_actively_playing = tts_actively_playing
        self.interruption_requested = interruption_requested
        self.program_is_shutting_down = program_is_shutting_down
        self.last_tts_text_ref = last_tts_text_ref
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.active_gen_id = None
        self.current_stream_rate = None
        self.prev_tail = np.array([], dtype=np.int16)
        self.next_sequence_to_play = 0
        self.clip_buffer = {}

    def _open_stream(self, rate):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except OSError as e:
                logging.warning(f"AUDIO_PLAYER_STREAM_CLOSE_WARN: Harmless error closing previous stream: {e}")

        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True, frames_per_buffer=int(rate * 0.05)) # 50ms buffer
        self.current_stream_rate = rate

    def _play_audio_data(self, data_samples, sample_rate):
        if not self.stream or not self.stream.is_active():
            self._open_stream(sample_rate)

        fade_len_samples = int(sample_rate * config.CROSSFADE_MS / 1000)
        actual_fade_len = min(len(self.prev_tail), len(data_samples), fade_len_samples)
        if actual_fade_len > 0:
            fade_out = np.linspace(1.0, 0.0, actual_fade_len, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, actual_fade_len, dtype=np.float32)
            blended_head = (self.prev_tail[:actual_fade_len].astype(np.float32) * fade_out + data_samples[:actual_fade_len].astype(np.float32) * fade_in).astype(np.int16)
            data_to_play = np.concatenate((blended_head, data_samples[actual_fade_len:]))
        else:
            data_to_play = data_samples

        chunk_size = 1024
        for i in range(0, len(data_to_play), chunk_size):
            if self.interruption_requested.is_set():
                logging.info("AUDIO_PLAYER_PLAYBACK_INTERRUPTED: Detected interruption flag.")
                if self.stream and self.stream.is_active():
                    self.stream.stop_stream()
                return
            self.stream.write(data_to_play[i:i+chunk_size].tobytes())

        if len(data_samples) >= fade_len_samples:
            self.prev_tail = data_samples[-fade_len_samples:]
        else:
            self.prev_tail = data_samples[:]

    def _reset_playback_state(self):
        logging.info("AUDIO_PLAYER_RESET: Resetting playback state.")
        self.active_gen_id = None
        self.tts_actively_playing.clear()
        self.prev_tail = np.array([], dtype=np.int16)
        self.next_sequence_to_play = 0
        self.clip_buffer.clear()

    def _play_clip(self, audio_path_obj, text_item):
        logging.info(f"AUDIO_PLAYER_PLAY_TTS: '{text_item[:60]}...'")
        print(f"🎵 TTS: '{text_item}'")
        self.last_tts_text_ref[0] = text_item
        try:
            samples, sr = sf.read(str(audio_path_obj), dtype='int16')
            self._play_audio_data(samples, sr)
        finally:
            if audio_path_obj.exists() and not (config.CACHE_DIR.resolve() in audio_path_obj.resolve().parents):
                try: audio_path_obj.unlink(missing_ok=True)
                except Exception as e_del: logging.warning(f"AUDIO_PLAYER_DELETE_FAIL: {e_del}")

    def run(self):
        logging.info("AUDIO_PLAYER_READY: Audio player thread started.")
        while not self.program_is_shutting_down.is_set():
            try:
                if self.interruption_requested.is_set():
                    self._reset_playback_state()
                    while not self.audio_q.empty():
                        try: self.audio_q.get_nowait(); self.audio_q.task_done()
                        except queue.Empty: break
                    time.sleep(0.1)
                    continue

                gen_id, sequence, data_item, text_item = self.audio_q.get(timeout=0.2)
                if data_item is None: break

                if gen_id != self.active_gen_id:
                    self._reset_playback_state()
                    self.active_gen_id = gen_id
                    self.tts_actively_playing.set()
                    logging.info(f"AUDIO_PLAYER_NEW_GEN: Starting generation {str(gen_id)[:8]}")

                self.clip_buffer[sequence] = (data_item, text_item)

                while self.next_sequence_to_play in self.clip_buffer:
                    if self.interruption_requested.is_set(): break
                    buffered_data, buffered_text = self.clip_buffer.pop(self.next_sequence_to_play)

                    if buffered_text == "[PAUSE]":
                        time.sleep(buffered_data / 1000.0)
                    else:
                        self._play_clip(buffered_data, buffered_text)
                    self.next_sequence_to_play += 1

                self.audio_q.task_done()

            except queue.Empty:
                if not self.clip_buffer and self.tts_actively_playing.is_set():
                    self._reset_playback_state()
                continue
            except Exception as e:
                logging.error(f"AUDIO_PLAYER_ERROR: An exception occurred: {e}", exc_info=True)
                if self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except Exception as e_close:
                        logging.error(f"AUDIO_PLAYER_STREAM_CLEANUP_ERROR: {e_close}")
                self.stream = None
                self._reset_playback_state()

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except: pass
        self.pa.terminate()
        logging.info("AUDIO_PLAYER_SHUTDOWN")