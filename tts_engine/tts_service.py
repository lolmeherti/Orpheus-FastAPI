# tts_service.py

import logging
import queue
import threading
from datetime import datetime
import uuid

import httpx
from pydub import AudioSegment

import config
from audio_processing import preprocess_clip

class TTSService:
    def __init__(self, prefetch_q: queue.Queue, audio_q: queue.Queue, interruption_requested: threading.Event, program_is_shutting_down: threading.Event, get_current_generation_id):
        self.prefetch_q = prefetch_q
        self.audio_q = audio_q
        self.interruption_requested = interruption_requested
        self.program_is_shutting_down = program_is_shutting_down

        # This is a "getter" function passed from the main script
        # to safely access the current generation ID without globals.
        self.get_current_generation_id = get_current_generation_id

    def _make_tts_request(self, txt: str, generation_id: uuid.UUID):
        """Makes a single, blocking TTS request and returns the audio file path."""
        current_gen_id = self.get_current_generation_id()
        if current_gen_id and generation_id != current_gen_id:
            return None

        try:
            payload = {"input": txt, "model": "orpheus", "voice": config.VOICE, "response_format": "wav", "speed": 1.0}
            start_time = datetime.now().timestamp()
            with httpx.Client(timeout=config.TTS_REQUEST_TIMEOUT) as client:
                response = client.post(config.ORPHEUS_API_URL, json=payload)

            duration_ms = (datetime.now().timestamp() - start_time) * 1000
            logging.info(f"TTS_API_RECEIVE: Status {response.status_code} ({duration_ms:.0f}ms) for: '{txt[:50]}...'")
            response.raise_for_status()

            wav_data = response.content
            if len(wav_data) < 1000:
                logging.warning(f"TTS_API_INVALID_DATA for: '{txt[:50]}...'")
                return None

            file_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}_{generation_id.hex[:6]}_{uuid.uuid4().hex[:4]}.wav"
            output_path = config.OUTPUT_DIR / file_name
            output_path.write_bytes(wav_data)
            return output_path
        except Exception as e:
            logging.error(f"TTS_REQUEST_ERROR for chunk '{txt[:50]}...': {e}", exc_info=True)
            return None

    def run(self):
        """
        The main loop for the TTS service thread.
        Pulls a list of chunks and processes them SERIALLY.
        """
        logging.info("TTS_SERVICE_READY: Serial TTS worker started.")
        while not self.program_is_shutting_down.is_set():
            try:
                generation_id, chunks_with_sequence = self.prefetch_q.get(timeout=0.2)

                if chunks_with_sequence is None: break
                if generation_id != self.get_current_generation_id():
                    self.prefetch_q.task_done()
                    continue

                for text_chunk, sequence in chunks_with_sequence:
                    if self.interruption_requested.is_set() or self.program_is_shutting_down.is_set():
                        break

                    # 1. Generate audio for the chunk
                    raw_audio_path = self._make_tts_request(text_chunk, generation_id)
                    if not raw_audio_path:
                        continue

                    # 2. Preprocess the audio
                    try:
                        raw_segment = AudioSegment.from_file(raw_audio_path, format="wav")
                        processed_segment = preprocess_clip(
                            seg=raw_segment, ref_f0=config.REFERENCE_F0, silence_thresh_db=config.SILENCE_THRES_DB,
                            safe_pause_ms=config.SAFE_PAUSE_MS, edge_fade_ms=config.EDGE_FADE_MS,
                            fade_in_ms=config.FADE_IN_MS, target_lufs=config.TARGET_LUFS
                        )
                        processed_segment.export(raw_audio_path, format="wav")
                        self.audio_q.put((generation_id, sequence, raw_audio_path, text_chunk))
                    except Exception as e:
                        logging.error(f"PREPROCESS_FAIL: {raw_audio_path.name}: {e}", exc_info=True)
                        try: raw_audio_path.unlink(missing_ok=True)
                        except: pass

                self.prefetch_q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS_SERVICE_ERROR: {e}", exc_info=True)

        logging.info("TTS_SERVICE_SHUTDOWN")