# state_manager.py
import threading

class StateManager:
    """A thread-safe class to manage shared state across different components."""
    def __init__(self):
        self._lock = threading.Lock()
        self._tts_is_playing = False
        self._interruption_requested = False

    def set_tts_playing(self, status: bool):
        """Set the status of TTS playback."""
        with self._lock:
            self._tts_is_playing = status

    def is_tts_playing(self) -> bool:
        """Check if TTS is currently playing."""
        with self._lock:
            return self._tts_is_playing

    def request_interruption(self) -> bool:
        """
        Request an interruption. Returns True if the request was newly set,
        False if it was already requested. This prevents runaway signals.
        """
        with self._lock:
            if not self._interruption_requested:
                self._interruption_requested = True
                return True
        return False

    def is_interruption_requested(self) -> bool:
        """Check if an interruption has been requested."""
        with self._lock:
            return self._interruption_requested

    def clear_interruption(self):
        """Clear the interruption flag. Should only be called by the main thread."""
        with self._lock:
            self._interruption_requested = False