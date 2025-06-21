#filler_queue_builder.py

import random
from pathlib import Path

class FillerQueueBuilder:
    def __init__(self, atomic_dir, mid_dir):
        self.atomic_dir = Path(atomic_dir)
        self.mid_dir = Path(mid_dir)
        self.atomic_files = self._load_files(self.atomic_dir)
        self.mid_files = self._load_files(self.mid_dir)
        self.last_atomic = None

    def _load_files(self, directory):
        return [f for f in directory.glob("*.wav") if f.is_file()]

    def _pick_atomic(self):
        candidates = [f for f in self.atomic_files if f != self.last_atomic] or self.atomic_files
        choice = random.choice(candidates)
        self.last_atomic = choice
        return choice

    def _pick_mid(self):
        return random.choice(self.mid_files)

    def build_queue_items(self):
        mode = random.choices(["atomic_pair", "atomic_plus_mid"], weights=[0.5, 0.5])[0]
        pause_ms = random.randint(150, 300)

        if mode == "atomic_pair":
            a1 = self._pick_atomic()
            a2 = self._pick_atomic()
            return [a1, pause_ms, a2]
        elif mode == "atomic_plus_mid":
            atomic = self._pick_atomic()
            mid = self._pick_mid()
            return [atomic, pause_ms, mid]

# Example Usage:
# from filler_queue_builder import FillerQueueBuilder
# builder = FillerQueueBuilder("../cached_clips/atomic_fillers", "../cached_clips/mid_fillers")
# queue_items = builder.build_queue_items()
# for item in queue_items:
#     if isinstance(item, Path):
#         audio_q.put((generation_id, item, "[FILLER_TIER_1]"))
#     elif isinstance(item, int):
#         time.sleep(item / 1000)
