# file: test_audio_stream_crossfade.py  (with NeMo pitch‑match)
import time
from pathlib import Path
import numpy as np
import pyaudio
from pydub import AudioSegment
from pydub.effects import strip_silence
from eq_loudnorm import loudnorm_stats, conform_clip
from prosody import match_pitch   # NEW

CLIP_DIR          = Path("cached_clips")
TMP_DIR           = Path("eq_cache")
TMP_DIR.mkdir(exist_ok=True)

CROSSFADE_MS      = 200
CHUNK_MS          = 20
TRIM_THRESHOLD_DB = -50
FADE_IN_MS        = 100
PADDING_MS        = 0

REF_CLIP_PATH: Path | None = None

def preprocess_clip(path: Path) -> Path:
    pitched = match_pitch(REF_CLIP_PATH, path, TMP_DIR) if REF_CLIP_PATH else path
    stats   = loudnorm_stats(pitched)
    return conform_clip(pitched, float(stats["input_i"]), float(stats["input_tp"]), TMP_DIR)


def load_clean(path: Path) -> AudioSegment:
    processed_path = preprocess_clip(path)
    seg = AudioSegment.from_wav(processed_path)
    seg = strip_silence(seg, silence_thresh=TRIM_THRESHOLD_DB, padding=PADDING_MS)
    seg = seg.fade_in(FADE_IN_MS).fade_out(60)
    return seg


def seg_to_np(seg: AudioSegment) -> np.ndarray:
    return np.array(seg.get_array_of_samples()).astype(np.int16)


def to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.int16).tobytes()


def stream_crossfade_tail(a: AudioSegment, b: AudioSegment):
    sr            = a.frame_rate
    chunk_samples = int(sr * CHUNK_MS / 1000)
    fade_samples  = int(sr * CROSSFADE_MS / 1000)

    pa  = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, output=True,
                  frames_per_buffer=chunk_samples)

    a_np = seg_to_np(a)
    b_np = seg_to_np(b)

    lead_len = len(a_np) - fade_samples
    pos = 0
    while pos < lead_len:
        out.write(to_bytes(a_np[pos:pos + chunk_samples]))
        pos += chunk_samples

    a_tail = a_np[lead_len:]
    b_head = b_np[:fade_samples]
    fade   = np.linspace(0, 1, fade_samples, dtype=np.float32)
    mixed  = ((1 - fade) * a_tail + fade * b_head).astype(np.int16)
    out.write(to_bytes(mixed))

    tail_b = b_np[fade_samples:]
    p = 0
    while p < len(tail_b):
        out.write(to_bytes(tail_b[p:p + chunk_samples]))
        p += chunk_samples

    out.stop_stream(); out.close(); pa.terminate()


def main():
    global REF_CLIP_PATH
    clips = sorted(CLIP_DIR.glob("*.wav"))
    if len(clips) < 2:
        print("Need ≥ 2 clips."); return

    REF_CLIP_PATH = clips[0]

    first = load_clean(clips[0])
    next_ = load_clean(clips[1])

    print(f"▶️ Playing {clips[0].name} → cross‑fading into {clips[1].name} (pitch‑matched) over {CROSSFADE_MS} ms")
    stream_crossfade_tail(first, next_)


if __name__ == "__main__":
    main()
