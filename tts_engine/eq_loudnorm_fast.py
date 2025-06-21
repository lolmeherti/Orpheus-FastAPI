# eq_loudnorm_fast.py  –  add a mild high-shelf / de-harsh step
import numpy as np
import pyloudnorm as pyln
from pydub import AudioSegment, effects          # ← effects already imported

def conform_clip_memory(seg: AudioSegment, target_loudness: float = -23.0) -> AudioSegment:
    """Applies loudness normalization entirely in memory using pyloudnorm,
       then tames harsh highs with a gentle 10 kHz low-pass filter."""
    # 1) loudness normalization (unchanged)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    meter   = pyln.Meter(seg.frame_rate)
    loudness = meter.integrated_loudness(samples)
    normalized = pyln.normalize.loudness(samples, loudness, target_loudness)
    pcm_i16 = (normalized * 32767).astype(np.int16)

    # 2) back to AudioSegment
    seg_norm = AudioSegment(
        pcm_i16.tobytes(),
        frame_rate = seg.frame_rate,
        sample_width = pcm_i16.dtype.itemsize,
        channels = 1
    )

    # 3) gentle high-shelf cut (low-pass at 10 kHz)
    seg_smoothed = effects.low_pass_filter(seg_norm, 10_000)

    return seg_smoothed
