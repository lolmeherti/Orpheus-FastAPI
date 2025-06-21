# file: audio_processing.py

from pydub import AudioSegment
from pydub.effects import strip_silence

# Import your other custom libraries
from prosody_fast import match_pitch
# Assuming you have this file from our previous work
from eq_loudnorm_fast import conform_clip_memory

def preprocess_clip(
    seg: AudioSegment,
    ref_f0: float,
    silence_thresh_db: int,
    safe_pause_ms: int,
    edge_fade_ms: int,
    fade_in_ms: int,
    target_lufs: float
) -> AudioSegment:
    """
    A direct implementation of the preprocessing pipeline from test_audio_stream.py.
    Processes a single AudioSegment object according to the provided parameters.
    """
    # 1) Detect leading silence
    try:
        # A short clip of pure silence can cause an error here
        trim_seg = strip_silence(seg, silence_thresh=silence_thresh_db, padding=0)
        lead_sil_ms = len(seg) - len(trim_seg)
    except Exception:
        trim_seg = seg
        lead_sil_ms = 0

    # 2) Decide how much to keep
    keep_ms = min(lead_sil_ms, safe_pause_ms)
    seg_kept = seg[:keep_ms] + trim_seg

    # 3) Tiny edge fades to prevent clicks
    seg_kept = seg_kept.fade_in(edge_fade_ms).fade_out(edge_fade_ms)

    # 4) Artistic fade-in for breathy starts
    seg_kept = seg_kept.fade_in(fade_in_ms)

    # 5) Pitch & loudness
    if ref_f0 > 0:
        seg_kept = match_pitch(ref_f0, seg_kept)
    seg_kept = conform_clip_memory(seg_kept, target_lufs)

    return seg_kept