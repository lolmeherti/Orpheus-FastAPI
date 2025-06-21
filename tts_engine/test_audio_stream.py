# file: test_audio_stream_crossfade.py
import math, shutil, numpy as np, pyaudio
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import strip_silence
from eq_loudnorm_fast import conform_clip_memory
from prosody_fast import match_pitch, get_f0

# ---------- config ----------
CLIP_DIR         = Path("cached_clips")
CROSSFADE_MS     = 150          # overlap duration
CHUNK_MS         = 20           # PyAudio buffer
SILENCE_THRES_DB = -50
FADE_IN_MS       = 100
TARGET_LUFS      = -27.2
SAFE_PAUSE_MS   = 250     # keep up to 0.25 s of intentional pause
EDGE_FADE_MS    = 5       # tiny click-killer fade

# ---------- reference pitch ----------
ref_clip = AudioSegment.from_file(CLIP_DIR / "alright_i_hear_you_pending_cache.wav", "wav")
REF_F0   = get_f0(ref_clip)
print(f"🎯 reference F0 = {REF_F0:.2f} Hz")

# ---------- preprocess ----------
def preprocess(seg: AudioSegment) -> AudioSegment:
    """
    • keep first SAFE_PAUSE_MS of silence
    • drop excess low-level noise beyond that
    • micro-fade edges, then loudness & pitch fix
    """
    # 1) detect leading silence
    trim_seg = strip_silence(seg,
                             silence_thresh=SILENCE_THRES_DB,
                             padding=0)
    lead_sil_ms = len(seg) - len(trim_seg)

    # 2) decide how much to keep
    keep_ms = min(lead_sil_ms, SAFE_PAUSE_MS)
    seg_kept = seg[:keep_ms] + trim_seg    # concatenate kept pause + voiced

    # 3) tiny edge fades
    seg_kept = seg_kept.fade_in(EDGE_FADE_MS).fade_out(EDGE_FADE_MS)

    # 4) artistic fade-in (for breathy starts)
    seg_kept = seg_kept.fade_in(FADE_IN_MS)

    # 5) pitch & loudness
    seg_kept = match_pitch(REF_F0, seg_kept)
    seg_kept = conform_clip_memory(seg_kept, TARGET_LUFS)
    return seg_kept

# ---------- player ----------
def stream_crossfade_sequence(segments: list[AudioSegment]):
    sr = segments[0].frame_rate
    chunk = int(sr * CHUNK_MS / 1000)
    fade  = int(sr * CROSSFADE_MS / 1000)

    pa  = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paInt16, channels=1, rate=sr,
                  output=True, frames_per_buffer=chunk)

    def seg_np(seg): return np.array(seg.get_array_of_samples(), np.int16)

    prev = seg_np(segments[0])
    pos  = 0
    # play body of first clip (leave 'fade' samples for overlap)
    lead_len = len(prev) - fade
    while pos < lead_len:
        out.write(prev[pos:pos + chunk].tobytes())
        pos += chunk

    for nxt_seg in segments[1:]:
        nxt = seg_np(nxt_seg)

        # ---- cross-fade prev tail ↔ next head ----
        blend = ((1 - np.linspace(0, 1, fade)) * prev[-fade:] +
                 np.linspace(0, 1, fade) * nxt[:fade]).astype(np.int16)
        out.write(blend.tobytes())

        # ---- stream rest of next clip ----
        pos = fade
        while pos < len(nxt):
            out.write(nxt[pos:pos + chunk].tobytes())
            pos += chunk
        prev = nxt                                # advance window

    out.stop_stream(); out.close(); pa.terminate()

# ---------- main ----------
def main():
    # Put clips in the **exact** order you want to hear them
    clip_paths = [
        CLIP_DIR / "fair_enough_pending_cache.wav",
        CLIP_DIR / "fine_im_listening_pending_cache.wav",
        CLIP_DIR / "okay_hang_on_i_might_have_an_idea_long_query_cache.wav",
    ]
    if len(clip_paths) < 2:
        print("Need at least two clips."); return

    segments = [preprocess(AudioSegment.from_file(p, "wav")) for p in clip_paths]
    print(f"▶️ Streaming {len(segments)} clips with {CROSSFADE_MS} ms overlaps…")
    stream_crossfade_sequence(segments)

if __name__ == "__main__":
    main()
