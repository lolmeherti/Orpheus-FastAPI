# file: test_audio_stream_crossfade.py
import math, shutil, numpy as np, pyaudio
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import strip_silence

from eq_loudnorm_fast import conform_clip_memory
from prosody_fast import match_pitch, get_f0

# ---------- paths / constants ----------
CLIP_DIR          = Path("cached_clips")
CROSSFADE_MS      = 150
CHUNK_MS          = 20
SILENCE_THRES_DB  = -50
FADE_IN_MS        = 100
TARGET_LUFS       = -27.2
FADE_OUT_MS       = 250

# ---------- reference F0 (one cached clip) ----------
ref_clip  = AudioSegment.from_file(CLIP_DIR / "alright_i_hear_you_pending_cache.wav", "wav")
REF_F0    = get_f0(ref_clip)
print(f"🎯 reference F0 = {REF_F0:.2f} Hz")

# ---------- helper funcs ----------
def preprocess(seg: AudioSegment) -> AudioSegment:
    seg = strip_silence(seg, silence_thresh=SILENCE_THRES_DB, padding=0)
    seg = seg.fade_in(FADE_IN_MS).fade_out(60)
    seg = match_pitch(REF_F0, seg)
    seg = conform_clip_memory(seg, TARGET_LUFS)
    return seg

def stream_crossfade_sequence(clips: list[AudioSegment]):
    sr = clips[0].frame_rate
    chunk_samples = int(sr * CHUNK_MS / 1000)
    fade_samples = int(sr * CROSSFADE_MS / 1000)

    pa = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paInt16,
                  channels=1,
                  rate=sr,
                  output=True,
                  frames_per_buffer=chunk_samples)

    def seg_to_np(seg: AudioSegment) -> np.ndarray:
        return np.array(seg.get_array_of_samples()).astype(np.int16)

    def to_bytes(arr: np.ndarray) -> bytes:
        return arr.astype(np.int16).tobytes()

    first_np = seg_to_np(clips[0])
    pos = 0
    lead_len = len(first_np) - fade_samples
    while pos < lead_len:
        out.write(to_bytes(first_np[pos:pos + chunk_samples]))
        pos += chunk_samples

    prev_np = first_np
    for idx in range(1, len(clips)):
        next_np = seg_to_np(clips[idx])

        # Crossfade tail of prev into head of next
        prev_tail = prev_np[-fade_samples:]
        next_head = next_np[:fade_samples]
        fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
        mixed = ((1 - fade) * prev_tail + fade * next_head).astype(np.int16)
        out.write(to_bytes(mixed))

        # Stream the rest of the next clip (excluding what was used in fade)
        pos = fade_samples
        while pos < len(next_np):
            out.write(to_bytes(next_np[pos:pos + chunk_samples]))
            pos += chunk_samples

        prev_np = next_np

    out.stop_stream()
    out.close()
    pa.terminate()

def stream_crossfade_tail(a: AudioSegment, b: AudioSegment):
    sr              = a.frame_rate
    chunk_samples   = int(sr * CHUNK_MS / 1000)
    fade_samples    = int(sr * CROSSFADE_MS / 1000)

    pa  = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paInt16,
                  channels=1, rate=sr, output=True,
                  frames_per_buffer=chunk_samples)

    a_np, b_np = (np.array(seg.get_array_of_samples(), np.int16) for seg in (a, b))
    lead_len   = len(a_np) - fade_samples

    # play main body of A
    for pos in range(0, lead_len, chunk_samples):
        out.write(a_np[pos:pos + chunk_samples].tobytes())

    # cross-fade
    fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
    mixed = ((1 - fade) * a_np[lead_len:] + fade * b_np[:fade_samples]).astype(np.int16)
    out.write(mixed.tobytes())

    # play remainder of B
    tail_b = b_np[fade_samples:]
    for pos in range(0, len(tail_b), chunk_samples):
        out.write(tail_b[pos:pos + chunk_samples].tobytes())

    out.stop_stream(); out.close(); pa.terminate()

# ---------- main ----------
def main():
    clip_paths = [
        CLIP_DIR / "fair_enough_pending_cache.wav",
        CLIP_DIR / "fine_im_listening_pending_cache.wav",
        CLIP_DIR / "okay_hang_on_i_might_have_an_idea_long_query_cache.wav"
    ]

    if len(clip_paths) < 2:
        print("Need ≥ 2 clips.")
        return

    segments = [preprocess(AudioSegment.from_file(path, "wav")) for path in clip_paths]

    stream_crossfade_sequence(segments)

#
#     # 🔁 Load and preprocess all clips
#     segments = [preprocess(AudioSegment.from_file(path, "wav")) for path in clip_paths]
#
#     # 🔀 Crossfade all clips sequentially
#     for i in range(len(segments) - 1):
#         print(f"▶️ {clip_paths[i].name} → cross-fading into {clip_paths[i+1].name}")
#         stream_crossfade_tail(segments[i], segments[i + 1])

if __name__ == "__main__":
    main()
