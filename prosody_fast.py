# prosody_fast.py  – minimal

import math, subprocess, tempfile, pathlib
import numpy as np, torch, torchaudio
import torchaudio.functional as F
from pydub import AudioSegment

# ---------- GPU pitch (median F0) ----------
@torch.no_grad()
def get_f0(seg: AudioSegment, target_sr=22050) -> float:
    y = torch.tensor(np.array(seg.set_channels(1).get_array_of_samples())
                     .astype(np.float32)).to("cuda") / 32768.0
    if seg.frame_rate != target_sr:
        y = F.resample(y, seg.frame_rate, target_sr)
    f0 = torchaudio.functional.detect_pitch_frequency(
        y.unsqueeze(0), target_sr
    )[0]
    voiced = f0[f0 > 0]
    return float(torch.median(voiced)) if voiced.numel() else 0.0

# ---------- Rubber Band CLI helper ----------
def rb_shift(seg: AudioSegment, semitones: float) -> AudioSegment:
    if abs(semitones) < 0.3 or abs(semitones) > 1.5:
        return seg                     # skip tiny or big shifts
    with tempfile.TemporaryDirectory() as td:
        in_wav, out_wav = pathlib.Path(td) / "in.wav", pathlib.Path(td) / "out.wav"
        seg.export(in_wav, format="wav")
        subprocess.run([
            "rubberband", "-3", "-F", "-p", f"{semitones:.3f}", "-q",
            str(in_wav), str(out_wav)
        ], check=True)
        return AudioSegment.from_file(out_wav, format="wav")

# ---------- public function ----------
def match_pitch(ref_f0: float, seg: AudioSegment) -> AudioSegment:
    clip_f0 = get_f0(seg)
    if ref_f0 == 0 or clip_f0 == 0:
        return seg

    # ---- octave-fold ----
    while clip_f0 >= 1.7 * ref_f0:
        clip_f0 /= 2
    while clip_f0 <= 0.6 * ref_f0:
        clip_f0 *= 2
    # -------------------------

    semitones = 12 * math.log2(ref_f0 / clip_f0)
    print(f"Pitch: target {ref_f0:.1f} | clip {clip_f0:.1f} | shift {semitones:+.2f} st")

    if abs(semitones) < 0.3 or abs(semitones) > 1.5:
        return seg
    return rb_shift(seg, semitones)

