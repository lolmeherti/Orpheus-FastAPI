# prosody_fast.py

import math, subprocess, tempfile, pathlib
import numpy as np, torch, torchaudio
import torchaudio.functional as F
from pydub import AudioSegment

@torch.no_grad()
def get_f0(seg: AudioSegment, target_sr=22050) -> float:
    y = torch.tensor(np.array(seg.set_channels(1).get_array_of_samples())
                     .astype(np.float32)).to("cuda") / 32768.0
    if seg.frame_rate != target_sr:
        y = F.resample(y, seg.frame_rate, target_sr)

    # --- FIX: BACKWARD-COMPATIBLE PITCH ESTIMATION ---
    # Run the detector without constraints.
    pitch = torchaudio.functional.detect_pitch_frequency(y.unsqueeze(0), target_sr)[0]

    # Manually filter the results to a valid vocal range.
    # This works on all versions of torchaudio.
    pitch_floor = 150
    pitch_ceiling = 450
    voiced = pitch[(pitch > pitch_floor) & (pitch < pitch_ceiling)]
    # --- END FIX ---

    return float(torch.median(voiced)) if voiced.numel() else 0.0

def rb_shift(seg: AudioSegment, semitones: float) -> AudioSegment:
    if abs(semitones) < 1.0 or abs(semitones) > 2.0:
        return seg
    with tempfile.TemporaryDirectory() as td:
        in_wav, out_wav = pathlib.Path(td) / "in.wav", pathlib.Path(td) / "out.wav"
        seg.export(in_wav, format="wav")
        subprocess.run([
            "rubberband", "-3", "-F", "-p", f"{semitones:.3f}", "-q",
            str(in_wav), str(out_wav)
        ], check=True, capture_output=True)
        return AudioSegment.from_file(out_wav, format="wav")

def match_pitch(ref_f0: float, seg: AudioSegment) -> AudioSegment:
    clip_f0 = get_f0(seg)
    if ref_f0 == 0 or clip_f0 == 0:
        return seg

    original_clip_f0 = clip_f0
    diff_orig, diff_up, diff_down = abs(ref_f0 - clip_f0), abs(ref_f0 - clip_f0 * 2), abs(ref_f0 - clip_f0 / 2)

    if diff_down < diff_orig and diff_down < diff_up:
        clip_f0 /= 2
    elif diff_up < diff_orig:
        clip_f0 *= 2

    semitones = 12 * math.log2(ref_f0 / clip_f0)
    if abs(semitones) >= 1.0:
        print(f"Pitch: target {ref_f0:.1f} | clip {original_clip_f0:.1f} -> {clip_f0:.1f} | shift {semitones:+.2f} st")

    return rb_shift(seg, semitones)