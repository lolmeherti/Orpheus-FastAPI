#!/usr/bin/env python
# median_loudnorm.py
import subprocess, re, statistics, sys
from pathlib import Path

FFMPEG = "ffmpeg"                                   # relies on PATH
CLIP_DIR = Path("cached_clips")
wav_files = sorted(CLIP_DIR.glob("*.wav"))

if not wav_files:
    sys.exit("No .wav files found in cached_clips/")

lufs_list, peak_list = [], []

pattern_i = re.compile(r"Input Integrated:\s+(-?[\d\.]+)")
pattern_p = re.compile(r"Input True Peak:\s+(-?[\d\.]+)")

for wav in wav_files:
    cmd = [
        FFMPEG, "-v", "error", "-i", str(wav),
        "-af", "loudnorm=print_format=summary", "-f", "null", "-"
    ]
    print(f"▶ {wav.name}")
    out = subprocess.run(cmd, capture_output=True, text=True).stderr

    match_i = pattern_i.search(out)
    match_p = pattern_p.search(out)

    if match_i and match_p:
        lufs = float(match_i.group(1))
        peak = float(match_p.group(1))
        lufs_list.append(lufs)
        peak_list.append(peak)
        print(f"    LUFS={lufs:.2f}  TP={peak:+.2f} dB")
    else:
        print("    loudnorm parsing failed")

# --- results ---
if lufs_list:
    print("\n=== MEDIANS ===")
    print(f"Median LUFS : {statistics.median(lufs_list):.2f}")
    print(f"Median TP   : {statistics.median(peak_list):+.2f} dBTP")
else:
    print("\nNo valid loudnorm summaries parsed.")
