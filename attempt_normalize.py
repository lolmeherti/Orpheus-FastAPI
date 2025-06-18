import os
import subprocess
import json
from pathlib import Path
import csv

# === CONFIG ===
input_folder = Path("cached_clips")
output_folder = Path("normalized")
output_folder.mkdir(exist_ok=True)

LUFS_TARGET = -26.5
LRA_THRESHOLD = 1.0

results = []

def analyze_audio(file_path):
    cmd = [
        "ffmpeg",
        "-i", str(file_path),
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = result.stdout

    try:
        json_start = output.index('{')
        json_end = output.rindex('}') + 1
        loudness_data = json.loads(output[json_start:json_end])
        return {
            "filename": file_path.name,
            "input_i": float(loudness_data["input_i"]),
            "input_tp": float(loudness_data["input_tp"]),
            "input_lra": float(loudness_data["input_lra"]),
            "normalize": (float(loudness_data["input_i"]) < LUFS_TARGET) and (float(loudness_data["input_lra"]) <= LRA_THRESHOLD),
            "error": ""
        }
    except Exception as e:
        return {
            "filename": file_path.name,
            "input_i": "",
            "input_tp": "",
            "input_lra": "",
            "normalize": False,
            "error": str(e)
        }

for wav_file in input_folder.glob("*.wav"):
    analysis = analyze_audio(wav_file)
    results.append(analysis)

    if analysis.get("normalize"):
        output_path = output_folder / wav_file.name
        subprocess.run([
            "ffmpeg-normalize", str(wav_file),
            "-o", str(output_path),
            "-nt", "ebu",
            "-t", str(LUFS_TARGET),
            "-f"
        ])

# === CSV Export ===
csv_path = output_folder / "normalization_log.csv"
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "input_i", "input_tp", "input_lra", "normalize", "error"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Done. Log saved to: {csv_path}")
