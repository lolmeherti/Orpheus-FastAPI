# eq_loudnorm.py
import json, subprocess, tempfile, shutil, os, re
from pathlib import Path
from pydub import AudioSegment

FFMPEG = "ffmpeg"

EQ_FILTER = (
    "equalizer=f=150:t=h:width=200:g=-4,"
    "equalizer=f=2500:t=q:width=1000:g=2,"
    "equalizer=f=8000:t=q:width=2000:g=1"
)

LOUDNORM_CMD = [
    FFMPEG, "-hide_banner", "-i", "{in}",
    "-af", "loudnorm=print_format=json",
    "-f", "null", "-"
]

def loudnorm_stats(wav: Path) -> dict:
    """Return {input_i, input_tp} from ffmpeg loudnorm JSON."""
    out = subprocess.check_output(
        [a.replace("{in}", str(wav)) for a in LOUDNORM_CMD],
        stderr=subprocess.STDOUT
    ).decode()
    m = re.search(r"\{[\s\S]+?\}", out)
    return json.loads(m.group(0))

def conform_clip(wav: Path, ref_lufs: float, ref_tp: float,
                 tmp_dir: Path, eq_filter: str | None = EQ_FILTER) -> Path:
    out = tmp_dir / f"eq_{wav.name}"

    target_tp = max(min(ref_tp, 0.0), -9.0)

    af = f"loudnorm=I={ref_lufs}:TP={target_tp}:LRA=2"
    if eq_filter:
        af = f"{af},{eq_filter}"

    subprocess.run(
        [FFMPEG, "-y", "-i", str(wav), "-af", af, str(out)],
        capture_output=True, check=True
    )
    return out
