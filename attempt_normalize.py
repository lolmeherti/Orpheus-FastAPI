# normalize_corpus.py - FINAL, DEFINITIVE, FULLY IN-MEMORY BATCH PROCESSOR
# This script uses your calculated "Golden Profile" to normalize
# both pitch and loudness for your entire audio library.

from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# Import the fast, in-memory processing functions
from prosody_fast import match_pitch_memory
from eq_loudnorm_fast import conform_clip_memory

# --- CONFIG: YOUR GOLDEN PROFILE ---
# Paste the exact average values you got from the analysis script.
GOLDEN_PROFILE_F0 = 195.6029
GOLDEN_PROFILE_LUFS = -27.8400

# Directories
SOURCE_DIR = Path("cached_clips")
TARGET_DIR = Path("normalized_clips")
TARGET_DIR.mkdir(exist_ok=True)

# --- FULLY IN-MEMORY BATCH PROCESSING PIPELINE ---
print("Starting full corpus normalization process (fully in-memory)...")
print(f"  - Target Pitch:    {GOLDEN_PROFILE_F0:.2f} Hz")
print(f"  - Target Loudness: {GOLDEN_PROFILE_LUFS:.2f} LUFS")

clip_paths = list(SOURCE_DIR.glob("*.wav"))

if not clip_paths:
    print(f"Error: No .wav files found in {SOURCE_DIR}.")
    exit()

for path in tqdm(clip_paths, desc="Processing Clips"):
    try:
        # === STEP 1: LOAD RAW AUDIO ===
        raw_seg = AudioSegment.from_wav(path)

        # === STEP 2: PITCH CORRECTION (In-Memory) ===
        pitched_seg = match_pitch_memory(GOLDEN_PROFILE_F0, raw_seg)

        # === STEP 3: LOUDNESS NORMALIZATION (In-Memory) ===
        # No more temp files! We use our fast function.
        final_seg = conform_clip_memory(pitched_seg, target_loudness=GOLDEN_PROFILE_LUFS)

        # === STEP 4: EXPORT FINAL FILE ===
        final_output_path = TARGET_DIR / path.name
        final_seg.export(final_output_path, format="wav")

    except Exception as e:
        print(f"\nERROR: Failed to process {path.name}. Reason: {e}")
        continue

print("\n--- NORMALIZATION COMPLETE ---")
print(f"All processed clips have been saved to the '{TARGET_DIR}' folder.")