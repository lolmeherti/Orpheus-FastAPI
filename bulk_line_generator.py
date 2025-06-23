#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bulk Voice Line Generator (Asynchronous)

This script is designed to receive a single core sentence and generate multiple,
emotionally distinct audio versions of it. It serves as a tool for an AI
to explore how different phrasing, punctuation, vocal tags, and speech speed
can alter the phonetic delivery and emotional subtext of a single idea.

---------------------------------
High-Level Goal
---------------------------------
The user will provide a single base sentence (e.g., "Hello" or "It's done").
The AI's task is to populate the `LINES` dictionary below by writing five
subtly different variations of that *same* sentence for *each* emotional tag.

The objective is to create distinct emotional performances of the same words.

---------------------------------
Usage (for the AI)
---------------------------------
1.  Receive the base sentence from the user.
2.  For each key (emotional tag) in the `LINES` dictionary:
3.  Write five variations of the base sentence in the corresponding list.
    Each variation must be a tuple containing the text and the speed:
    ( "sentence text", speed_float )
4.  Modify each variation using the techniques in the authoring guide below
    to match the phonetic and emotional profile of the tag.

---------------------------------
Authoring Guide for AI Variations
---------------------------------
Your primary task is to manipulate the performance of the base sentence using
four tools: punctuation, capitalization, vocal tags, and speech speed.

1.  **Speech Speed (float):** Controls the pace of the delivery.
    -   Slower (< 1.0): For sadness, intimacy, thoughtfulness. (e.g., 0.85)
    -   Normal (1.0): For neutral, calm, or standard delivery.
    -   Faster (> 1.0): For urgency, excitement, or panic. (e.g., 1.25)

2.  **Punctuation Controls Pacing:**
    -   `.` (Period): A standard, neutral stop.
    -   `..` (Double Period): A slightly longer, more thoughtful or weary pause.
    -   `...` (Ellipsis): A trailing off, hesitant, or uncertain pause.
    -   `!` (Exclamation Mark): Urgency, excitement, or force.
    -   `?` (Question Mark): Inquiry or uncertainty.

3.  **Capitalization Controls Intensity:**
    -   `lowercase`: Tends to produce a softer, more subdued delivery.
    -   `ALL CAPS`: Shouting, urgency, or extreme emphasis.

4.  **Vocal Effect Tags Control Emotion:**
    Embed these tags directly in the text. They are performance directions.

    TAG        EFFECT
    ---------- ------------------------------------------
    <sigh>     A soft breath of weariness, sadness, or relief.
    <chuckle>  Light amusement, warmth, or quiet laughter.
    <gasp>     A sharp intake of breath for surprise or shock.
    <whisper>  A very soft, close, and quiet delivery.

---------------------------------
Concrete Example
---------------------------------
If the user's base sentence is: **"It's over"**

Your generated `LINES` dictionary should look something like this:

"grief": [
    ( "<sigh> ..it's over..", 0.85 ),
    ( "It's... it's over.", 0.9 ),
    ( "So, it's over..", 0.8 ),
    ( "It's over...", 0.9 ),
    ( "<sigh> It's finally over..", 0.85 ),
],
"emergency": [
    ( "IT'S OVER! GET OUT!", 1.3 ),
    ( "It's over! We have to move, now!", 1.4 ),
    ( "It's over! Is everyone okay?!", 1.25 ),
    ( "NOW! It's over!", 1.35 ),
    ( "IT'S OVER! I repeat, it is over!", 1.2 ),
],
# ...and so on for all other tags.
"""

import re
import time
import uuid
import logging
import asyncio
from pathlib import Path
from typing import List, Tuple
import httpx
from itertools import islice

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR
CACHE_DIR = ROOT_DIR / "cached_clips"

TTS_API_URL = "http://127.0.0.1:5005/v1/audio/speech"
VOICE = "tara"

# ------------------------------------------------------------------------------------
# --  VOICE LINES TO GENERATE (AI POPULATES THIS)  --
# ------------------------------------------------------------------------------------
# This dictionary is the target for the AI. It should be populated with
# tuples of ("sentence text", speed_float).
#
# Placeholder for AI generation:
LINES: dict[str, List[Tuple[str, float]]] = {
#     "philosophical": [
#
#     ],
#     "emergency": [
#
#     ],
#     "grief": [
#
#     ],
#     "intimate": [
#
#     ],
#     "banter": [
#
#     ],
#     "affirming": [
#
#     ],
    "neutral": [

    ],
#     "flirt": [
#
#     ],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def slugify(text: str) -> str:
    """Creates a filesystem-safe, snake_case filename from a string."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s]", " ", text).lower()
    return re.sub(r"\s+", "_", text).strip("_")

def assert_counts(data: dict, want: int = 5) -> None:
    """Verify that each tag has the expected number of lines."""
    for tag, lst in data.items():
        if len(lst) != want:
            raise ValueError(f"Tag '{tag}' needs {want} lines, but it has {len(lst)}")

async def tts_request(
    client: httpx.AsyncClient,
    txt: str,
    speed: float,
    generation_id: uuid.UUID
) -> bytes | None:
    """(Async) Makes a single request to the TTS API with a specific speed."""
    payload = {
        "input": txt,
        "model": "orpheus",
        "voice": VOICE,
        "response_format": "wav",
        "speed": speed,
        "generation_id": str(generation_id)
    }
    try:
        response = await client.post(TTS_API_URL, json=payload, timeout=3000)
        response.raise_for_status()
        return response.content
    except httpx.RequestError as e:
        logging.error(f"Request failed for '{txt[:30]}...': {e}")
        return None
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP Error for '{txt[:30]}...': {e.response.status_code} - {e.response.text}")
        return None

async def generate_and_save_clip(
    client: httpx.AsyncClient,
    sentence_text: str,
    sentence_speed: float,
    out_dir: Path,
    gen_id: uuid.UUID,
    index: int
) -> None:
    """Orchestrates TTS request and file saving for a single line."""
    wav_bytes = await tts_request(
        client=client,
        txt=sentence_text,
        speed=sentence_speed,
        generation_id=gen_id,
    )
    if not wav_bytes:
        logging.error(f"  × failed for: {sentence_text}")
        return

    base_filename = slugify(sentence_text)
    dest = out_dir / f"{base_filename}_{index + 1}.wav"

    dest.write_bytes(wav_bytes)
    logging.info(f"  ✓ {dest.relative_to(CACHE_DIR)} (speed: {sentence_speed})")

async def main() -> None:
    """Simple sequential voice line generation with unique filenames."""
    gen_id = uuid.uuid4()
    start = time.time()

    async with httpx.AsyncClient() as client:
        for tag, sentences in LINES.items():
            out_dir = CACHE_DIR / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"[{tag}] Total lines: {len(sentences)} → Output dir: {out_dir}")

            for index, (sentence_text, sentence_speed) in enumerate(sentences):
                logging.info(f"  Generating [{index + 1}/{len(sentences)}]: {sentence_text}")
                await generate_and_save_clip(
                    client,
                    sentence_text,
                    sentence_speed,
                    out_dir,
                    gen_id,
                    index=index
                )
                await asyncio.sleep(0.25)  # Optional throttle

    logging.info("\nAll done. Total time: %.1fs" % (time.time() - start))

if __name__ == "__main__":
    asyncio.run(main())