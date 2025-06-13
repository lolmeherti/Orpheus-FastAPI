#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes ➜ Orpheus voice chat
• First sentence instant
• Prefetch batches the rest
• Rolling memory compressed into a ‘🗂 Memories’ system message
"""

import os, json, requests, pathlib, uuid, threading, queue, re, sys, time, traceback
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# ── endpoints / voice ───────────────────────────────────────────────────────
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
ORPHEUS_API_URL    = "http://127.0.0.1:5005/v1/audio/speech"
VOICE              = "tara"

# ── batching params ─────────────────────────────────────────────────────────
CHUNK_CHAR_MAX   = 160
CHUNK_SENT_MAX   = 3
TOKEN_SOFT_LIMIT = 6500
TOKEN_HARD_LIMIT = 7700

OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── queues / threads ────────────────────────────────────────────────────────
_audio_q:    "queue.Queue[pathlib.Path]" = queue.Queue()
_prefetch_q: "queue.Queue[tuple[list[str], float]]" = queue.Queue()

def _audio_player():
    while True:
        p = _audio_q.get()
        if p is None:
            break
        data, sr = sf.read(str(p), dtype="float32")
        sd.play(data, sr); sd.wait()
        p.unlink(missing_ok=True)

def _prefetch_worker():
    while True:
        item = _prefetch_q.get()
        if item is None:
            break
        sentences, avail = item
        start = time.time()
        buf, cnt = "", 0
        for i, line in enumerate(sentences):
            cand = f"{buf} {line}".strip() if buf else line
            if (len(cand) <= CHUNK_CHAR_MAX and
                cnt + 1 <= CHUNK_SENT_MAX and
                not re.fullmatch(r"<.*?>", line)):
                buf, cnt = cand, cnt + 1
                continue
            if buf:
                print(f"→ {buf}")
                enqueue_audio(tts_request(buf))
                buf, cnt = "", 0
            print(f"→ {line}")
            enqueue_audio(tts_request(line))
            if time.time() - start > avail:
                _prefetch_q.put((sentences[i+1:], 0))
                break
        if buf:
            print(f"→ {buf}")
            enqueue_audio(tts_request(buf))

# ── helpers ─────────────────────────────────────────────────────────────────
SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
def rough_tokens(t: str) -> int: return max(1, len(t) // 4)

def hermes_sentences(prompt: str) -> list[str]:
    raw = requests.post(
        LM_STUDIO_CHAT_URL,
        json={"model":"hermes",
              "messages":[{"role":"user","content":prompt}],
              "temperature":0.9,"max_tokens":512},
        timeout=30).json()["choices"][0]["message"]["content"]

    out=[]
    for line in raw.strip().splitlines():
        line=line.strip()
        if not line: continue
        if re.fullmatch(r"<.*?>", line): out.append(line)
        else: out += SPLIT_RE.split(line)
    return [s.strip() for s in out if s.strip()]

SUMMARY_PROMPT = (
    "You are compressing a chat transcript for future context. "
    "Summarise it as briefly as possible while preserving key facts, "
    "emotional tone, and open questions. Prioritise the most recent events."
)

def summarise(text:str)->str:
    try:
        r=requests.post(
            LM_STUDIO_CHAT_URL,
            json={"model":"hermes",
                  "messages":[
                      {"role":"system","content":SUMMARY_PROMPT},
                      {"role":"user","content":text}],
                  "temperature":0.2,"max_tokens":256},
            timeout=30)
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("‼️ Summary failed:", e)
        return "[summary unavailable]"

def tts_request(txt:str)->pathlib.Path:
    wav=requests.post(
        ORPHEUS_API_URL,
        json={"input":txt,"model":"orpheus","voice":VOICE,
              "response_format":"wav","speed":1},
        timeout=60).content
    fn=datetime.utcnow().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+".wav"
    path=OUTPUT_DIR/fn; path.write_bytes(wav); return path

def enqueue_audio(p:pathlib.Path)->None:_audio_q.put(p)

def chunk_for_audio(text:str)->list[str]:
    sents=hermes_sentences(text)
    if not sents: return []
    first, rest=sents[0], sents[1:]
    chunks, buf, n=[first], "", 0
    for s in rest:
        cand=f"{buf} {s}".strip() if buf else s
        if (len(cand)<=CHUNK_CHAR_MAX and n+1<=CHUNK_SENT_MAX
            and not re.fullmatch(r"<.*?>",s)):
            buf, n=cand, n+1; continue
        if buf: chunks.append(buf); buf, n="",0
        chunks.append(s)
    if buf: chunks.append(buf)
    return chunks

# ── main ────────────────────────────────────────────────────────────────────
def main():
    threading.Thread(target=_audio_player,daemon=True).start()
    threading.Thread(target=_prefetch_worker,daemon=True).start()
    print("🎙️  Type 'quit' to exit.\n")

    chat: list[dict] = []          # no default system persona

    try:
        while True:
            user=input("You: ").strip()
            if user.lower() in {"quit","exit"}: break
            chat.append({"role":"user","content":user})

            # summarise when near limit
            if sum(rough_tokens(m["content"]) for m in chat) > TOKEN_SOFT_LIMIT:
                older, recent = chat[:-10], chat[-10:]
                older_text="\n".join(f"{m['role']}: {m['content']}"
                                    for m in older)
                summary=summarise(older_text)
                chat=[{"role":"system",
                       "content":f"🗂 Memories:\n{summary}"}]+recent
                print("🔹 Memories updated.")

            # hard cap safeguard
            while sum(rough_tokens(m["content"]) for m in chat) > TOKEN_HARD_LIMIT:
                for i,m in enumerate(chat):
                    if m["role"]!="system":
                        del chat[i]; break

            # call Hermes
            try:
                resp=requests.post(
                    LM_STUDIO_CHAT_URL,
                    json={"model":"hermes","messages":chat,
                          "temperature":0.9},timeout=30)\
                        .json()["choices"][0]["message"]["content"]
            except Exception as e:
                print("‼️ Hermes error:",e); traceback.print_exc(); continue
            chat.append({"role":"assistant","content":resp})

            # audio pipeline
            chunks=chunk_for_audio(resp)
            if not chunks: continue
            print(f"→ {chunks[0]}")
            first_path=tts_request(chunks[0]); enqueue_audio(first_path)
            dur=sf.info(str(first_path)).frames/sf.info(str(first_path)).samplerate
            _prefetch_q.put((chunks[1:],max(dur-0.5,0)))

    finally:
        _audio_q.put(None); _prefetch_q.put(None)
        if not sys.stdin.isatty():
            input("Press ENTER to close…")

if __name__=="__main__":
    main()
