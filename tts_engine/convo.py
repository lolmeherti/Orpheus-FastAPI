#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes ➜ Orpheus voice chat
• 1st chunk instant (≥10 words)
• Later chunks ≈35 words, tag-aware
• Rolling 🗂 Memories triggered strictly by TOKEN_SOFT_LIMIT
• Live token counter, mem + dump commands
• Strips *asterisk*, _underscore_, and (parenthetical) actions
"""

import requests, pathlib, uuid, threading, queue, re, sys, traceback
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# ── endpoints / settings ────────────────────────────────────────────────────
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
ORPHEUS_API_URL    = "http://127.0.0.1:5005/v1/audio/speech"
VOICE              = "tara"

MIN_FIRST_WORDS   = 10
CHUNK_WORD_MAX    = 35
TOKEN_SOFT_LIMIT  = 500          # dev threshold
TOKEN_HARD_LIMIT  = 7700
KEEP_RECENT       = 1            # keep last turn verbatim

OUTPUT_DIR = pathlib.Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

# ── queues / background workers ─────────────────────────────────────────────
_audio_q:    "queue.Queue[pathlib.Path]" = queue.Queue()
_prefetch_q: "queue.Queue[list[str]]"    = queue.Queue()

def _audio_player():
    while True:
        p = _audio_q.get(); _audio_q.task_done()
        if p is None: break
        data, sr = sf.read(str(p), dtype="float32")
        sd.play(data, sr); sd.wait(); p.unlink(missing_ok=True)

def _prefetch_worker():
    while True:
        lst = _prefetch_q.get(); _prefetch_q.task_done()
        if lst is None: break
        for chunk in lst:
            print(f"→ {chunk}")
            enqueue_audio(tts_request(chunk))

# ── regex helpers ──────────────────────────────────────────────────────────
SENT_RE   = re.compile(r'(?<=[.!?])\s+')
ACTION_RE = re.compile(r'(\*[^\*]+\*|_[^_]+_|\([^)]+\))')

def rough_tokens(txt:str)->int: return max(1, len(txt)//4)
def strip_stage_dirs(txt:str)->str: return ACTION_RE.sub('', txt).strip()

def split_sentences(txt:str)->list[str]:
    out=[]
    for ln in txt.strip().splitlines():
        ln=ln.strip()
        if not ln: continue
        if re.fullmatch(r"<.*?>", ln): out.append(ln)
        else: out.extend(SENT_RE.split(ln))
    return [s.strip() for s in out if s.strip()]

def batch_sentences(sents:list[str], max_words:int=35, first_min:int=10)->list[str]:
    if not sents: return []
    first, words, i = [], 0, 0
    while i < len(sents) and words < first_min:
        if first and re.fullmatch(r"<.*?>", sents[i]): break
        first.append(sents[i]); words += len(sents[i].split()); i += 1
    batches, buf, buf_words = [" ".join(first)], [], 0
    for s in sents[i:]:
        if re.fullmatch(r"<.*?>", s):
            if buf: batches.append(" ".join(buf)); buf, buf_words = [], 0
            batches.append(s); continue
        w=len(s.split())
        if buf_words+w>max_words:
            if buf: batches.append(" ".join(buf))
            buf, buf_words = [s], w
        else:
            buf.append(s); buf_words+=w
    if buf: batches.append(" ".join(buf))
    return batches

# ── summarizer ─────────────────────────────────────────────────────────────
SUMMARY_PROMPT=("Summarize the transcript provided. Keep it brief and factual. "
                "Focus on recent events, feelings, open questions. Do NOT invent.")

def summarise_with_hermes(text:str)->str:
    print("🧾 Summarizing block (first 800 chars):\n" + text[:800] + ("…" if len(text)>800 else ""))
    r=requests.post(
        LM_STUDIO_CHAT_URL,
        json={"model":"hermes",
              "messages":[
                  {"role":"system","content":SUMMARY_PROMPT},
                  {"role":"user","content":"Transcript:\n"+text}],
              "temperature":0.2,"max_tokens":256},
        timeout=30)
    return r.json()["choices"][0]["message"]["content"].strip()

# ── network helpers ─────────────────────────────────────────────────────────
def hermes_chat(msgs:list[dict])->str:
    return requests.post(
        LM_STUDIO_CHAT_URL,
        json={"model":"hermes","messages":msgs,"temperature":0.9},
        timeout=30).json()["choices"][0]["message"]["content"]

def tts_request(text:str)->pathlib.Path:
    wav=requests.post(
        ORPHEUS_API_URL,
        json={"input":text,"model":"orpheus","voice":VOICE,
              "response_format":"wav","speed":1},
        timeout=60).content
    fn=datetime.utcnow().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+".wav"
    p=OUTPUT_DIR/fn; p.write_bytes(wav); return p
def enqueue_audio(p:pathlib.Path)->None:_audio_q.put(p)

# ── main loop ──────────────────────────────────────────────────────────────
def main():
    threading.Thread(target=_audio_player,daemon=True).start()
    threading.Thread(target=_prefetch_worker,daemon=True).start()
    print("🎙️  type 'quit' to exit — 'mem' for current summary — 'dump' for full chat\n")

    chat=[]; memories=None

    try:
        while True:
            user=input("\nYou: ").strip()
            cmd=user.lower()
            if cmd in {"quit","exit"}: break
            if cmd=="mem":
                print("\n🗂 Memories:\n" + (memories or "[none yet]")); continue
            if cmd=="dump":
                print("\n📜 Full chat list:")
                for m in chat:
                    print(f"[{m['role']}] {m['content']}")
                continue

            chat.append({"role":"user","content":user})

            try: raw=hermes_chat(chat)
            except Exception as e:
                print("‼️ Hermes error:", e); traceback.print_exc(); continue
            resp=strip_stage_dirs(raw)
            chat.append({"role":"assistant","content":resp})

            total=sum(rough_tokens(m["content"]) for m in chat)
            print(f"🧠 Tokens → {total}")

            # summarize strictly by TOKEN_SOFT_LIMIT
            if total > TOKEN_SOFT_LIMIT:
                older, recent = chat[:-KEEP_RECENT], chat[-KEEP_RECENT:]
                if older:  # ensure transcript isn't empty
                    older_text="\n".join(f"{m['role']}: {m['content']}" for m in older)
                    memories=summarise_with_hermes(older_text)
                    chat=[{"role":"system","content":f"🗂 Memories:\n{memories}"}]+recent
                    print("\n🔹 Memories updated:\n"+memories+"\n")

            # hard cap safeguard
            while sum(rough_tokens(m["content"]) for m in chat) > TOKEN_HARD_LIMIT:
                for i,m in enumerate(chat):
                    if m["role"]!="system": del chat[i]; break

            # batching & audio
            chunks=batch_sentences(split_sentences(resp))
            if not chunks: continue
            print(f"→ {chunks[0]}")
            enqueue_audio(tts_request(chunks[0]))
            _prefetch_q.put(chunks[1:])

    finally:
        _audio_q.put(None); _prefetch_q.put(None)
        if not sys.stdin.isatty():
            input("Press ENTER to close…")

if __name__=="__main__":
    main()
