#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
import subprocess
import re
import urllib.request
import uuid
import threading
from typing import Optional, List, Dict

import whisper
import spacy

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware


# ---------- TUNING ----------
PLAY_RES_X = 1080
PLAY_RES_Y = 1920

LIST_Y = 820
LIST_X = 140
WORD_X = 250

FONT_SIZE = 45
OUTLINE = 5

LOGO_SCALE_W = 160
LOGO_PAD_X = 30
LOGO_PAD_Y = 30
# --------------------------

app = FastAPI()

# CORS for Rork web preview and browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log requests so you can see what Rork hits in Railway logs
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"REQ {request.method} {request.url.path}")
    resp = await call_next(request)
    print(f"RES {request.method} {request.url.path} {resp.status_code}")
    return resp

# Preflight for browsers. Some clients call /api/process, some call /process
@app.options("/process")
def options_process():
    return Response(status_code=204)

@app.options("/api/process")
def options_api_process():
    return Response(status_code=204)


nlp = None
_whisper_model = None
TRANSCRIBE_LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()


class ProcessReq(BaseModel):
    video_url: str = Field(..., description="Direct public or signed mp4/mov url")
    slots: int = 5
    target_fps: int = 30
    sub_primary_hex: str = "FFFF00"
    logo_enabled: bool = True
    logo_url: Optional[str] = None
    output_prefix: str = "ReelFive_"


@app.on_event("startup")
def _startup():
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError(f"startup failed: {e}")


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def ensure_tools():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise HTTPException(status_code=500, detail="ffmpeg or ffprobe missing")


def download_file(url: str, dest_path: str):
    if not url:
        raise HTTPException(status_code=400, detail="video_url missing")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise HTTPException(status_code=400, detail=f"download failed http {status}")
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download failed: {e}")


def get_duration_or_zero(path: str) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ],
        capture_output=True, text=True
    )
    out = (r.stdout or "").strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def normalize_word(w: str) -> str:
    w = (w or "").strip().lower()
    w = w.replace("’", "'")
    w = re.sub(r"[^a-z']", "", w)
    return w


def strip_ass_tags(s: str) -> str:
    return re.sub(r"\{\\[^}]*\}", "", s or "").strip()


def esc_ass_text(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def clamp_time(t: float, lo: float, hi: float) -> float:
    if t < lo:
        return lo
    if t > hi:
        return hi
    return t


def ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    cs = int(round(t * 100.0))
    h = cs // 360000
    cs -= h * 360000
    m = cs // 6000
    cs -= m * 6000
    s = cs // 100
    cs -= s * 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def esc_ff_filter(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_numbers_block(slots: int) -> str:
    return "\\N".join([f"{i}." for i in range(1, slots + 1)])


def build_words_block(slots: int, filled: Dict[int, str]) -> str:
    out = []
    for i in range(1, slots + 1):
        out.append(esc_ass_text(filled.get(i, "")))
    return "\\N".join(out)


def write_ass(path: str, duration: float, slots: int, events_words: List[Dict], primary_hex: str):
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", (primary_hex or "")):
        primary_hex = "FFFF00"

    rr = primary_hex[0:2]
    gg = primary_hex[2:4]
    bb = primary_hex[4:6]
    primary = f"&H00{bb}{gg}{rr}"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: NUM,Arial,{FONT_SIZE},{primary},&H00000000,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{OUTLINE},0,4,0,0,0,0
Style: WORD,Arial,{FONT_SIZE},{primary},&H00000000,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{OUTLINE},0,4,0,0,0,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    numbers_text = build_numbers_block(slots)
    num_event = (
        f"Dialogue: 0,{ass_time(0.0)},{ass_time(duration)},NUM,,0,0,0,,"
        f"{{\\pos({LIST_X},{LIST_Y})\\an4\\fsp0}}{numbers_text}\n"
    )

    word_lines = []
    for ev in events_words:
        start = clamp_time(float(ev["start"]), 0.0, duration)
        end = clamp_time(float(ev["end"]), 0.0, duration)
        if end <= start:
            continue
        txt = ev["text"]
        word_lines.append(
            f"Dialogue: 1,{ass_time(start)},{ass_time(end)},WORD,,0,0,0,,"
            f"{{\\pos({WORD_X},{LIST_Y})\\an4\\fsp0}}{txt}\n"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(num_event)
        for line in word_lines:
            f.write(line)


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/health")
def health_api():
    return {"ok": True}


@app.post("/process")
def process(req: ProcessReq):
    ensure_tools()

    if nlp is None:
        raise HTTPException(status_code=500, detail="spaCy not loaded")

    slots = int(req.slots)
    if slots < 1:
        raise HTTPException(status_code=400, detail="slots must be >= 1")
    if slots > 10:
        raise HTTPException(status_code=400, detail="slots too high, max 10")

    work = f"/tmp/work_{uuid.uuid4().hex}"
    os.makedirs(work, exist_ok=True)

    out_path = ""
    try:
        in_path = os.path.join(work, "input.mp4")
        download_file(req.video_url, in_path)
        if not os.path.exists(in_path):
            raise HTTPException(status_code=400, detail="video download failed")

        duration = get_duration_or_zero(in_path)
        if duration <= 0:
            raise HTTPException(status_code=400, detail="ffprobe duration failed")

        model = get_whisper_model()

        with PROCESS_LOCK:
            with TRANSCRIBE_LOCK:
                try:
                    result = model.transcribe(in_path, word_timestamps=True, fp16=False)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"whisper transcribe failed: {e}")

            words: List[Dict] = []
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    token = (w.get("word") or "").strip()
                    if token:
                        words.append({"word": token, "start": float(w.get("start", 0) or 0)})

            if len(words) == 0:
                raise HTTPException(status_code=500, detail="no transcript words")

            docs = list(nlp.pipe([w["word"] for w in words]))
            for i, doc in enumerate(docs):
                words[i]["pos"] = doc[0].pos_ if len(doc) else "X"

            preferred = [w for w in words if w["pos"] in ("NOUN", "PROPN")]
            seg_len = duration / max(slots, 1)

            overlay = []
            for i in range(slots):
                seg_start = i * seg_len
                seg_end = seg_start + seg_len
                seg_words = [w for w in preferred if seg_start <= w["start"] < seg_end]
                chosen = seg_words[0] if len(seg_words) > 0 else words[min(i, len(words) - 1)]

                clean = normalize_word(chosen["word"]).upper()
                final_word = clean if clean else (chosen["word"] or "").strip().upper()
                final_word = strip_ass_tags(final_word)

                t = float(chosen["start"])
                t = clamp_time(t, 0.0, max(0.0, duration - 0.01))

                overlay.append({"slot": i + 1, "word": final_word, "time": t})

            overlay.sort(key=lambda x: x["time"])

            min_chunk = 0.60
            filled: Dict[int, str] = {}
            events_words: List[Dict] = []

            first_time = overlay[0]["time"]
            if first_time < min_chunk:
                first_time = min_chunk

            events_words.append({"start": 0.0, "end": first_time, "text": build_words_block(slots, filled)})

            for i, item in enumerate(overlay):
                filled[item["slot"]] = item["word"]
                start = item["time"]
                end = overlay[i + 1]["time"] if i < len(overlay) - 1 else duration
                if end - start < min_chunk:
                    end = min(duration, start + min_chunk)
                events_words.append({"start": start, "end": end, "text": build_words_block(slots, filled)})

            ass_path = os.path.join(work, "list.ass")
            write_ass(ass_path, duration, slots, events_words, req.sub_primary_hex)

            use_logo = False
            logo_path = os.path.join(work, "logo.png")
            if req.logo_enabled:
                if req.logo_url:
                    download_file(req.logo_url, logo_path)
                    use_logo = os.path.exists(logo_path)
                else:
                    local_logo = "/app/logo.png"
                    if os.path.exists(local_logo):
                        shutil.copy(local_logo, logo_path)
                        use_logo = os.path.exists(logo_path)

            out_name = f"{req.output_prefix}{uuid.uuid4().hex}.mp4"
            out_path = os.path.join(work, out_name)

            ass_f = esc_ff_filter(ass_path)
            logo_f = esc_ff_filter(logo_path)

            sub_filter = f"subtitles='{ass_f}'"
            fps_filter = f"fps={int(req.target_fps)}"

            if use_logo:
                vf = (
                    f"[0:v]{fps_filter},{sub_filter}[v];"
                    f"movie='{logo_f}',scale={LOGO_SCALE_W}:-1[logo];"
                    f"[v][logo]overlay=W-w-{LOGO_PAD_X}:{LOGO_PAD_Y}:format=auto"
                )
                cmd = [
                    "ffmpeg", "-y",
                    "-i", in_path,
                    "-filter_complex", vf,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-r", str(int(req.target_fps)),
                    "-c:a", "aac", "-b:a", "128k",
                    out_path
                ]
            else:
                vf = f"{fps_filter},{sub_filter}"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", in_path,
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-r", str(int(req.target_fps)),
                    "-c:a", "aac", "-b:a", "128k",
                    out_path
                ]

            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                tail = (p.stderr or "")[-2500:]
                raise HTTPException(status_code=500, detail=f"ffmpeg failed: {tail}")

            if not os.path.exists(out_path):
                raise HTTPException(status_code=500, detail="output mp4 missing")

        cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
        return FileResponse(out_path, media_type="video/mp4", filename=os.path.basename(out_path), background=cleanup)

    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"server error: {e}")


# Rork calls /api/process. Keep this alias.
@app.post("/api/process")
def process_api(req: ProcessReq):
    return process(req)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




