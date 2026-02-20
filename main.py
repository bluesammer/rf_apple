#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import shutil
import subprocess
import re
import urllib.request
import uuid
import threading
from typing import Optional, List, Dict

import whisper
import spacy

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
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

# speed defaults
DEFAULT_FPS = 24
SCALE_W = 720
FFMPEG_PRESET = "ultrafast"
FFMPEG_CRF = "28"

# logo
LOGO_SCALE_W = 160
LOGO_PAD_X = 30
LOGO_PAD_Y = 30
# --------------------------


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"REQ {request.method} {request.url.path}")
    resp = await call_next(request)
    print(f"RES {request.method} {request.url.path} {resp.status_code}")
    return resp


@app.options("/process")
def options_process():
    return Response(status_code=204)

@app.options("/api/process")
def options_api_process():
    return Response(status_code=204)

@app.options("/api/process_upload")
def options_api_process_upload():
    return Response(status_code=204)

@app.options("/api/output/{name}")
def options_api_output(name: str):
    return Response(status_code=204)


nlp = None
_whisper_model = None
TRANSCRIBE_LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()

OUTPUT_DIR = "/tmp/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ProcessReq(BaseModel):
    video_url: str = Field(..., description="Direct public or signed mp4/mov url")
    slots: int = 5
    target_fps: int = DEFAULT_FPS
    sub_primary_hex: str = "FFFF00"
    logo_enabled: bool = False
    logo_url: Optional[str] = None
    output_prefix: str = "ReelFive_"


@app.on_event("startup")
def _startup():
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError(f"startup failed: {e}")


def tlog(label: str, t0: float):
    dt = time.time() - t0
    print(f"STAGE {label} {dt:.2f}s")


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        # speed-first model
        _whisper_model = whisper.load_model("tiny")
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


def save_upload(upload: UploadFile, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


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


def persist_output(out_path: str) -> str:
    out_base = os.path.basename(out_path)
    persist_name = f"{uuid.uuid4().hex}_{out_base}"
    persist_path = os.path.join(OUTPUT_DIR, persist_name)
    shutil.copyfile(out_path, persist_path)
    return persist_name


def pick_keywords(full_text: str, slots: int) -> List[str]:
    if nlp is None:
        return [""] * slots

    doc = nlp(full_text)
    picked: List[str] = []
    seen = set()

    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and tok.is_alpha:
            w = tok.text.strip().upper()
            w = strip_ass_tags(w)
            if not w:
                continue
            if w in seen:
                continue
            seen.add(w)
            picked.append(w)
            if len(picked) >= slots:
                break

    while len(picked) < slots:
        picked.append("")

    return picked[:slots]


def build_progressive_events(duration: float, slots: int, words: List[str]) -> List[Dict]:
    min_chunk = 0.8
    seg_len = max(duration / max(slots, 1), min_chunk)

    # reveal each slot at evenly spaced times
    reveal_times = []
    for i in range(slots):
        t = i * seg_len
        t = clamp_time(t, 0.0, max(0.0, duration - 0.01))
        reveal_times.append(t)

    events: List[Dict] = []
    filled: Dict[int, str] = {}

    # first event from 0 to first reveal
    first_end = reveal_times[0] if reveal_times else duration
    first_end = max(first_end, min_chunk)
    first_end = clamp_time(first_end, 0.0, duration)
    events.append({"start": 0.0, "end": first_end, "text": build_words_block(slots, filled)})

    for i in range(slots):
        filled[i + 1] = words[i]
        start = reveal_times[i]
        end = reveal_times[i + 1] if i < slots - 1 else duration
        end = max(end, start + min_chunk)
        end = clamp_time(end, 0.0, duration)
        events.append({"start": start, "end": end, "text": build_words_block(slots, filled)})

    return events


def render_video(in_path: str, duration: float, slots: int, target_fps: int, sub_primary_hex: str,
                 logo_enabled: bool, logo_url: Optional[str], output_prefix: str) -> (str, str):
    t0 = time.time()

    model = get_whisper_model()

    t_wh = time.time()
    print("STAGE whisper start")
    # speed-first: no word timestamps
    result = model.transcribe(in_path, word_timestamps=False, fp16=False)
    tlog("whisper done", t_wh)

    full_text = (result.get("text") or "").strip()
    print("STAGE transcript_chars", len(full_text))

    if not full_text:
        raise HTTPException(status_code=500, detail="empty transcript text")

    t_sp = time.time()
    words = pick_keywords(full_text, slots)
    tlog("spacy pick done", t_sp)
    print("STAGE picked", words)

    events_words = build_progressive_events(duration, slots, words)

    ass_path = os.path.join(os.path.dirname(in_path), "list.ass")
    write_ass(ass_path, duration, slots, events_words, sub_primary_hex)

    use_logo = False
    logo_path = os.path.join(os.path.dirname(in_path), "logo.png")
    if logo_enabled:
        if logo_url:
            download_file(logo_url, logo_path)
            use_logo = os.path.exists(logo_path)
        else:
            local_logo = "/app/logo.png"
            if os.path.exists(local_logo):
                shutil.copy(local_logo, logo_path)
                use_logo = os.path.exists(logo_path)

    out_name = f"{output_prefix}{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(os.path.dirname(in_path), out_name)

    ass_f = esc_ff_filter(ass_path)
    logo_f = esc_ff_filter(logo_path)

    fps = int(target_fps) if int(target_fps) > 0 else DEFAULT_FPS
    fps_filter = f"fps={fps}"
    sub_filter = f"subtitles='{ass_f}'"

    # speed: scale down
    scale_filter = f"scale={SCALE_W}:-2"

    t_ff = time.time()
    print("STAGE ffmpeg start")

    if use_logo:
        vf = (
            f"[0:v]{scale_filter},{fps_filter},{sub_filter}[v];"
            f"movie='{logo_f}',scale={LOGO_SCALE_W}:-1[logo];"
            f"[v][logo]overlay=W-w-{LOGO_PAD_X}:{LOGO_PAD_Y}:format=auto"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-filter_complex", vf,
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-r", str(fps),
            "-c:a", "aac", "-b:a", "128k",
            out_path
        ]
    else:
        vf = f"{scale_filter},{fps_filter},{sub_filter}"
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-r", str(fps),
            "-c:a", "aac", "-b:a", "128k",
            out_path
        ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "")[-2500:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {tail}")

    tlog("ffmpeg done", t_ff)

    if not os.path.exists(out_path):
        raise HTTPException(status_code=500, detail="output mp4 missing")

    print("STAGE out_bytes", os.path.getsize(out_path))
    tlog("render total", t0)

    persist_name = persist_output(out_path)
    return persist_name, out_path


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/health")
def health_api():
    return {"ok": True}


@app.get("/api/output/{name}")
def get_output(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="bad name")

    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="missing")

    return FileResponse(path, media_type="video/mp4", filename=name)


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

    t0_all = time.time()

    work = f"/tmp/work_{uuid.uuid4().hex}"
    os.makedirs(work, exist_ok=True)

    try:
        in_path = os.path.join(work, "input.mp4")

        t_dl = time.time()
        download_file(req.video_url, in_path)
        tlog("download", t_dl)

        print("STAGE input_bytes", os.path.getsize(in_path))

        t_pr = time.time()
        duration = get_duration_or_zero(in_path)
        tlog("ffprobe", t_pr)

        if duration <= 0:
            raise HTTPException(status_code=400, detail="ffprobe duration failed")

        with PROCESS_LOCK:
            with TRANSCRIBE_LOCK:
                persist_name, _ = render_video(
                    in_path=in_path,
                    duration=duration,
                    slots=slots,
                    target_fps=int(req.target_fps),
                    sub_primary_hex=req.sub_primary_hex,
                    logo_enabled=bool(req.logo_enabled),
                    logo_url=req.logo_url,
                    output_prefix=req.output_prefix,
                )

        print("STAGE output_name", persist_name)
        tlog("total", t0_all)

        cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
        return JSONResponse(
            {
                "ok": True,
                "output_name": persist_name,
                "output_url": f"/api/output/{persist_name}",
            },
            background=cleanup,
        )

    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"server error: {e}")


@app.post("/api/process")
def process_api(req: ProcessReq):
    return process(req)


@app.post("/api/process_upload")
def process_upload(
    file: UploadFile = File(...),
    slots: int = Form(5),
    target_fps: int = Form(DEFAULT_FPS),
    sub_primary_hex: str = Form("FFFF00"),
    logo_enabled: bool = Form(False),
    logo_url: Optional[str] = Form(None),
    output_prefix: str = Form("ReelFive_"),
):
    ensure_tools()
    if nlp is None:
        raise HTTPException(status_code=500, detail="spaCy not loaded")

    slots = int(slots)
    if slots < 1:
        raise HTTPException(status_code=400, detail="slots must be >= 1")
    if slots > 10:
        raise HTTPException(status_code=400, detail="slots too high, max 10")

    t0_all = time.time()

    work = f"/tmp/work_{uuid.uuid4().hex}"
    os.makedirs(work, exist_ok=True)

    try:
        print("STAGE upload_start", file.filename)

        t_up = time.time()
        in_path = os.path.join(work, "input.mp4")
        save_upload(file, in_path)
        tlog("upload_write", t_up)

        print("STAGE input_bytes", os.path.getsize(in_path))

        t_pr = time.time()
        duration = get_duration_or_zero(in_path)
        tlog("ffprobe", t_pr)

        if duration <= 0:
            raise HTTPException(status_code=400, detail="ffprobe duration failed")

        with PROCESS_LOCK:
            with TRANSCRIBE_LOCK:
                persist_name, _ = render_video(
                    in_path=in_path,
                    duration=duration,
                    slots=slots,
                    target_fps=int(target_fps),
                    sub_primary_hex=sub_primary_hex,
                    logo_enabled=bool(logo_enabled),
                    logo_url=logo_url,
                    output_prefix=output_prefix,
                )

        print("STAGE output_name", persist_name)
        tlog("total", t0_all)

        cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
        return JSONResponse(
            {
                "ok": True,
                "output_name": persist_name,
                "output_url": f"/api/output/{persist_name}",
            },
            background=cleanup,
        )

    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"server error: {e}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




