import asyncio
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel

APP_DIR = Path(__file__).resolve().parent
IN_DIR = Path("/work/in")
OUT_DIR = Path("/work/out")
UPLOAD_DIR = Path("/work/uploads")

OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Twoje mapowanie "ja" -> "jp"
LANG_ALIAS = {
    "ja": "jp",
}

MEDIA_EXT = {".mkv", ".mp4", ".avi", ".mov", ".mp3", ".wav", ".m4a", ".flac", ".webm"}

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

MODEL_CACHE: Dict[str, WhisperModel] = {}
MODEL_LOCK = asyncio.Lock()

def safe_relpath(p: str) -> str:
    # blokujemy path traversal
    p = p.replace("\\", "/").strip()
    if p.startswith("/") or ".." in p.split("/"):
        raise ValueError("Invalid path")
    return p

def get_duration_seconds(path: Path) -> Optional[float]:
    # ffprobe z ffmpeg (w kontenerze)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, check=True
        )
        s = result.stdout.strip()
        return float(s) if s else None
    except Exception:
        return None

def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

async def get_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    key = f"{model_name}|{device}|{compute_type}"
    async with MODEL_LOCK:
        if key in MODEL_CACHE:
            return MODEL_CACHE[key]
        # ładowanie modelu może chwilę potrwać (cache w /hf)
        m = WhisperModel(model_name, device=device, compute_type=compute_type)
        MODEL_CACHE[key] = m
        return m

async def update_job(job_id: str, **fields):
    async with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id].update(fields)

def build_output_name(input_path: Path, detected_lang: str) -> str:
    lang = (detected_lang or "unk").lower()
    lang_tag = LANG_ALIAS.get(lang, lang)
    return f"{input_path.stem}.{lang_tag}.srt"

def run_transcription_sync(loop, job_id: str, input_path: Path, model_name: str, device: str,
                          compute_type: str, vad: bool, beam_size: int):
    duration = get_duration_seconds(input_path)

    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id, status="running", progress=0.0, duration=duration
    ))
    loop.call_soon_threadsafe(asyncio.create_task, update_job(job_id, status="running", progress=0.0, duration=duration))

    # load model (sync stworzymy przez async getter w innym kroku - prościej: osobno)
    # tu model przekazujemy inaczej, ale zostawiamy wątkiem: model będzie w cache w async.
    # obejście: w sync tworzymy nowy model, ale to wolniejsze; więc robimy prosty trick:
    # w tym demo: model w sync tworzymy, bo i tak raz na start.
    # (jeśli chcesz, mogę to przepiąć 1:1 na async cache)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(input_path),
        language=None,          # auto detect
        task="transcribe",      # zawsze oryginał
        vad_filter=vad,
        beam_size=beam_size,
    )

    detected = (info.language or "unk").lower()
    out_name = build_output_name(input_path, detected)
    out_path = OUT_DIR / out_name

    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id,
        detected_language=detected,
        language_probability=float(info.language_probability or 0.0),
        output_name=out_name,
        output_path=str(out_path),
    ))

    try:
        with out_path.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                f.write(f"{i}\n")
                f.write(f"{srt_time(seg.start)} --> {srt_time(seg.end)}\n")
                f.write(seg.text.strip() + "\n\n")

                # progres: segment.end / duration
                if duration and duration > 0:
                    pct = max(0.0, min(99.0, (seg.end / duration) * 100.0))
                    loop.call_soon_threadsafe(asyncio.create_task, update_job(job_id, progress=pct))

        loop.call_soon_threadsafe(asyncio.create_task, update_job(job_id, progress=100.0, status="done"))
    except Exception as e:
        loop.call_soon_threadsafe(asyncio.create_task, update_job(job_id, status="error", error=str(e)))

class StartJobRequest(BaseModel):
    path: str                  # relative to /work/in OR special "upload:<filename>"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    vad: bool = True
    beam_size: int = 5

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/files")
def list_files():
    if not IN_DIR.exists():
        return {"files": []}

    files = []
    # proste listowanie (rekurencja, ale ograniczona sensownie)
    for p in IN_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in MEDIA_EXT:
            rel = p.relative_to(IN_DIR).as_posix()
            files.append(rel)

    # sort dla wygody
    files.sort()
    return {"files": files}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in MEDIA_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # zapisujemy w /work/uploads
    safe_name = re.sub(r"[^\w\[\]\(\)\.\-\s]+", "_", Path(file.filename).name)
    dst = UPLOAD_DIR / safe_name
    with dst.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return {"uploaded": True, "server_path": f"upload:{safe_name}"}

@app.post("/api/jobs")
async def start_job(req: StartJobRequest):
    job_id = uuid.uuid4().hex

    # resolve input path
    if req.path.startswith("upload:"):
        name = req.path.split("upload:", 1)[1]
        input_path = UPLOAD_DIR / name
        display_path = f"(upload) {name}"
    else:
        try:
            rel = safe_relpath(req.path)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")
        input_path = IN_DIR / rel
        display_path = rel

    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {display_path}")

    # init job state
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "input": str(input_path),
            "display_input": display_path,
            "output_name": None,
            "detected_language": None,
            "language_probability": None,
            "error": None,
            "duration": None,
        }

    # start in background thread
    loop = asyncio.get_running_loop()

    asyncio.create_task(asyncio.to_thread(
        run_transcription_sync,
        loop,
        job_id,
        input_path,
        req.model,
        req.device,
        req.compute_type,
        req.vad,
        req.beam_size
    ))

    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    async def gen():
        # SSE: co ~0.5s wysyłamy stan
        while True:
            async with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                yield "event: error\ndata: {\"error\":\"not_found\"}\n\n"
                return

            payload = json.dumps(job, ensure_ascii=False)
            yield f"data: {payload}\n\n"

            if job["status"] in ("done", "error"):
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/jobs/{job_id}/srt")
async def download_srt(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job.get("output_name"):
        raise HTTPException(status_code=400, detail="SRT not ready")

    out_path = OUT_DIR / job["output_name"]
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="SRT file missing on disk")

    return FileResponse(str(out_path), filename=job["output_name"])
