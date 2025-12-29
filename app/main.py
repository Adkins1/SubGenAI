import asyncio
import gc
import json
import logging
import os
import re
import subprocess
import threading
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

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fw-web")

# Twoje mapowanie "ja" -> "jp"
LANG_ALIAS = {
    "ja": "jp",
}

MEDIA_EXT = {".mkv", ".mp4", ".avi", ".mov", ".mp3", ".wav", ".m4a", ".flac", ".webm"}

TRANSLATION_MODEL_NAME = "facebook/nllb-200-3.3B"
NLLB_SRC_LANG = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "pl": "pol_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}
NLLB_TGT_LANG = {
    "pl": "pol_Latn",
}

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

MODEL_CACHE: Dict[str, WhisperModel] = {}
MODEL_LOCK = asyncio.Lock()

TRANSLATION_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
TRANSLATION_MODEL_LOCK = threading.Lock()

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

def cleanup_cuda_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def pick_translation_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        supported = torch.cuda.get_arch_list()
        if arch not in supported:
            logger.warning("Torch does not support GPU arch %s; falling back to CPU", arch)
            return "cpu"
    except Exception:
        logger.warning("Failed to validate CUDA arch; falling back to CPU")
        return "cpu"

    return "cuda"

def is_cuda_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda out of memory" in msg

def build_output_name(input_path: Path, detected_lang: str) -> str:
    lang = (detected_lang or "unk").lower()
    lang_tag = LANG_ALIAS.get(lang, lang)
    return f"{input_path.stem}.{lang_tag}.srt"

def build_output_name_lang(input_path: Path, lang_tag: str) -> str:
    return f"{input_path.stem}.{lang_tag}.srt"

def get_translation_model(device: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    with TRANSLATION_MODEL_LOCK:
        cached = TRANSLATION_MODEL_CACHE.get(device)
        if cached:
            return cached["tokenizer"], cached["model"]

        logger.info("Loading translation model %s on %s", TRANSLATION_MODEL_NAME, device)
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL_NAME,
            torch_dtype=torch_dtype,
        )
        if device == "cuda":
            model = model.to(device)
        model.eval()

        TRANSLATION_MODEL_CACHE[device] = {"tokenizer": tokenizer, "model": model}
        logger.info("Translation model loaded")
        return tokenizer, model

def release_translation_model(device: str):
    with TRANSLATION_MODEL_LOCK:
        cached = TRANSLATION_MODEL_CACHE.pop(device, None)
    if cached:
        try:
            del cached
        finally:
            cleanup_cuda_memory()

def translate_segments(segments, src_lang: str, tgt_lang: str, job_id: str, loop, device: str):
    import torch

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    else:
        logger.info("Translation running on CPU")

    tokenizer, model = get_translation_model(device)
    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id, translation_status="running", translation_device=device
    ))
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_lang)
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_token_id is None:
        raise ValueError(f"Unsupported target language: {tgt_lang}")

    tokenizer.src_lang = src_lang

    translated = []
    total = len(segments)
    for idx, seg in enumerate(segments, start=1):
        text = seg["text"].strip()
        if text:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=256,
                )
            translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        else:
            translated_text = ""

        translated.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_text,
        })

        if total and (idx % 5 == 0 or idx == total):
            pct = (idx / total) * 100.0
            loop.call_soon_threadsafe(asyncio.create_task, update_job(
                job_id, translation_progress=pct
            ))

    return translated

def run_transcription_sync(loop, job_id: str, input_path: Path, model_name: str, device: str,
                          compute_type: str, vad: bool, beam_size: int,
                          translate_to: Optional[str]):
    duration = get_duration_seconds(input_path)

    logger.info("Job %s: start transcription (%s)", job_id, input_path)
    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id,
        status="running",
        transcription_status="running",
        progress=0.0,
        duration=duration,
    ))

    # load model (sync stworzymy przez async getter w innym kroku - pro'ciej: osobno)
    # tu model przekazujemy inaczej, ale zostawiamy w atkiem: model bedzie w cache w async.
    # obej cie: w sync tworzymy nowy model, ale to wolniejsze; wiec robimy prosty trick:
    # w tym demo: model w sync tworzymy, bo i tak raz na start.
    # (jesli chcesz, moge to przepisac 1:1 na async cache)
    def transcribe_attempt(attempt_compute_type: str):
        logger.info("Job %s: loading whisper model (compute_type=%s)", job_id, attempt_compute_type)
        model = WhisperModel(model_name, device=device, compute_type=attempt_compute_type)
        try:
            segments, info = model.transcribe(
                str(input_path),
                language=None,          # auto detect
                task="transcribe",      # zawsze oryginal
                vad_filter=vad,
                beam_size=beam_size,
            )

            segments_list = []
            for seg in segments:
                segments_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                })

                # progress: segment.end / duration
                if duration and duration > 0:
                    pct = max(0.0, min(99.0, (seg.end / duration) * 100.0))
                    loop.call_soon_threadsafe(asyncio.create_task, update_job(job_id, progress=pct))

            return segments_list, info
        finally:
            del model
            cleanup_cuda_memory()

    compute_types = [compute_type]
    if device == "cuda" and compute_type != "int8_float16":
        compute_types.append("int8_float16")

    segments_list = None
    info = None
    used_compute_type = compute_type
    try:
        for idx, attempt_compute_type in enumerate(compute_types):
            if idx > 0:
                loop.call_soon_threadsafe(asyncio.create_task, update_job(
                    job_id,
                    transcription_status="retry_int8",
                    progress=0.0,
                ))
                logger.warning(
                    "Job %s: retrying transcription with compute_type=%s",
                    job_id,
                    attempt_compute_type,
                )
            try:
                segments_list, info = transcribe_attempt(attempt_compute_type)
                used_compute_type = attempt_compute_type
                break
            except RuntimeError as e:
                if is_cuda_oom(e) and idx < len(compute_types) - 1:
                    logger.warning("Job %s: CUDA OOM with compute_type=%s", job_id, attempt_compute_type)
                    continue
                raise

        if segments_list is None or info is None:
            raise RuntimeError("Transcription failed")
    except Exception as e:
        logger.exception("Job %s: transcription failed", job_id)
        error_msg = str(e)
        if is_cuda_oom(e):
            error_msg = "CUDA OOM during transcription. Try compute_type=int8_float16 or a smaller model."
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            transcription_status="error",
            error=error_msg,
        ))
        return

    detected = (info.language or "unk").lower()
    out_name = build_output_name(input_path, detected)
    out_path = OUT_DIR / out_name

    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id,
        detected_language=detected,
        language_probability=float(info.language_probability or 0.0),
        output_name=out_name,
        output_path=str(out_path),
        transcription_compute_type=used_compute_type,
    ))

    try:
        with out_path.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(segments_list, start=1):
                f.write(f"{i}\n")
                f.write(f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n")
                f.write(seg["text"].strip() + "\n\n")

        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            progress=100.0,
            transcription_status="done",
        ))
        logger.info("Job %s: transcription done (lang=%s, output=%s)", job_id, detected, out_name)
    except Exception as e:
        logger.exception("Job %s: transcription write failed", job_id)
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            transcription_status="error",
            error=str(e),
        ))
        return
    if not translate_to:
        logger.info("Job %s: translation skipped", job_id)
        loop.call_soon_threadsafe(asyncio.create_task, update_job(job_id, status="done"))
        return

    translate_to = translate_to.lower()
    src_lang = NLLB_SRC_LANG.get(detected)
    tgt_lang = NLLB_TGT_LANG.get(translate_to)
    if not src_lang or not tgt_lang:
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            translation_status="error",
            translation_error="Unsupported language for translation",
        ))
        return

    translation_name = build_output_name_lang(input_path, translate_to)
    translation_path = OUT_DIR / translation_name
    translation_device = pick_translation_device()
    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id,
        translation_status="loading_model",
        translation_progress=0.0,
        translation_output_name=translation_name,
        translation_output_path=str(translation_path),
        translation_language=translate_to,
        translation_device=translation_device,
    ))
    logger.info("Job %s: translation start (%s -> %s)", job_id, detected, translate_to)

    try:
        translated_segments = translate_segments(
            segments_list,
            src_lang,
            tgt_lang,
            job_id,
            loop,
            translation_device,
        )
        with translation_path.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(translated_segments, start=1):
                f.write(f"{i}\n")
                f.write(f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n")
                f.write(seg["text"].strip() + "\n\n")

        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            translation_status="done",
            translation_progress=100.0,
            status="done",
        ))
        logger.info("Job %s: translation done (output=%s)", job_id, translation_name)
    except Exception as e:
        logger.exception("Job %s: translation failed", job_id)
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            translation_status="error",
            translation_error=str(e),
        ))
    finally:
        if translation_device == "cuda":
            release_translation_model("cuda")

class StartJobRequest(BaseModel):
    path: str                  # relative to /work/in OR special "upload:<filename>"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    vad: bool = True
    beam_size: int = 5
    translate_to: Optional[str] = None

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
            "transcription_status": "queued",
            "translation_status": "pending" if req.translate_to else "skipped",
            "progress": 0.0,
            "translation_progress": 0.0,
            "input": str(input_path),
            "display_input": display_path,
            "output_name": None,
            "translation_output_name": None,
            "translation_output_path": None,
            "translation_language": req.translate_to,
            "transcription_compute_type": req.compute_type,
            "translation_device": None,
            "detected_language": None,
            "language_probability": None,
            "error": None,
            "translation_error": None,
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
        req.beam_size,
        req.translate_to,
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
    ready = job.get("transcription_status", job.get("status")) == "done"
    if not ready or not job.get("output_name"):
        raise HTTPException(status_code=400, detail="SRT not ready")

    out_path = OUT_DIR / job["output_name"]
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="SRT file missing on disk")

    return FileResponse(str(out_path), filename=job["output_name"])

@app.get("/api/jobs/{job_id}/srt-translated")
async def download_srt_translated(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("translation_status") != "done" or not job.get("translation_output_name"):
        raise HTTPException(status_code=400, detail="Translated SRT not ready")

    out_path = OUT_DIR / job["translation_output_name"]
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Translated SRT missing on disk")

    return FileResponse(str(out_path), filename=job["translation_output_name"])
