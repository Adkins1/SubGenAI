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
LANG_ALIAS_REV = {v: k for k, v in LANG_ALIAS.items()}

MEDIA_EXT = {".mkv", ".mp4", ".avi", ".mov", ".mp3", ".wav", ".m4a", ".flac", ".webm"}

TRANSLATION_MODEL_NAME = "facebook/nllb-200-3.3B"
WHISPER_LANGUAGES = {
    "af": "afrikaans",
    "am": "amharic",
    "ar": "arabic",
    "as": "assamese",
    "az": "azerbaijani",
    "ba": "bashkir",
    "be": "belarusian",
    "bg": "bulgarian",
    "bn": "bengali",
    "bo": "tibetan",
    "br": "breton",
    "bs": "bosnian",
    "ca": "catalan",
    "cs": "czech",
    "cy": "welsh",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "eu": "basque",
    "fa": "persian",
    "fi": "finnish",
    "fo": "faroese",
    "fr": "french",
    "gl": "galician",
    "gu": "gujarati",
    "ha": "hausa",
    "haw": "hawaiian",
    "he": "hebrew",
    "hi": "hindi",
    "hr": "croatian",
    "ht": "haitian creole",
    "hu": "hungarian",
    "hy": "armenian",
    "id": "indonesian",
    "is": "icelandic",
    "it": "italian",
    "ja": "japanese",
    "jw": "javanese",
    "ka": "georgian",
    "kk": "kazakh",
    "km": "khmer",
    "kn": "kannada",
    "ko": "korean",
    "la": "latin",
    "lb": "luxembourgish",
    "ln": "lingala",
    "lo": "lao",
    "lt": "lithuanian",
    "lv": "latvian",
    "mg": "malagasy",
    "mi": "maori",
    "mk": "macedonian",
    "ml": "malayalam",
    "mn": "mongolian",
    "mr": "marathi",
    "ms": "malay",
    "mt": "maltese",
    "my": "myanmar",
    "ne": "nepali",
    "nl": "dutch",
    "nn": "norwegian nynorsk",
    "no": "norwegian",
    "oc": "occitan",
    "pa": "punjabi",
    "pl": "polish",
    "ps": "pashto",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sa": "sanskrit",
    "sd": "sindhi",
    "si": "sinhala",
    "sk": "slovak",
    "sl": "slovenian",
    "sn": "shona",
    "so": "somali",
    "sq": "albanian",
    "sr": "serbian",
    "su": "sundanese",
    "sv": "swedish",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "tg": "tajik",
    "th": "thai",
    "tk": "turkmen",
    "tl": "tagalog",
    "tr": "turkish",
    "tt": "tatar",
    "uk": "ukrainian",
    "ur": "urdu",
    "uz": "uzbek",
    "vi": "vietnamese",
    "yi": "yiddish",
    "yo": "yoruba",
    "yue": "cantonese",
    "zh": "chinese",
}
WHISPER_LANG_CODES = set(WHISPER_LANGUAGES.keys())
NLLB_LANG_MAP = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "as": "asm_Beng",
    "az": "azj_Latn",
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bo": "bod_Tibt",
    "br": "bre_Latn",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fo": "fao_Latn",
    "fr": "fra_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "haw": "haw_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "ht": "hat_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "jw": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "la": "lat_Latn",
    "lb": "ltz_Latn",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mg": "mlg_Latn",
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "ne": "nep_Deva",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",
    "oc": "oci_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "tk": "tuk_Latn",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "tt": "tat_Cyrl",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "yi": "yid_Hebr",
    "yo": "yor_Latn",
    "yue": "yue_Hant",
    "zh": "zho_Hans",
}
NLLB_LANG_CODES = set(NLLB_LANG_MAP.keys())
KNOWN_LANG_TAGS = set(LANG_ALIAS.keys()) | set(LANG_ALIAS.values()) | set(NLLB_LANG_MAP.keys())

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

MODEL_CACHE: Dict[str, WhisperModel] = {}
MODEL_LOCK = asyncio.Lock()

TRANSLATION_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
TRANSLATION_MODEL_LOCK = threading.Lock()

JOB_CANCEL_EVENTS: Dict[str, threading.Event] = {}
JOB_CANCEL_LOCK = threading.Lock()

class JobCancelled(Exception):
    pass

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

def create_cancel_event(job_id: str) -> threading.Event:
    ev = threading.Event()
    with JOB_CANCEL_LOCK:
        JOB_CANCEL_EVENTS[job_id] = ev
    return ev

def get_cancel_event(job_id: str) -> Optional[threading.Event]:
    with JOB_CANCEL_LOCK:
        return JOB_CANCEL_EVENTS.get(job_id)

def clear_cancel_event(job_id: str):
    with JOB_CANCEL_LOCK:
        JOB_CANCEL_EVENTS.pop(job_id, None)

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
            logger.warning(
                "Torch does not support GPU arch %s; falling back to CPU. "
                "Rebuild with TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 or /cu130.",
                arch,
            )
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

def parse_srt_time(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0)

def parse_srt(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    segments = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.isdigit():
            i += 1
            if i >= len(lines):
                break
            line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue
        start_str, end_str = [p.strip() for p in line.split("-->", 1)]
        try:
            start = parse_srt_time(start_str)
            end = parse_srt_time(end_str)
        except Exception:
            i += 1
            continue
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].rstrip())
            i += 1
        text_joined = " ".join(t.strip() for t in text_lines if t.strip())
        segments.append({"start": start, "end": end, "text": text_joined})
    return segments

def infer_lang_from_srt_path(path: Path) -> Optional[str]:
    parts = path.stem.split(".")
    if len(parts) < 2:
        return None
    candidate = parts[-1].lower()
    candidate = LANG_ALIAS_REV.get(candidate, candidate)
    return candidate if candidate in NLLB_LANG_MAP else None

def build_translation_output_name(srt_path: Path, lang_tag: str) -> str:
    base = srt_path.stem
    parts = base.split(".")
    if len(parts) > 1 and parts[-1].lower() in KNOWN_LANG_TAGS:
        base = ".".join(parts[:-1])
    return f"{base}.{lang_tag}.srt"

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

def translate_segments(segments, src_lang: str, tgt_lang: str, job_id: str, loop, device: str,
                       cancel_event: Optional[threading.Event]):
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
        if cancel_event and cancel_event.is_set():
            raise JobCancelled("Translation cancelled")
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
                          language: Optional[str]):
    duration = get_duration_seconds(input_path)

    logger.info("Job %s: start transcription (%s)", job_id, input_path)
    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id,
        status="running",
        transcription_status="running",
        progress=0.0,
        duration=duration,
    ))
    cancel_event = get_cancel_event(job_id)

    # load model (sync stworzymy przez async getter w innym kroku - pro'ciej: osobno)
    # tu model przekazujemy inaczej, ale zostawiamy w atkiem: model bedzie w cache w async.
    # obej cie: w sync tworzymy nowy model, ale to wolniejsze; wiec robimy prosty trick:
    # w tym demo: model w sync tworzymy, bo i tak raz na start.
    # (jesli chcesz, moge to przepisac 1:1 na async cache)
    def transcribe_attempt(attempt_compute_type: str):
        logger.info("Job %s: loading whisper model (compute_type=%s)", job_id, attempt_compute_type)
        model = WhisperModel(model_name, device=device, compute_type=attempt_compute_type)
        try:
            if cancel_event and cancel_event.is_set():
                raise JobCancelled("Transcription cancelled")
            segments, info = model.transcribe(
                str(input_path),
                language=language,      # None == auto detect
                task="transcribe",      # zawsze oryginal
                vad_filter=vad,
                beam_size=beam_size,
            )

            segments_list = []
            for seg in segments:
                if cancel_event and cancel_event.is_set():
                    raise JobCancelled("Transcription cancelled")
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
    except JobCancelled:
        logger.info("Job %s: transcription cancelled", job_id)
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="cancelled",
            transcription_status="cancelled",
        ))
        clear_cancel_event(job_id)
        return
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
        clear_cancel_event(job_id)
        return

    detected = (info.language or language or "unk").lower()
    out_name = build_output_name(input_path, detected)
    out_path = input_path.parent / out_name

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
            status="done",
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
        clear_cancel_event(job_id)
        return
    clear_cancel_event(job_id)

def run_translation_sync(loop, job_id: str, srt_path: Path, translate_to: str):
    translate_to = translate_to.lower()
    cancel_event = get_cancel_event(job_id)

    loop.call_soon_threadsafe(asyncio.create_task, update_job(
        job_id,
        status="running",
        translation_status="loading_srt",
        translation_progress=0.0,
    ))

    try:
        segments_list = parse_srt(srt_path)
    except Exception as e:
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            translation_status="error",
            translation_error=f"Failed to parse SRT: {e}",
        ))
        clear_cancel_event(job_id)
        return

    if cancel_event and cancel_event.is_set():
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="cancelled",
            translation_status="cancelled",
        ))
        clear_cancel_event(job_id)
        return

    if not segments_list:
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            translation_status="error",
            translation_error="No segments found in SRT",
        ))
        clear_cancel_event(job_id)
        return

    src_lang_key = infer_lang_from_srt_path(srt_path)
    if not src_lang_key:
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            translation_status="error",
            translation_error="Cannot infer source language from SRT filename",
        ))
        clear_cancel_event(job_id)
        return

    src_lang = NLLB_LANG_MAP.get(src_lang_key)
    tgt_lang = NLLB_LANG_MAP.get(translate_to)
    if not src_lang or not tgt_lang:
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="error",
            translation_status="error",
            translation_error="Unsupported language for translation",
        ))
        clear_cancel_event(job_id)
        return

    translation_name = build_translation_output_name(srt_path, translate_to)
    translation_path = srt_path.parent / translation_name
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
    logger.info(
        "Job %s: translation start (%s -> %s, device=%s)",
        job_id,
        src_lang_key,
        translate_to,
        translation_device,
    )

    try:
        translated_segments = translate_segments(
            segments_list,
            src_lang,
            tgt_lang,
            job_id,
            loop,
            translation_device,
            cancel_event,
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
    except JobCancelled:
        loop.call_soon_threadsafe(asyncio.create_task, update_job(
            job_id,
            status="cancelled",
            translation_status="cancelled",
        ))
        logger.info("Job %s: translation cancelled", job_id)
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
        clear_cancel_event(job_id)
class StartTranscriptionRequest(BaseModel):
    path: str                  # relative to /work/in OR special "upload:<filename>"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    vad: bool = True
    beam_size: int = 5
    language: str = "auto"

class StartTranslationRequest(BaseModel):
    srt_path: str              # prefixed with "in:" or "upload:"
    translate_to: str = "pl"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/files")
def list_files(dir: Optional[str] = None):
    if not IN_DIR.exists():
        return {"dir": "", "parent": None, "dirs": [], "files": []}

    rel = ""
    if dir:
        try:
            rel = safe_relpath(dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")

    base = IN_DIR / rel if rel else IN_DIR
    if not base.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="Not a folder")

    dirs = []
    files = []
    for p in base.iterdir():
        if p.is_dir():
            dirs.append(p.name)
        elif p.is_file() and p.suffix.lower() in MEDIA_EXT:
            files.append(p.name)

    dirs.sort()
    files.sort()

    parent = None
    if rel:
        parent = Path(rel).parent.as_posix()
        if parent == ".":
            parent = ""

    return {"dir": rel, "parent": parent, "dirs": dirs, "files": files}

@app.get("/api/languages")
def list_languages():
    whisper = [{"code": code, "name": WHISPER_LANGUAGES[code]} for code in sorted(WHISPER_LANGUAGES)]
    translation = [
        {"code": code, "name": WHISPER_LANGUAGES.get(code, code)}
        for code in sorted(NLLB_LANG_MAP)
    ]
    return {"whisper": whisper, "translation": translation}

@app.get("/api/srt-files")
def list_srt_files(dir: Optional[str] = None):
    if not IN_DIR.exists():
        return {"dir": "", "files": []}

    rel = ""
    if dir:
        try:
            rel = safe_relpath(dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")

    base = IN_DIR / rel if rel else IN_DIR
    if not base.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="Not a folder")

    files = []
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() == ".srt":
            files.append(p.name)

    files.sort()
    return {"dir": rel, "files": files}

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
async def start_transcription(req: StartTranscriptionRequest):
    job_id = uuid.uuid4().hex
    language = (req.language or "auto").lower()
    if language == "auto":
        language = None
    elif language not in WHISPER_LANG_CODES:
        raise HTTPException(status_code=400, detail="Unsupported language")

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
    create_cancel_event(job_id)
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "type": "transcription",
            "status": "queued",
            "transcription_status": "queued",
            "translation_status": "skipped",
            "progress": 0.0,
            "input": str(input_path),
            "display_input": display_path,
            "output_name": None,
            "output_path": None,
            "transcription_compute_type": req.compute_type,
            "requested_language": language or "auto",
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
        req.beam_size,
        language,
    ))

    return {"job_id": job_id}

@app.post("/api/translations")
async def start_translation(req: StartTranslationRequest):
    job_id = uuid.uuid4().hex
    if req.srt_path.startswith("in:"):
        try:
            rel = safe_relpath(req.srt_path.split("in:", 1)[1])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid SRT path")
        input_path = IN_DIR / rel
        display_input = f"in:{rel}"
    elif req.srt_path.startswith("upload:"):
        name = req.srt_path.split("upload:", 1)[1]
        try:
            rel = safe_relpath(name)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid SRT path")
        input_path = UPLOAD_DIR / rel
        display_input = f"upload:{rel}"
    else:
        raise HTTPException(status_code=400, detail="Invalid SRT path")
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"SRT not found: {rel}")
    if input_path.suffix.lower() != ".srt":
        raise HTTPException(status_code=400, detail="Selected file is not an SRT")

    create_cancel_event(job_id)
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "type": "translation",
            "status": "queued",
            "transcription_status": "skipped",
            "translation_status": "queued",
            "progress": 0.0,
            "translation_progress": 0.0,
            "input": str(input_path),
            "display_input": display_input,
            "translation_output_name": None,
            "translation_output_path": None,
            "translation_language": req.translate_to,
            "translation_device": None,
            "translation_error": None,
        }

    loop = asyncio.get_running_loop()
    asyncio.create_task(asyncio.to_thread(
        run_translation_sync,
        loop,
        job_id,
        input_path,
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

            if job["status"] in ("done", "error", "cancelled"):
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    ev = get_cancel_event(job_id)
    if ev:
        ev.set()

    fields = {"status": "cancelling"}
    if job.get("type") == "translation":
        fields["translation_status"] = "stopping"
    else:
        fields["transcription_status"] = "stopping"
    await update_job(job_id, **fields)
    return {"stopping": True}

@app.get("/api/jobs/{job_id}/srt")
async def download_srt(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ready = job.get("transcription_status", job.get("status")) == "done"
    if not ready or not job.get("output_name"):
        raise HTTPException(status_code=400, detail="SRT not ready")

    output_path = job.get("output_path")
    if not output_path:
        raise HTTPException(status_code=400, detail="SRT not ready")
    out_path = Path(output_path)
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="SRT file missing on disk")

    filename = job.get("output_name") or out_path.name
    return FileResponse(str(out_path), filename=filename)

@app.get("/api/jobs/{job_id}/srt-translated")
async def download_srt_translated(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("translation_status") != "done" or not job.get("translation_output_name"):
        raise HTTPException(status_code=400, detail="Translated SRT not ready")

    output_path = job.get("translation_output_path")
    if not output_path:
        raise HTTPException(status_code=400, detail="Translated SRT not ready")
    out_path = Path(output_path)
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Translated SRT missing on disk")

    filename = job.get("translation_output_name") or out_path.name
    return FileResponse(str(out_path), filename=filename)
