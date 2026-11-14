from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import edge_tts
import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("STUDIO_DATA_DIR", ROOT / ".studio-data")).resolve()
FILES = DATA / "files"
PROJECTS = DATA / "projects"
UPLOAD_LIMIT = int(os.getenv("STUDIO_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DEMUCS_MODEL = os.getenv("STUDIO_DEMUCS_MODEL", "83fc094f")
DEMUCS_SHIFTS = int(os.getenv("STUDIO_DEMUCS_SHIFTS", "0"))
MALE_VOICE = os.getenv("EDGE_TTS_MALE_VOICE", "km-KH-PisethNeural")
FEMALE_VOICE = os.getenv("EDGE_TTS_FEMALE_VOICE", "km-KH-SreymomNeural")

for directory in (FILES, PROJECTS):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="KH Drama Studio API", version="1.0.0")
app.mount("/files", StaticFiles(directory=FILES), name="files")

_operations: dict[str, dict[str, Any]] = {}
_exports: dict[str, dict[str, Any]] = {}
_bgm: dict[str, dict[str, Any]] = {}
_operation_lock = threading.Lock()
_whisper_model: Any = None
_whisper_lock = threading.Lock()

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
}


class TranslateRequest(BaseModel):
    lines: list[str] = Field(min_length=1, max_length=500)
    progress_id: str | None = None


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    char: str = ""
    char_id: str = ""
    video: str = ""
    idx: int = 0
    gender: str = ""
    speed: float = Field(default=0, ge=0, le=1)
    fit_to: float = Field(default=0, ge=0, le=60)


class DetectRequest(BaseModel):
    id: str
    segments: list[dict[str, Any]]
    use_audio: bool = True
    voicebook: bool = True
    reassign: bool = True
    progress_id: str | None = None


class ScanRequest(BaseModel):
    id: str
    segments: list[dict[str, Any]]
    progress_id: str | None = None


class AssignRequest(BaseModel):
    id: str
    overrides: dict[str, str]


class FixRequest(BaseModel):
    id: str
    segments: list[dict[str, Any]]


class ExportRequest(BaseModel):
    id: str
    segments: list[dict[str, Any]]
    subs: bool = True
    bgm: str = ""
    quality: str = ""
    resolution: str = "1080p"
    layers: list[dict[str, Any]] = Field(default_factory=list)
    boxes: list[dict[str, Any]] = Field(default_factory=list)


class TTSExportRequest(BaseModel):
    kind: str = "zip"
    segments: list[dict[str, Any]]


def _safe_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid project id")
    return value


def _project_path(pid: str) -> Path:
    return PROJECTS / f"{_safe_id(pid)}.json"


def _load_project(pid: str) -> dict[str, Any]:
    path = _project_path(pid)
    if not path.exists():
        return {"pid": pid, "video": "", "segments": [], "characters": [], "final": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read project metadata: {exc}") from exc


def _save_project(pid: str, data: dict[str, Any]) -> None:
    path = _project_path(pid)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _operation(progress_id: str | None, **state: Any) -> None:
    if not progress_id:
        return
    with _operation_lock:
        _operations.setdefault(progress_id, {}).update(state, updated_at=time.time())


# Keep references to background tasks so they are not garbage-collected mid-run.
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro: Any) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _video_path(pid: str) -> Path:
    pid = _safe_id(pid)
    matches = [p for p in FILES.glob(f"{pid}.*") if p.is_file()]
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded media not found")
    return matches[0]


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise HTTPException(status_code=503, detail="FFmpeg is required; install it and restart the backend")
    return path


_ffmpeg_filter_cache: dict[str, bool] = {}


def _ffprobe_video_size(path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return (0, 0)
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        width, height = result.stdout.strip().split(",")
        return max(0, int(width)), max(0, int(height))
    except ValueError:
        return (0, 0)


def _ffmpeg_has_filter(name: str) -> bool:
    """True when the installed FFmpeg build ships the named filter (e.g. libass)."""
    if name not in _ffmpeg_filter_cache:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-filters"],
                capture_output=True, text=True, check=False, timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        _ffmpeg_filter_cache[name] = re.search(rf"\b{name}\b", result.stdout or "") is not None
    return _ffmpeg_filter_cache[name]


def _ffprobe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 0.0


def _get_whisper() -> Any:
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Transcription dependency missing. Install backend requirements and retry.",
                ) from exc
            try:
                _whisper_model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Could not load Whisper model: {exc}") from exc
    return _whisper_model


def _segment_f0(audio: Any, sample_rate: int, start: float, end: float) -> float | None:
    """Median fundamental frequency (pitch) of one segment via autocorrelation."""
    seg = audio[max(0, int(start * sample_rate)):max(0, int(end * sample_rate))]
    frame_size, hop = int(sample_rate * 0.04), int(sample_rate * 0.01)
    lag_lo, lag_hi = int(sample_rate / 400), int(sample_rate / 60)
    if len(seg) < frame_size or lag_hi <= lag_lo:
        return None
    f0s: list[float] = []
    for offset in range(0, len(seg) - frame_size, hop):
        frame = seg[offset:offset + frame_size]
        if float(np.sqrt(np.mean(frame ** 2))) < 0.012:  # skip silence
            continue
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1:]
        corr = corr / (corr[0] + 1e-9)
        if lag_hi >= len(corr):
            break
        lag = lag_lo + int(np.argmax(corr[lag_lo:lag_hi]))
        if corr[lag] < 0.35:
            continue
        f0s.append(sample_rate / lag)
    if not f0s:
        return None
    return float(np.median(f0s))


def _diarize_by_pitch(media: Path, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimate a speaker per segment from audio pitch (male/female separation).

    This is honest, dependency-free diarization: it reliably separates male and
    female speakers but two same-gender speakers may share one voice.
    Returns one character dict per segment, in segment order.
    """
    import wave
    ffmpeg = _ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "mono.wav"
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(media),
             "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode or not wav_path.is_file():
            raise RuntimeError(result.stderr[-300:] or "ffmpeg could not decode the audio")
        with wave.open(str(wav_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    if channels > 1:
        audio = audio[::channels].copy()
    audio = audio.astype(np.float64) / 32768.0

    pitches = [
        _segment_f0(audio, sample_rate, float(s.get("start", 0)), float(s.get("end", 0)))
        for s in segments
    ]
    pitches = [
        _segment_f0(audio, sample_rate, float(s.get("start", 0)), float(s.get("end", 0)))
        for s in segments
    ]
    # Two-pass clustering: split all pitches on natural breaks (relative gap),
    # then assign segments to the nearest cluster. The previous greedy
    # running-mean first-fit drifted and mislabelled low-pitch female voices.
    import statistics
    valid = sorted(p for p in pitches if p is not None)
    centers: list[float] = []
    if valid:
        groups: list[list[float]] = [[valid[0]]]
        for pitch in valid[1:]:
            if pitch / groups[-1][-1] > 1.25:  # natural break in the pitch range
                groups.append([pitch])
            else:
                groups[-1].append(pitch)
        centers = [statistics.median(g) for g in groups]
        # Cap at 6 voices: merge the two closest clusters until small enough.
        while len(centers) > 6:
            pair = min(
                ((i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))),
                key=lambda ij: abs(ij[0] - ij[1]) / max(centers[ij[0]], centers[ij[1]]),
            )
            i, j = pair
            centers[i] = (centers[i] + centers[j]) / 2
            del centers[j]
    else:
        centers = [200.0]

    assignment: list[int] = []
    for pitch in pitches:
        reference = pitch if pitch is not None else (
            centers[assignment[-1]] if assignment else centers[0]
        )
        nearest = min(range(len(centers)), key=lambda i: abs(reference - centers[i]) / centers[i])
        assignment.append(nearest)

    order: list[int] = []
    for spk_index in assignment:
        if spk_index not in order:
            order.append(spk_index)
    per_segment: list[dict[str, Any]] = []
    for spk_index in assignment:
        number = order.index(spk_index) + 1
        f0 = centers[spk_index]
        is_male = f0 < 165.0
        per_segment.append({
            "id": f"speaker_{number}",
            "name": f"Speaker {number} ({'male' if is_male else 'female'})",
            "voice": "male" if is_male else "female",
            "voiceName": MALE_VOICE if is_male else FEMALE_VOICE,
            "pitch": round(f0, 1),
        })
    return per_segment


def _gemini_json(prompt: str) -> Any:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Set GEMINI_API_KEY to enable translation")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(GEMINI_MODEL)}:generateContent"
    try:
        response = httpx.post(
            url,
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except HTTPException:
        raise
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        detail = "Gemini request failed"
        if isinstance(exc, httpx.HTTPStatusError):
            detail += f" (HTTP {exc.response.status_code}): {exc.response.text[:500]}"
        else:
            detail += f": {exc}"
        raise HTTPException(status_code=502, detail=detail) from exc


async def _free_translate(lines: list[str], progress_id: str | None = None) -> list[str]:
    """No-key fallback: the free Google Translate endpoint (dict-chrome-ex client).

    Lines are sent in batches (multiple `q` params) to keep the request count low,
    with retries and mild concurrency to stay under the rate limit.
    """
    results: list[str] = [""] * len(lines)
    indexes = [i for i, line in enumerate(lines) if line.strip()]
    BATCH = 25
    groups = [indexes[i:i + BATCH] for i in range(0, len(indexes), BATCH)]
    done = 0
    async with httpx.AsyncClient(timeout=30) as client:
        sem = asyncio.Semaphore(2)

        async def one_batch(group: list[int]) -> None:
            nonlocal done
            params = [("client", "dict-chrome-ex"), ("sl", "auto"), ("tl", "km")]
            params += [("q", lines[i]) for i in group]
            for attempt in range(3):
                try:
                    response = await client.get(
                        "https://clients5.google.com/translate_a/t", params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, list) or len(data) != len(group):
                        raise ValueError("unexpected translation response shape")
                    texts = []
                    for item in data:
                        if isinstance(item, list):
                            item = item[0] if item and isinstance(item[0], str) else ""
                        texts.append(item if isinstance(item, str) else "")
                    for i, text in zip(group, texts):
                        results[i] = text.strip()
                    done += len(group)
                    _operation(progress_id, state="running", phase="Free translation",
                               completed=done, total=len(indexes))
                    return
                except (httpx.HTTPError, ValueError) as exc:
                    if attempt == 2:
                        raise RuntimeError(f"Free translation failed: {exc}") from exc
                    await asyncio.sleep(1.0 + attempt)

        await asyncio.gather(*(one_batch(g) for g in groups))
    return results


async def _save_upload(
    file: UploadFile,
    stem: str | None = None,
    allowed: set[str] | None = None,
) -> tuple[str, Path]:
    original = Path(file.filename or "upload.bin")
    suffix = original.suffix.lower()
    if suffix not in (allowed or ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=415, detail=f"Unsupported media extension: {suffix or '(none)'}")
    pid = stem or uuid.uuid4().hex
    destination = FILES / f"{pid}{suffix}"
    total = 0
    try:
        with destination.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > UPLOAD_LIMIT:
                    destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="Upload exceeds STUDIO_MAX_UPLOAD_BYTES")
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return pid, destination


def _public_url(path: Path) -> str:
    return "/files/" + quote(path.name)


@app.get("/")
def home() -> FileResponse:
    page = ROOT / "studio.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="studio.html is missing")
    return FileResponse(page, media_type="text/html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, str]:
    pid, path = await _save_upload(file)
    meta = _load_project(pid)
    meta.update(pid=pid, video=path.name, media_url=_public_url(path))
    _save_project(pid, meta)
    return {"id": pid, "url": _public_url(path), "filename": path.name}


@app.post("/api/transcribe")
async def transcribe(body: dict[str, Any]) -> list[dict[str, Any]]:
    pid = _safe_id(str(body.get("id", "")))
    progress_id = body.get("progress_id")
    media = _video_path(pid)
    _operation(progress_id, state="running", phase="Loading Whisper", completed=0, total=1)

    def run() -> list[dict[str, Any]]:
        try:
            model = _get_whisper()
            segments, info = model.transcribe(str(media), beam_size=5, vad_filter=True)
            result = []
            for segment in segments:
                result.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "zh": str(segment.text).strip(),
                    "char": "",
                })
                _operation(progress_id, state="running", phase="Transcribing", completed=len(result), total=0)
            _operation(
                progress_id, state="done", phase=f"Whisper {info.language}",
                completed=len(result), total=len(result),
            )
            meta = _load_project(pid)
            meta["segments"] = result
            _save_project(pid, meta)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            _operation(progress_id, state="error", phase="Transcription failed", error=str(exc))
            raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc

    return await run_in_threadpool(run)


@app.post("/api/translate")
async def translate(body: TranslateRequest) -> list[dict[str, str]]:
    lines = [line.strip() for line in body.lines]
    if any(len(line) > 5000 for line in lines):
        raise HTTPException(status_code=413, detail="A transcript line exceeds 5000 characters")
    provider = "Gemini" if GEMINI_API_KEY else "Free Google Translate"
    _operation(body.progress_id, state="running", phase=f"{provider} translation", completed=0, total=len(lines))
    try:
        if GEMINI_API_KEY:
            answer = _gemini_json(
                "Translate each input line into natural, concise spoken Khmer. "
                "Preserve meaning and order. Return JSON only as an array of objects with a `km` string. "
                "Do not add commentary or merge lines.\nINPUT:\n"
                + json.dumps(lines, ensure_ascii=False)
            )
            if not isinstance(answer, list) or len(answer) != len(lines):
                raise ValueError("Gemini returned an invalid translation list")
            result = []
            for item in answer:
                text = item.get("km", item.get("text", "")) if isinstance(item, dict) else item
                if not isinstance(text, str):
                    raise ValueError("Gemini returned a non-text translation")
                result.append({"km": text.strip()})
        else:
            texts = await _free_translate(lines, body.progress_id)
            result = [{"km": t} for t in texts]
    except Exception as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        _operation(body.progress_id, state="error", phase="Translation failed", error=detail)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=detail) from exc
    _operation(body.progress_id, state="done", phase="Translation complete", completed=len(lines), total=len(lines))
    return result


@app.post("/api/tts")
async def tts(body: TTSRequest) -> dict[str, Any]:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be blank")
    voice = MALE_VOICE if body.gender == "male" or "男" in body.char else FEMALE_VOICE
    if body.video and body.char_id:
        project = _load_project(_safe_id(body.video))
        voice = project.get("voice_overrides", {}).get(body.char_id, voice)
        if body.char_id not in project.get("voice_overrides", {}):
            for character in project.get("characters", []):
                if character.get("id") == body.char_id and character.get("voice") in ("male", "female"):
                    voice = MALE_VOICE if character["voice"] == "male" else FEMALE_VOICE
                    break
    output = FILES / f"tts_{uuid.uuid4().hex}.mp3"
    rate = int(round(max(0.0, body.speed) * 100))
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            await edge_tts.Communicate(text, voice, rate=f"+{rate}%").save(str(output))
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            output.unlink(missing_ok=True)
            if attempt == 0:
                await asyncio.sleep(0.8)  # transient Edge TTS hiccups are common
    if last_error is not None:
        raise HTTPException(status_code=502, detail=f"Edge TTS failed for {voice}: {last_error}") from last_error
    audio_duration = _ffprobe_duration(output)
    # Guaranteed lip-sync fit: when the Khmer voice runs longer than the slot
    # (its window until the next line), speed the clip with atempo so voices
    # never overlap into the next character's shot.
    if body.fit_to >= 0.5 and audio_duration > body.fit_to:
        remaining = audio_duration / body.fit_to
        atempo: list[str] = []
        while remaining > 2.0:
            atempo.append("atempo=2.0")
            remaining /= 2.0
        atempo.append(f"atempo={min(2.0, max(0.5, remaining)):.4f}")
        fitted = output.with_suffix(".fit.mp3")
        result = subprocess.run(
            [_ffmpeg(), "-v", "error", "-y", "-i", str(output),
             "-filter:a", ",".join(atempo), str(fitted)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and fitted.is_file():
            fitted_duration = _ffprobe_duration(fitted)
            if fitted_duration <= body.fit_to * 1.05:
                output.unlink(missing_ok=True)
                fitted.replace(output)
                audio_duration = fitted_duration
    return {"url": _public_url(output), "audio_duration": audio_duration, "voice": voice}


def _scene_cuts(media: Path, duration: float) -> list[float]:
    """Shot boundaries via ffmpeg scene detection (free, no extra deps)."""
    result = subprocess.run(
        [_ffmpeg(), "-v", "info", "-i", str(media),
         "-vf", "select='gt(scene,0.27)',showinfo", "-an", "-f", "null", "-"],
        capture_output=True, text=True, check=False, timeout=240,
    )
    cuts = []
    for m in re.finditer(r"pts_time:([0-9.]+)", result.stderr or ""):
        t = float(m.group(1))
        if 0.05 < t < duration:
            cuts.append(round(t, 3))
    return sorted(set(cuts))[:400]


def _speech_profile(audio: Any, sample_rate: int, lo: float, hi: float) -> tuple[float, Any]:
    """RMS energy profile (per 10ms) of one window + the voiced-time list."""
    win = audio[int(lo * sample_rate):int(hi * sample_rate)]
    hop = int(sample_rate * 0.01)
    n = win.size // hop if win.size else 0
    if n == 0:
        return lo, None
    rms = np.sqrt(np.mean((win[:n * hop].reshape(n, hop)) ** 2, axis=1))
    return lo, rms


def _speech_window(audio: Any, sample_rate: int, start: float, end: float,
                   next_start: float) -> tuple[float, float, Any, float]:
    """Refine start/end to the actual speech-energy window (lip-sync).

    Returns (first, last, rms, lo) — rms is a per-10ms energy array starting at
    `lo` (None when the window is empty) so the caller can make shot decisions.
    """
    lo = max(0.0, start - 0.45)
    hi = max(start + 0.1, min(next_start, end + 0.45))
    lo, rms = _speech_profile(audio, sample_rate, lo, hi)
    if rms is None:
        return round(start, 3), round(end, 3), None, lo
    peak = float(rms.max())
    if peak < 1e-4:
        return round(start, 3), round(end, 3), rms, lo
    threshold = max(0.012, peak * 0.22)
    voiced = np.where(rms > threshold)[0]
    if voiced.size == 0:
        return round(start, 3), round(end, 3), rms, lo
    first = min(lo + float(voiced[0]) * 0.01, start + 0.8)
    last = min(max(lo + (float(voiced[-1]) + 1) * 0.01, end - 0.8), next_start)
    return round(max(0.0, first), 3), round(max(start + 0.05, last), 3), rms, lo


@app.post("/api/action_scan")
def action_scan(body: ScanRequest) -> dict[str, Any]:
    """Scan the video's shots + speech energy and re-align every dub line so the
    voice starts exactly when the on-screen character starts talking."""
    import bisect
    pid = _safe_id(body.id)
    media = _video_path(pid)
    duration = _ffprobe_duration(media)
    _operation(body.progress_id, state="running", phase="Scanning actions",
               completed=0, total=len(body.segments))
    try:
        import wave
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "mono.wav"
            result = subprocess.run(
                [_ffmpeg(), "-v", "error", "-y", "-i", str(media),
                 "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode or not wav_path.is_file():
                raise RuntimeError("ffmpeg could not decode the audio")
            with wave.open(str(wav_path), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if channels > 1:
            audio = audio[::channels].copy()
        audio = audio.astype(np.float64) / 32768.0

        cuts = _scene_cuts(media, duration)
        boundaries = [0.0] + cuts + [duration]
        segs = sorted(body.segments, key=lambda s: float(s.get("start", 0)))
        out: list[dict[str, Any]] = []
        adjusted_count = 0
        for index, segment in enumerate(segs):
            start = max(0.0, float(segment.get("start", 0)))
            end = max(start + 0.05, float(segment.get("end", start + 1)))
            next_start = float(segs[index + 1].get("start", duration)) if index + 1 < len(segs) else duration
            refined_start, refined_end, rms, lo = _speech_window(audio, sample_rate, start, end, next_start)
            shot = bisect.bisect_right(boundaries, refined_start)  # shot index (1-based)
            shot_end = boundaries[shot] if shot < len(boundaries) else duration
            adjusted = False
            if rms is not None and shot_end < refined_end:
                # Energy-weighted shot decision: only touch the line when the
                # speech clearly belongs to one side of the cut.
                i_first = int(max(0, (refined_start - lo) / 0.01))
                i_last = int(max(0, (refined_end - lo) / 0.01))
                i_cut = int(max(0, (shot_end - lo) / 0.01))
                voiced_idx = np.where(rms[i_first:i_last] > max(0.012, float(rms.max()) * 0.22))[0]
                total = voiced_idx.size
                after = int((voiced_idx >= i_cut - i_first).sum()) if total else 0
                if total and after < 0.3 * total:
                    refined_end = round(shot_end, 3)  # speech is before the cut
                    adjusted = True
                elif total and after > 0.7 * total:
                    refined_start = round(shot_end, 3)  # speech is after the cut
                    adjusted = True
            if refined_end - refined_start < 0.05:
                refined_start, refined_end, adjusted = round(start, 3), round(end, 3), False
            if adjusted:
                adjusted_count += 1
            out.append({**segment, "start": refined_start, "end": refined_end,
                        "shot": shot, "shot_adjusted": adjusted})
            _operation(body.progress_id, state="running", phase="Scanning actions",
                       completed=index + 1, total=len(segs))
        _operation(body.progress_id, state="done", phase="Action scan complete",
                   completed=len(out), total=len(out))
        return {"segments": out, "cuts": len(cuts), "shot_trimmed": adjusted_count}
    except Exception as exc:
        _operation(body.progress_id, state="error", phase="Scan failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Action scan failed: {exc}") from exc


@app.post("/api/characters/detect")
def detect_characters(body: DetectRequest) -> dict[str, Any]:
    pid = _safe_id(body.id)
    per_segment: list[dict[str, Any]] = []
    warning = ""
    if body.use_audio and body.segments:
        try:
            per_segment = _diarize_by_pitch(_video_path(pid), body.segments)
        except Exception as exc:
            warning = f"Audio diarization failed ({str(exc)[:160]}); default voice assigned."
    if not per_segment:
        # Whisper provides timestamps and text, not speaker identities. Keep the
        # default voice assignment explicit rather than claiming diarization.
        per_segment = [{
            "id": "speaker_1", "name": "Speaker 1 (female)", "voice": "female",
            "voiceName": FEMALE_VOICE,
        } for _ in body.segments]
        warning = warning or "Speaker diarization unavailable; one default voice is assigned."

    characters: list[dict[str, Any]] = []
    seen: set[str] = set()
    for character in per_segment:
        if character["id"] not in seen:
            seen.add(character["id"])
            characters.append({k: character[k] for k in ("id", "name", "voice", "voiceName")})
    items = [{"idx": i, "char_id": c["id"]} for i, c in enumerate(per_segment)]
    meta = _load_project(pid)
    meta["characters"] = characters
    if body.use_audio and body.segments and not warning:
        # Fresh detection: stale manual voice overrides (e.g. an earlier
        # all-male assignment) would otherwise make female characters speak
        # with the male voice while the UI shows the detected gender.
        meta["voice_overrides"] = {}
    _save_project(pid, meta)
    _operation(body.progress_id, state="done", phase="Speaker assignment ready",
               completed=len(items), total=len(items))
    if not warning:
        warning = ("Pitch-based diarization: male/female speakers are separated; "
                   "two same-gender speakers may share one voice.")
    return {"characters": characters, "items": items, "warning": warning}


@app.get("/api/characters")
def characters() -> dict[str, Any]:
    return {
        "roles": [
            {"char": "speaker_1", "voiceName": FEMALE_VOICE},
            {"char": "speaker_2", "voiceName": MALE_VOICE},
        ],
        "voices": [
            {"id": FEMALE_VOICE, "gender": "female", "age": "adult"},
            {"id": MALE_VOICE, "gender": "male", "age": "adult"},
        ],
    }


@app.post("/api/characters/assign")
def assign_characters(body: AssignRequest) -> dict[str, bool]:
    pid = _safe_id(body.id)
    meta = _load_project(pid)
    meta["voice_overrides"] = body.overrides
    _save_project(pid, meta)
    return {"ok": True}


@app.get("/api/operation_status")
def operation_status(id: str = Query(...)) -> dict[str, Any]:
    with _operation_lock:
        return dict(_operations.get(id, {"state": "queued", "phase": "Queued", "completed": 0, "total": 0}))


@app.post("/api/segments/fix")
def fix_segments(body: FixRequest) -> dict[str, Any]:
    duration = _ffprobe_duration(_video_path(body.id))
    cleaned = []
    for segment in sorted(body.segments, key=lambda row: max(0.0, float(row.get("start", 0)))):
        start = max(0.0, float(segment.get("start", 0)))
        end = max(start + 0.05, float(segment.get("end", start + 1)))
        if duration:
            start = min(start, duration)
            end = min(max(start + 0.05, end), duration)
        if end > start:
            cleaned.append({**segment, "start": round(start, 3), "end": round(end, 3)})
    return {"segments": cleaned, "count": len(cleaned)}


@app.post("/api/voice_preview")
async def voice_preview(body: dict[str, Any]) -> dict[str, Any]:
    voice = body.get("voice")
    if voice in ("male_adult_1", MALE_VOICE):
        voice = MALE_VOICE
    elif voice in ("female_adult_1", FEMALE_VOICE):
        voice = FEMALE_VOICE
    else:
        raise HTTPException(status_code=400, detail="Unknown preview voice")
    output = FILES / f"preview_{uuid.uuid4().hex}.mp3"
    try:
        await edge_tts.Communicate("សូមសាកល្បងសំឡេង", voice).save(str(output))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Edge TTS preview failed: {exc}") from exc
    return {"url": _public_url(output)}


@app.get("/api/settings")
def settings() -> dict[str, Any]:
    return {
        "translate_provider": "Gemini" if GEMINI_API_KEY else "Free Google Translate (no API key needed)",
        "stt_model": f"Whisper {WHISPER_MODEL} · {WHISPER_DEVICE}",
        "tts_provider": "Edge TTS",
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
    }


@app.get("/api/tiers")
def tiers() -> dict[str, str]:
    return {"current": "local"}


@app.get("/api/projects")
def projects() -> list[dict[str, Any]]:
    result = []
    for path in sorted(PROJECTS.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            project = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result.append({
            "pid": project.get("pid", path.stem),
            "lines": len(project.get("segments", [])),
            "characters": len(project.get("characters", [])),
            "final": project.get("final", ""),
        })
    return result


@app.get("/api/project/{pid}")
def get_project(pid: str) -> dict[str, Any]:
    project = _load_project(pid)
    if not project.get("video"):
        raise HTTPException(status_code=404, detail="Project not found")
    project["video_file"] = project["video"]
    project["video"] = Path(project["video"]).stem
    return project


@app.post("/api/layer")
async def upload_layer(
    id: str = Query(...),
    kind: str = Query(..., pattern="^(logo|sponsor|overlay)$"),
    file: UploadFile = File(...),
) -> dict[str, str]:
    pid = _safe_id(id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm"}:
        raise HTTPException(status_code=415, detail="Unsupported layer type")
    name = f"{pid}_{kind}_{uuid.uuid4().hex}{suffix}"
    _, path = await _save_upload(
        file,
        stem=Path(name).stem,
        allowed={".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm"},
    )
    return {"url": _public_url(path), "filename": path.name}


@app.delete("/api/layer")
def delete_layer(id: str = Query(...), kind: str = Query(...), file: str = Query(...)) -> dict[str, bool]:
    pid = _safe_id(id)
    if kind not in {"logo", "sponsor", "overlay"} or Path(file).name != file:
        raise HTTPException(status_code=400, detail="Invalid layer reference")
    if not file.startswith(f"{pid}_{kind}_"):
        raise HTTPException(status_code=400, detail="Layer does not belong to this project")
    target = FILES / file
    target.unlink(missing_ok=True)
    return {"ok": True}


def _write_data_url(data: str, pid: str, index: int) -> Path:
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", data)
    if not match:
        raise HTTPException(status_code=400, detail="Layer data must be a base64 PNG, JPEG, or WebP")
    suffix = ".jpg" if match.group(1) == "jpeg" else "." + match.group(1)
    path = FILES / f"{pid}_export_layer_{index}{suffix}"
    try:
        path.write_bytes(base64.b64decode(match.group(2), validate=True))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid layer image data") from exc
    return path


def _make_srt(segments: list[dict[str, Any]]) -> str:
    def stamp(value: float) -> str:
        millis = max(0, round(float(value) * 1000))
        hours, millis = divmod(millis, 3_600_000)
        minutes, millis = divmod(millis, 60_000)
        seconds, millis = divmod(millis, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    return "\n\n".join(
        f"{i}\n{stamp(s['start'])} --> {stamp(s['end'])}\n{s.get('km') or s.get('zh', '')}"
        for i, s in enumerate(segments, 1)
    ) + "\n"


def _resolve_file_url(value: str) -> Path:
    if not value.startswith("/files/"):
        raise HTTPException(status_code=400, detail="Expected a local /files/ asset URL")
    path = (FILES / Path(value.removeprefix("/files/")).name).resolve()
    if path.parent != FILES or not path.is_file():
        raise HTTPException(status_code=404, detail="Media asset not found")
    return path


async def _run_export(pid: str, body: ExportRequest) -> None:
    _exports[pid] = {"state": "running", "percent": 1, "eta": ""}
    export_warning = ""
    output = FILES / f"{pid}_km.mp4"
    stderr_path = FILES / f"{pid}_export.log"
    progress_path = FILES / f"{pid}_export.progress"
    temp_assets: list[Path] = []
    try:
        source = _video_path(pid)
        duration = _ffprobe_duration(source)
        if duration <= 0:
            raise RuntimeError("Could not read input duration; ensure ffprobe is installed")
        main_w, main_h = _ffprobe_video_size(source)
        command = [_ffmpeg(), "-y", "-i", str(source)]
        input_index = 1
        layer_inputs: list[tuple[int, dict[str, Any], Path]] = []
        for index, layer in enumerate(body.layers):
            layer_path: Path | None = None
            if layer.get("data"):
                layer_path = _write_data_url(str(layer["data"]), pid, index)
                temp_assets.append(layer_path)
            elif layer.get("file"):
                candidate = (FILES / Path(str(layer["file"])).name).resolve()
                if candidate.parent != FILES or not candidate.is_file():
                    raise RuntimeError("A requested layer asset is missing")
                layer_path = candidate
            if layer_path:
                is_video = layer_path.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"}
                if is_video:
                    command.extend(["-stream_loop", "-1"])
                command.extend(["-i", str(layer_path)])
                layer_inputs.append((input_index, layer, layer_path))
                input_index += 1

        bgm_input: int | None = None
        if body.bgm:
            bgm_path = _resolve_file_url(body.bgm)
            command.extend(["-stream_loop", "-1", "-i", str(bgm_path)])
            bgm_input = input_index
            input_index += 1

        voice_inputs: list[tuple[int, dict[str, Any]]] = []
        for segment in body.segments:
            if segment.get("audio"):
                audio_path = _resolve_file_url(str(segment["audio"]))
                command.extend(["-i", str(audio_path)])
                voice_inputs.append((input_index, segment))
                input_index += 1

        filters: list[str] = []
        video_label = "0:v"
        # Full HD output: scale so the SHORT side is 1080 (1920x1080 landscape /
        # 1080x1920 portrait), preserving aspect. "original" keeps source size.
        if main_w and main_h and body.resolution.lower() not in {"", "original", "source"}:
            if main_w >= main_h:
                filters.append("[0:v]scale=1920:-2:flags=lanczos,format=yuv420p[fhd]")
            else:
                filters.append("[0:v]scale=-2:1920:flags=lanczos,format=yuv420p[fhd]")
            video_label = "fhd"
        for index, (input_no, layer, layer_path) in enumerate(layer_inputs):
            scaled, output_label = f"layer{index}", f"v{index + 1}"
            size = max(1, min(100, float(layer.get("size", 25)))) / 100
            x = max(0, min(100, float(layer.get("x", 50)))) / 100
            y = max(0, min(100, float(layer.get("y", 50)))) / 100
            opacity = max(0, min(100, float(layer.get("opacity", 100)))) / 100
            # Size layers in Python from probed dimensions (the preview sizes
            # layers as % of frame WIDTH, preserving aspect). scale2ref's
            # main_w expression proved unreliable on newer FFmpeg builds.
            layer_w, layer_h = _ffprobe_video_size(layer_path)
            target_w = max(2, round(main_w * size / 2) * 2) if main_w else 0
            if target_w and layer_w and layer_h:
                target_h = max(2, round(target_w * layer_h / layer_w / 2) * 2)
            else:
                target_w = max(2, round(main_w * size / 2) * 2) if main_w else 0
                target_h = max(2, round(main_h * size / 2) * 2) if main_h else 0
            if not target_w or not target_h:
                raise RuntimeError("Could not determine layer/video dimensions")
            filters.append(
                f"[{input_no}:v]scale={target_w}:{target_h},format=rgba,"
                f"colorchannelmixer=aa={opacity}[{scaled}a]"
            )
            # NOTE: no shortest=1 — a static image layer is a single frame and
            # would freeze the whole video output at the first frame.
            filters.append(
                f"[{video_label}][{scaled}a]overlay=x=(W-w)*{x}:y=(H-h)*{y}[{output_label}]"
            )
            video_label = output_label

        for index, box in enumerate(body.boxes):
            x = max(0.0, min(0.96, float(box.get("x", 30)) / 100))
            y = max(0.0, min(0.96, float(box.get("y", 30)) / 100))
            width = max(0.04, min(0.96 - x, float(box.get("w", 40)) / 100))
            height = max(0.04, min(0.96 - y, float(box.get("h", 25)) / 100))
            strength = max(1, min(12, int(float(box.get("strength", 20)) / 3)))
            original, crop, blurred, output_label = (
                f"blurbase{index}", f"blurcrop{index}", f"blurred{index}", f"vblur{index + 1}"
            )
            filters.append(f"[{video_label}]split[{original}][{crop}]")
            filters.append(
                f"[{crop}]crop=w=iw*{width}:h=ih*{height}:x=iw*{x}:y=ih*{y},"
                f"boxblur={strength}:2[{blurred}]"
            )
            filters.append(
                f"[{original}][{blurred}]overlay=x=W*{x}:y=H*{y}[{output_label}]"
            )
            video_label = output_label

        if body.subs and body.segments:
            srt_path = FILES / f"{pid}_export.srt"
            srt_path.write_text(_make_srt(body.segments), encoding="utf-8")
            if _ffmpeg_has_filter("subtitles"):
                temp_assets.append(srt_path)
                escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
                filters.append(f"[{video_label}]subtitles='{escaped}'[vout]")
                video_label = "vout"
            else:
                # No libass in this FFmpeg build: keep the SRT next to the export
                # instead of failing the whole render.
                export_warning = (
                    "FFmpeg lacks the subtitles filter (libass); "
                    "the SRT was saved but not burned into the video."
                )
                _exports[pid] = {**_exports[pid], "warning": export_warning}
                print(f"KH Drama Studio: {export_warning}")

        audio_labels = []
        if bgm_input is not None:
            filters.append(f"[{bgm_input}:a]volume=0.25[bgm]")
            audio_labels.append("[bgm]")
        for index, (input_no, segment) in enumerate(voice_inputs):
            delay = max(0, int(float(segment.get("start", 0)) * 1000))
            label = f"voice{index}"
            filters.append(f"[{input_no}:a]adelay=delays={delay}:all=1,volume=1.0[{label}]")
            audio_labels.append(f"[{label}]")
        if audio_labels:
            filters.append(
                "".join(audio_labels) + f"amix=inputs={len(audio_labels)}:duration=longest:dropout_transition=2:normalize=0[aout]"
            )

        if filters:
            command.extend(["-filter_complex", ";".join(filters)])
            if video_label != "0:v":
                command.extend(["-map", f"[{video_label}]"])
            else:
                command.extend(["-map", "0:v"])
            if audio_labels:
                command.extend(["-map", "[aout]"])
            else:
                command.extend(["-map", "0:a?"])
        else:
            command.extend(["-map", "0:v", "-map", "0:a?"])

        quality = body.quality.lower()
        crf = "18" if quality in {"high", "best"} else "23"
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", crf,
                        "-c:a", "aac", "-movflags", "+faststart"])
        # Dubbed/BGM audio is mixed with delays; clamp the output to the source
        # duration so a late voice line cannot extend the file past the video.
        if audio_labels:
            command.extend(["-t", f"{duration:.3f}"])
        command.extend(["-progress", str(progress_path), "-nostats", str(output)])
        _exports[pid] = {"state": "running", "percent": 2, "eta": ""}
        with stderr_path.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.DEVNULL, stderr=stderr
            )
            while process.returncode is None:
                await asyncio.sleep(0.6)
                if progress_path.exists():
                    data = progress_path.read_text(errors="ignore")
                    matches = re.findall(r"out_time_ms=(\d+)", data)
                    if matches:
                        seconds = int(matches[-1]) / 1_000_000
                        percent = max(2, min(99, round(seconds / duration * 100)))
                        _exports[pid] = {"state": "running", "percent": percent, "eta": ""}
            code = await process.wait()
        if code:
            message = stderr_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(f"FFmpeg export failed: {message}")
        meta = _load_project(pid)
        meta["final"] = output.name
        meta["segments"] = body.segments
        _save_project(pid, meta)
        _exports[pid] = {"state": "done", "percent": 100, "eta": "", "url": _public_url(output), "warning": export_warning}
    except Exception as exc:
        output.unlink(missing_ok=True)
        _exports[pid] = {"state": "error", "percent": 0, "eta": "", "error": str(exc)}
    finally:
        for path in temp_assets:
            path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


@app.post("/api/export")
async def start_export(body: ExportRequest) -> dict[str, str]:
    pid = _safe_id(body.id)
    _video_path(pid)
    if _exports.get(pid, {}).get("state") == "running":
        raise HTTPException(status_code=409, detail="An export is already running for this project")
    _exports[pid] = {"state": "queued", "percent": 0, "eta": ""}
    _spawn(_run_export(pid, body))
    return {"state": "queued"}


@app.get("/api/export_status")
def export_status(id: str = Query(...)) -> dict[str, Any]:
    return dict(_exports.get(_safe_id(id), {"state": "idle", "percent": 0, "eta": ""}))


@app.post("/api/tts_export")
def tts_export(body: TTSExportRequest) -> dict[str, str]:
    if body.kind != "zip":
        raise HTTPException(status_code=400, detail="Only ZIP export is supported")
    path = FILES / f"tts_export_{uuid.uuid4().hex}.zip"
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, segment in enumerate(body.segments):
                audio = segment.get("audio")
                if not audio:
                    continue
                source = _resolve_file_url(str(audio))
                start = max(0.0, float(segment.get("start", 0)))
                archive.write(source, f"{index + 1:04d}_{start:.2f}{source.suffix}")
                archive.writestr(
                    f"{index + 1:04d}.txt",
                    f"{segment.get('km') or segment.get('zh', '')}\n",
                )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"url": _public_url(path)}


async def _process_bgm(pid: str, source: Path) -> None:
    _bgm[pid] = {"state": "running", "phase": "Separating vocals with Demucs"}
    try:
        demucs = shutil.which("demucs")
        if not demucs:
            raise RuntimeError("Demucs is not installed. Install the optional `demucs` package to remove vocals.")
        output_dir = DATA / "demucs"
        # torch.hub downloads model weights over SSL; some Python installs lack
        # a system CA bundle, so point it at certifi's bundle when available.
        env = dict(os.environ)
        try:
            import certifi
            env.setdefault("SSL_CERT_FILE", certifi.where())
        except ImportError:
            pass
        proc = await asyncio.create_subprocess_exec(
            # Single demucs member + shifts=0 is ~5x faster than the default
            # 4-model ensemble on CPU; quality is still solid for BGM extraction.
            # Override with STUDIO_DEMUCS_MODEL / STUDIO_DEMUCS_SHIFTS if wanted.
            demucs, "--two-stems", "vocals", "-n", DEMUCS_MODEL,
            "--shifts", str(DEMUCS_SHIFTS), "-o", str(output_dir), str(source),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        _, stderr = await proc.communicate()
        if proc.returncode:
            raise RuntimeError(stderr.decode(errors="replace")[-1500:])
        # The stem lands in {output_dir}/{model_name}/{track}/; the model folder
        # name depends on the installed demucs version, so search for it.
        matches = sorted(output_dir.glob(f"*/{source.stem}/no_vocals.wav"))
        no_vocals = matches[0] if matches else None
        if no_vocals is None:
            raise RuntimeError("Demucs completed without producing a no_vocals.wav stem")
        target = FILES / f"{pid}_bgm.wav"
        shutil.copy2(no_vocals, target)
        _bgm[pid] = {"state": "done", "phase": "BGM ready", "url": _public_url(target)}
    except Exception as exc:
        _bgm[pid] = {"state": "error", "phase": "BGM failed", "error": str(exc)}


@app.post("/api/bgm")
async def upload_bgm(id: str = Query(...), file: UploadFile = File(...)) -> dict[str, str]:
    pid = _safe_id(id)
    source_id, source = await _save_upload(file, stem=f"{pid}_bgm_source_{uuid.uuid4().hex}")
    _bgm[pid] = {"state": "queued", "phase": "Queued"}
    _spawn(_process_bgm(pid, source))
    return {"state": "queued", "id": source_id}


@app.get("/api/bgm_status")
def bgm_status(id: str = Query(...)) -> dict[str, Any]:
    return dict(_bgm.get(_safe_id(id), {"state": "idle", "phase": "No BGM job"}))


@app.post("/api/bgm_demo")
async def bgm_demo(id: str = Query(...)) -> dict[str, str]:
    pid = _safe_id(id)
    source = _video_path(pid)
    duration = max(1.0, _ffprobe_duration(source))
    output = FILES / f"{pid}_bgm_demo.wav"
    command = [
        _ffmpeg(), "-y",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=277.18:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=329.63:duration={duration}",
        "-filter_complex", "[0:a]volume=0.12[a0];[1:a]volume=0.08[a1];[2:a]volume=0.06[a2];[a0][a1][a2]amix=inputs=3:duration=longest",
        "-c:a", "pcm_s16le", str(output),
    ]
    result = await run_in_threadpool(subprocess.run, command, capture_output=True, check=False)
    if result.returncode:
        raise HTTPException(status_code=502, detail=f"Could not generate demo BGM: {result.stderr.decode(errors='replace')[-1000:]}")
    return {"url": _public_url(output)}


@app.on_event("startup")
async def report_configuration() -> None:
    if not shutil.which("ffmpeg"):
        print("KH Drama Studio: FFmpeg not found; export and demo BGM are disabled.")

