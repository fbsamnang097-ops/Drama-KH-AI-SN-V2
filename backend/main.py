import os
import uuid
import math
import wave
import struct
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
BIN = BASE / "bin"
FFMPEG = str(BIN / "ffmpeg") if (BIN / "ffmpeg").exists() else "ffmpeg"
FFPROBE = str(BIN / "ffprobe") if (BIN / "ffprobe").exists() else "ffprobe"

def ffmpeg_ok():
    try:
        subprocess.run([FFMPEG, "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def gtts_speak(text, path):
    from gtts import gTTS
    t = (text or "").strip()
    if not t:
        return False
    try:
        gTTS(text=t, lang="km", slow=False).save(str(path))
        return True
    except Exception:
        return False

def probe_duration(path):
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0

app = FastAPI(title="KH Drama Pro")

VIDEOS = {}
EXPORTS = {}
BGM = {}
OPS = {}
VID_SEGMENTS = {}

ROLES = [
    {"char": "F1", "voiceName": "F1 (female)"},
    {"char": "S0", "voiceName": "S0 (female)"},
    {"char": "M2", "voiceName": "M2 (male)"},
    {"char": "M1", "voiceName": "M1 (male)"},
    {"char": "Sr", "voiceName": "Sreymon"},
    {"char": "男主", "voiceName": "Male lead"},
    {"char": "女主", "voiceName": "Female lead"},
    {"char": "旁白", "voiceName": "Narrator"},
]
VOICES = [
    {"id": "male_adult_1", "gender": "male", "age": "adult"},
    {"id": "female_adult_1", "gender": "female", "age": "adult"},
]

DEMO_LINES = [
    {"start": 3.4, "end": 4.2, "zh": "青山呀", "char": "F1"},
    {"start": 4.2, "end": 4.8, "zh": "青山", "char": "S0"},
    {"start": 4.9, "end": 6.4, "zh": "你才八十八", "char": "M2"},
    {"start": 6.7, "end": 8.0, "zh": "水都遮忍不住了", "char": "M1"},
    {"start": 8.1, "end": 10.5, "zh": "太让我失望了", "char": "F1"},
    {"start": 15.9, "end": 16.7, "zh": "老祖宗", "char": "S0"},
    {"start": 17.3, "end": 18.6, "zh": "你值得留仙人?", "char": "M2"},
    {"start": 19.0, "end": 21.3, "zh": "我还没死这个你先出", "char": "F1"},
    {"start": 22.5, "end": 24.1, "zh": "送手镯了", "char": "S0"},
]

KH_MOCK = [
    "សូមស្វាគមន៍",
    "ស្អប់ខ្លាំង",
    "អ្នកទើបតែប្រាំបួន",
    "ទឹកក៏បិទមិនបាន",
    "ខ្ញុំខកចិត្តណាស់",
    "សូមគោរព",
    "តើអ្នកសមនឹងទេ?",
    "ខ្ញុំមិនទាន់ស្លាប់ទេ",
    "ជូនកាដូ",
]


def _save_upload(f: UploadFile):
    ext = Path(f.filename or "media").suffix or ".bin"
    vid = uuid.uuid4().hex[:12]
    name = f"{vid}{ext}"
    path = DATA / name
    with open(path, "wb") as out:
        shutil.copyfileobj(f.file, out)
    VIDEOS[vid] = name
    return vid, f"/files/{name}"


@app.get("/api/characters")
def characters():
    return {"roles": ROLES, "voices": VOICES}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    vid, url = _save_upload(file)
    return {"id": vid, "url": url}


@app.get("/api/operation_status")
def operation_status(id: str = Query(...)):
    st = OPS.get(id, {"state": "done", "phase": "Done", "total": 1, "completed": 1})
    return st


@app.post("/api/transcribe")
async def transcribe(payload: dict):
    vid = payload.get("id")
    prog = payload.get("progress_id")
    if prog:
        OPS[prog] = {"state": "running", "phase": "Transcribing", "total": len(DEMO_LINES), "completed": 0}
    segs = [dict(s) for s in DEMO_LINES]
    if vid:
        VID_SEGMENTS[vid] = segs
    if prog:
        OPS[prog] = {"state": "done", "phase": "Done", "total": len(segs), "completed": len(segs)}
    return segs


@app.post("/api/translate")
async def translate(payload: dict):
    lines = payload.get("lines") or []
    prog = payload.get("progress_id")
    if prog:
        OPS[prog] = {"state": "running", "phase": "Translating", "total": len(lines), "completed": 0}
    out = []
    for i, ln in enumerate(lines):
        km = KH_MOCK[i % len(KH_MOCK)]
        out.append({"km": km, "text": km})
    if prog:
        OPS[prog] = {"state": "done", "phase": "Done", "total": len(lines), "completed": len(lines)}
    return out


@app.post("/api/characters/detect")
async def detect(payload: dict):
    segs = payload.get("segments") or []
    prog = payload.get("progress_id")
    if prog:
        OPS[prog] = {"state": "running", "phase": "Detecting", "total": len(segs), "completed": 0}
    items = []
    for i, s in enumerate(segs):
        code = s.get("char") or "F1"
        items.append({"idx": i, "char_id": code})
    chars = [{"id": r["char"], "name": r["voiceName"], "voiceName": r["voiceName"], "voice": r["char"],
              "gender": "female" if r["char"] in ("F1", "S0", "女主") else "male"} for r in ROLES[:5]]
    if prog:
        OPS[prog] = {"state": "done", "phase": "Done", "total": len(segs), "completed": len(segs)}
    return {"items": items, "characters": chars}


@app.post("/api/tts")
async def tts(payload: dict):
    vid = payload.get("video") or ""
    idx = payload.get("idx")
    text = (payload.get("text") or "").strip()
    # try to get khmer text from stored segments if not provided
    segs = VID_SEGMENTS.get(vid) or []
    km_text = text
    if not km_text and idx is not None:
        try:
            i = int(idx)
            if 0 <= i < len(segs):
                km_text = (segs[i].get("km") or segs[i].get("zh") or "").strip()
        except Exception:
            pass
    # strip latin fallback markers
    km_text = km_text.replace("បកប្រែ: ", "").strip() if km_text.startswith("បកប្រែ:") else km_text
    # use provided text if it contains khmer chars else use mock
    if not any("\u1780" <= ch <= "\u17ff" for ch in km_text):
        try:
            i = int(idx) if idx is not None else 0
            km_text = KH_MOCK[i % len(KH_MOCK)].replace("បកប្រែ: ", "").strip()
        except Exception:
            km_text = "សួស្តី"

    name = f"{vid or 'x'}_{idx if idx is not None else 'tts'}.mp3"
    path = DATA / name
    ok = False
    if km_text:
        ok = gtts_speak(km_text, path)
    if not ok:
        # fallback beep
        from pathlib import Path as _P
        beep = DATA / "beep.wav"
        if not beep.exists():
            # generate
            fr = 22050; dur = 1.2; n = int(fr * dur)
            import wave as _w
            with _w.open(str(beep), "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fr)
                buf = bytearray()
                for i in range(n):
                    t = i / fr
                    val = 0.3 * math.sin(2 * math.pi * 440 * t)
                    iv = int(max(-1, min(1, val)) * 32767)
                    buf += iv.to_bytes(2, "little", signed=True)
                wf.writeframes(buf)
        path = beep
        name = beep.name
    dur = probe_duration(path)
    if dur <= 0:
        dur = 1.2
    return {"url": f"/files/{name}", "audio_duration": dur}


@app.post("/api/layer")
async def layer(id: str = Query(...), kind: str = Query("overlay"), file: UploadFile = File(...)):
    ext = Path(file.filename or "layer").suffix or ".bin"
    name = f"{id}_layer_{uuid.uuid4().hex[:6]}{ext}"
    path = DATA / name
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"url": f"/files/{name}"}


@app.post("/api/bgm")
async def bgm(id: str = Query(...), file: UploadFile = File(...)):
    ext = Path(file.filename or "bgm").suffix or ".bin"
    name = f"{id}_bgm{ext}"
    path = DATA / name
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    BGM[id] = {"state": "done", "url": f"/files/{name}", "warning": None}
    return {"ok": True}


@app.get("/api/bgm_status")
def bgm_status(id: str = Query(...)):
    st = BGM.get(id, {"state": "done", "url": "/files/beep.wav", "warning": None})
    return st


@app.post("/api/bgm_demo")
async def bgm_demo(id: str = Query(...)):
    BGM[id] = {"state": "done", "url": "/files/beep.wav", "warning": None}
    return {"url": "/files/beep.wav"}


@app.post("/api/voice_preview")
async def voice_preview(payload: dict):
    vid = payload.get("voice") or ""
    text = "សួស្តី"
    name = f"preview_{uuid.uuid4().hex[:6]}.mp3"
    path = DATA / name
    ok = gtts_speak(text, path)
    if not ok:
        return {"url": "/files/beep.wav"}
    return {"url": f"/files/{name}"}


@app.post("/api/export")
async def export_job(payload: dict):
    vid = payload.get("id")
    segs = payload.get("segments") or []
    video_path = DATA / VIDEOS.get(vid, "") if vid in VIDEOS else None
    out_path = DATA / f"{vid}_km.mp4"

    if not video_path or not video_path.exists() or not ffmpeg_ok():
        # fallback: return original or beep
        url = f"/files/{VIDEOS.get(vid)}" if vid in VIDEOS else "/files/beep.wav"
        EXPORTS[vid] = {"state": "done", "percent": 100, "eta": "0s", "url": url}
        return {"ok": True}

    # collect audio clips
    clips = []  # (start_seconds, local_path)
    for s in segs:
        audio_url = s.get("audio") or ""
        if not audio_url:
            continue
        fname = audio_url.split("/")[-1]
        fpath = DATA / fname
        if not fpath.exists():
            continue
        start = float(s.get("start") or 0)
        clips.append((start, fpath))

    if not clips:
        # no audio: just copy/remux video
        subprocess.run([FFMPEG, "-y", "-i", str(video_path), "-c", "copy", str(out_path)],
                       capture_output=True)
        EXPORTS[vid] = {"state": "done", "percent": 100, "eta": "0s", "url": f"/files/{out_path.name}"}
        return {"ok": True}

    # build ffmpeg command with adelay + amix
    cmd = [FFMPEG, "-y", "-i", str(video_path)]
    # silent base audio
    cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono"]
    for start, fpath in clips:
        cmd += ["-i", str(fpath)]

    # filter complex
    # input 0 = video, 1 = anullsrc, 2.. = clips
    filter_parts = []
    amix_inputs = ["[1:a]"]
    for i, (start, fpath) in enumerate(clips):
        inp_idx = 2 + i
        delay_ms = int(start * 1000)
        tag = f"[a{i}]"
        filter_parts.append(f"[{inp_idx}:a]adelay={delay_ms}|{delay_ms}{tag}")
        amix_inputs.append(tag)
    n = len(amix_inputs)
    filter_parts.append(f"{''.join(amix_inputs)}amix=inputs={n}:duration=first:dropout_transition=0[aout]")
    filter_complex = ";".join(filter_parts)

    cmd += ["-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(out_path)]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        # fallback to original
        shutil.copyfile(video_path, out_path)
    url = f"/files/{out_path.name}" if out_path.exists() else f"/files/{VIDEOS.get(vid)}"
    EXPORTS[vid] = {"state": "done", "percent": 100, "eta": "0s", "url": url}
    return {"ok": True}


@app.get("/api/export_status")
def export_status(id: str = Query(...)):
    st = EXPORTS.get(id, {"state": "done", "percent": 100, "eta": "0s", "url": f"/files/{VIDEOS.get(id, 'beep.wav')}"})
    return st


app.mount("/files", StaticFiles(directory=str(DATA)), name="files")
app.mount("/", StaticFiles(directory=str(BASE), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
