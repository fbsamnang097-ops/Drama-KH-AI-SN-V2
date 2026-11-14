import os
import uuid
import json
import wave
import struct
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
FILES = DATA
SILENT_WAV = DATA / "silent.wav"

# create a tiny silent wav once
if not SILENT_WAV.exists():
    with wave.open(str(SILENT_WAV), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050)

app = FastAPI(title="KH Drama Pro (mock backend)")

# in-memory state
VIDEOS = {}          # id -> filename
EXPORTS = {}         # id -> {state, percent, url}
BGM = {}             # id -> {state, url, warning}
OPS = {}             # progress_id -> {state, phase, total, completed}

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
    "បកប្រែ: សូមស្វាគមន៍",
    "បកប្រែ: ស្អប់ខ្លាំង",
    "បកប្រែ: អ្នកទើបតែប្រាំបួន",
    "បកប្រែ: ទឹកក៏បិទមិនបាន",
    "បកប្រែ: ខ្ញុំខកចិត្តណាស់",
    "បកប្រែ: សូមគោរព",
    "បកប្រែ: តើអ្នកសមនឹងទេ?",
    "បកប្រែ: ខ្ញុំមិនទាន់ស្លាប់ទេ",
    "បកប្រែ: ជូនកាដូ",
]


def _save_upload(f: UploadFile) -> (str, str):
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
    # simulate
    import time
    segs = [dict(s) for s in DEMO_LINES]
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
        # include original hint
        out.append({"km": km, "text": km})
    if prog:
        OPS[prog] = {"state": "done", "phase": "Done", "total": len(lines), "completed": len(lines)}
    return out


@app.post("/api/characters/detect")
async def detect(payload: dict):
    vid = payload.get("id")
    segs = payload.get("segments") or []
    prog = payload.get("progress_id")
    if prog:
        OPS[prog] = {"state": "running", "phase": "Detecting", "total": len(segs), "completed": 0}
    items = []
    for i, s in enumerate(segs):
        code = s.get("char") or "F1"
        items.append({"idx": i, "char_id": code})
    chars = [{"id": r["char"], "name": r["voiceName"], "voiceName": r["voiceName"], "voice": r["char"], "gender": "female" if r["char"] in ("F1", "S0", "女主") else "male"} for r in ROLES[:5]]
    if prog:
        OPS[prog] = {"state": "done", "phase": "Done", "total": len(segs), "completed": len(segs)}
    return {"items": items, "characters": chars}


@app.post("/api/tts")
async def tts(payload: dict):
    text = payload.get("text") or ""
    # return silent wav url; duration ~2.0
    return {"url": "/files/silent.wav", "audio_duration": 2.0}


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
    st = BGM.get(id, {"state": "done", "url": "/files/silent.wav", "warning": None})
    return st


@app.post("/api/bgm_demo")
async def bgm_demo(id: str = Query(...)):
    BGM[id] = {"state": "done", "url": "/files/silent.wav", "warning": None}
    return {"url": "/files/silent.wav"}


@app.post("/api/voice_preview")
async def voice_preview(payload: dict):
    return {"url": "/files/silent.wav"}


@app.post("/api/export")
async def export_job(payload: dict):
    vid = payload.get("id")
    EXPORTS[vid] = {"state": "done", "percent": 100, "eta": "0s", "url": VIDEOS.get(vid) and f"/files/{VIDEOS[vid]}" or "/files/silent.wav"}
    # if original video exists use it as result
    if vid in VIDEOS:
        EXPORTS[vid]["url"] = f"/files/{VIDEOS[vid]}"
    return {"ok": True}


@app.get("/api/export_status")
def export_status(id: str = Query(...)):
    st = EXPORTS.get(id, {"state": "done", "percent": 100, "eta": "0s", "url": f"/files/{VIDEOS.get(id, 'silent.wav')}"})
    return st


# serve uploaded files
app.mount("/files", StaticFiles(directory=str(DATA)), name="files")

# serve frontend (index.html) at root - mount last
app.mount("/", StaticFiles(directory=str(BASE), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)