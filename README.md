# KH Drama Pro

Translate Chinese drama into spoken Khmer — dubbing studio UI + mock backend.

## Run locally

```bash
./run.sh
```

Then open http://127.0.0.1:8000 (or set `PORT=3000 ./run.sh`).

Backend is FastAPI serving the frontend from the same origin.
Transcription uses faster-whisper (Chinese); TTS uses gTTS (Khmer);
export muxing needs FFmpeg on PATH (or ./bin).