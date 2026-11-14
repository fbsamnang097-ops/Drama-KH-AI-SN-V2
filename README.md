# KH Drama Studio

The studio page is served by its local FastAPI backend so its relative `/api/*`
requests and `/files/*` media URLs use the same origin.

## Requirements

- Python 3.10+
- FFmpeg and ffprobe on `PATH`
- A Gemini API key for translation (optional — a free translation fallback is
  used automatically when no key is set)
- Network access on first Whisper/Demucs use to download the configured model

## Start (one click, every time)

Double-click **`start.command`** (or run `./start.command` in a terminal).
It starts the backend on `http://127.0.0.1:8765`, opens the browser, and
reuses the running server if it is already up. Close the window to stop it.

Manual start:

```bash
python3 -m uvicorn backend:app --host 127.0.0.1 --port 8765
```

Set `GEMINI_API_KEY` in your shell (or in a local `.env` file — `start.command`
loads it automatically) to upgrade translation quality from the free fallback
to Gemini. Do **not** open `studio.html` as a `file://` URL; always use the
server address above. Uploaded files and projects are stored under
`.studio-data/`.

## API capabilities and current limits

- Upload, local Whisper transcription, Khmer translation, Edge TTS,
  project listing/loading, SRT subtitle rendering, and Full HD FFmpeg export.
- **Detect Speakers** estimates each line's speaker from audio pitch
  (offline, dependency-free); male/female speakers are separated reliably,
  two same-gender speakers may share one voice. Re-detecting resets manual
  voice overrides. Click a row's voice chip to cycle auto → male → female
  for one line.
- **Scan Action** aligns every dub line to the on-screen action: it detects
  scene cuts, snaps line boundaries to speech energy (lip-sync), and trims
  lines that spill across a shot change.
- Lines without generated audio are marked (striped block + dashed subtitle
  chip) and retried per the Smart-retry setting; playback queues every line so
  no line is skipped silently.
- Vocal removal requires the Demucs CLI (installed). Exports default to Full
  HD (`resolution: "1080p"`); pass `"original"` to keep the source size.
- Export requires FFmpeg with `libx264`. If the build lacks the `subtitles`
  filter (libass), the export still completes without burned-in text and the
  SRT is saved next to the output video.
