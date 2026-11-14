#!/bin/bash
# KH Drama Studio — one-click launcher (double-click on macOS)
cd "$(dirname "$0")" || exit 1

PORT=8765
URL="http://127.0.0.1:$PORT"

# load .env if present (GEMINI_API_KEY, etc.)
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# already running? just open the page
if curl -s -m 2 "$URL/api/health" | grep -q ok; then
  open "$URL"
  echo "Server already running at $URL"
  exit 0
fi

echo "Starting KH Drama Studio at $URL ..."
python3 -m uvicorn backend:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!

# wait until healthy, then open the browser once
for i in $(seq 1 30); do
  if curl -s -m 1 "$URL/api/health" | grep -q ok; then
    open "$URL"
    break
  fi
  sleep 0.5
done

echo "Server running (PID $SERVER_PID). Close this window to stop it."
wait $SERVER_PID
