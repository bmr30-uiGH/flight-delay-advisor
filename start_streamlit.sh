#!/bin/bash
set -u

# Wait briefly for the local MLflow registry before serving the UI.
for attempt in $(seq 1 60); do
  if python - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=2).read()
PY
  then
    echo "MLflow registry is ready."
    break
  fi
  sleep 1
done

exec streamlit run /app/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
