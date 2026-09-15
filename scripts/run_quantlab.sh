#!/usr/bin/env bash
# QuantLab launcher for Linux/macOS. Also used by the .command file and the .desktop entry.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

if ! "$PY" -c "import streamlit" 2>/dev/null; then
    echo "Streamlit is not installed for $PY."
    echo "Run:  $PY -m pip install -r requirements.txt"
    read -r -p "Press enter to close."
    exit 1
fi

echo "Starting QuantLab... the browser opens by itself. Ctrl+C or close this window to stop it."
exec "$PY" -m streamlit run quantlab/app.py
