@echo off
rem QuantLab launcher for Windows. Double-click, or point a desktop shortcut at it.
rem Uses the repo's .venv if it exists, otherwise whatever `python` is on PATH.
setlocal
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit is not installed for %PY%.
    echo Run:  %PY% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting QuantLab... the browser opens by itself. Close this window to stop it.
"%PY%" -m streamlit run quantlab\app.py
if errorlevel 1 pause
