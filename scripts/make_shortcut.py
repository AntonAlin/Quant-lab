"""Put a QuantLab shortcut on the desktop. Run once: `python scripts/make_shortcut.py`.

Windows -> QuantLab.lnk pointing at run_quantlab.bat (via PowerShell, so no pywin32).
macOS   -> QuantLab.command on the Desktop that execs the launcher.
Linux   -> QuantLab.desktop entry, marked trusted where the desktop supports it.

The Desktop folder is asked from the OS rather than guessed, because OneDrive
loves to move it and nobody wants a shortcut in a folder that no longer exists.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def desktop_dir() -> Path:
    if platform.system() == "Windows":
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(out)
    xdg = ""
    if platform.system() == "Linux":
        try:
            xdg = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            pass  # minimal box without xdg-utils; fall back to ~/Desktop
    d = Path(xdg) if xdg else Path.home() / "Desktop"
    d.mkdir(parents=True, exist_ok=True)
    return d


def windows(desktop: Path) -> Path:
    target = SCRIPTS / "run_quantlab.bat"
    link = desktop / "QuantLab.lnk"
    ps = f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{link}')
$s.TargetPath = '{target}'
$s.WorkingDirectory = '{ROOT}'
$s.Description = 'QuantLab strategy laboratory'
$s.IconLocation = '%SystemRoot%\\System32\\shell32.dll,165'
$s.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return link


def macos(desktop: Path) -> Path:
    link = desktop / "QuantLab.command"
    link.write_text(f'#!/usr/bin/env bash\nexec "{SCRIPTS / "run_quantlab.sh"}"\n')
    link.chmod(0o755)
    # Strip the quarantine flag so Gatekeeper does not nag on first double-click.
    try:
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(link)], capture_output=True)
    except FileNotFoundError:
        pass
    return link


def linux(desktop: Path) -> Path:
    link = desktop / "QuantLab.desktop"
    link.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=QuantLab\n"
        "Comment=ML strategy laboratory\n"
        f'Exec=bash -c "{SCRIPTS / "run_quantlab.sh"}"\n'
        f"Path={ROOT}\n"
        "Terminal=true\n"
        "Icon=utilities-terminal\n"
        "Categories=Science;Finance;\n"
    )
    link.chmod(0o755)
    # GNOME wants an explicit "trusted" stamp or it shows a grey icon that does nothing.
    try:
        subprocess.run(["gio", "set", str(link), "metadata::trusted", "true"], capture_output=True)
    except FileNotFoundError:
        pass
    return link


def main() -> int:
    system = platform.system()
    desktop = desktop_dir()
    if system == "Windows":
        link = windows(desktop)
    elif system == "Darwin":
        link = macos(desktop)
    elif system == "Linux":
        link = linux(desktop)
    else:
        print(f"No idea how to make a shortcut on {system}. Run scripts/run_quantlab.sh by hand.")
        return 1
    print(f"Shortcut created: {link}")
    print(f"It launches: {SCRIPTS / ('run_quantlab.bat' if system == 'Windows' else 'run_quantlab.sh')}")
    if not (ROOT / ".venv").exists():
        print("Note: no .venv found in the repo. The launcher will use the system python; make sure requirements are installed there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
