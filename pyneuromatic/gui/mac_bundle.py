"""On macOS, run pyNeuroMatic as pyNeuroMatic.app, named and with its icon in
the Dock (#545).

Started from a script, Python runs inside its own Python.app, so the Dock
says "Python" (it takes the name of the app a process starts in, and keeps
it). :func:`relaunch_in_bundle` restarts the script, in the same process,
through a small pyNeuroMatic.app: its name, its icon and a link to this
Python's Python.app executable, so macOS takes the process to be
pyNeuroMatic.app. It runs this Python's environment (e.g. a .venv), and it is
still the terminal's process, so it may use the files the terminal may (an
app opened from Finder couldn't use a .venv in Documents).

The bundle is made in ~/Library/Application Support/pyNeuroMatic, and made
again when it no longer fits this Python.
"""
from __future__ import annotations

import os
import plistlib
import shutil
import sys
from pathlib import Path

APP_NAME = "pyNeuroMatic"
BUNDLE_ID = "uk.ac.ucl.silverlab.pyneuromatic"
ICONS = Path(__file__).resolve().parent / "icons"
BUNDLE_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
# Set in the restarted process, so it doesn't restart again
IN_BUNDLE = "PYNEUROMATIC_IN_BUNDLE"


def python_app_executable() -> Path | None:
    """The executable in this Python's Python.app (a framework build's:
    python.org's, Homebrew's, the Command Line Tools'), or None."""
    executable = (Path(sys.base_prefix) / "Resources" / "Python.app" / "Contents"
                  / "MacOS" / "Python")
    return executable if executable.exists() else None


def info_plist() -> dict:
    return {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": "python",
        "CFBundleIconFile": "pyneuromatic.icns",
        "CFBundlePackageType": "APPL",
        "NSHighResolutionCapable": True,
    }


def make_bundle(executable: Path, folder: Path = BUNDLE_DIR) -> Path:
    """*folder*/pyNeuroMatic.app, linking to *executable* (made, or made
    again if it links elsewhere or its icon changed); its python's path."""
    app = folder / f"{APP_NAME}.app"
    python = app / "Contents" / "MacOS" / "python"
    icon = app / "Contents" / "Resources" / "pyneuromatic.icns"
    source_icon = ICONS / "pyneuromatic.icns"
    fits = (python.is_symlink() and Path(os.readlink(python)) == executable
            and icon.exists() and icon.stat().st_size == source_icon.stat().st_size)
    if fits:
        return python
    if app.exists():
        shutil.rmtree(app)
    python.parent.mkdir(parents=True)
    icon.parent.mkdir()
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info_plist(), f)
    shutil.copy(source_icon, icon)
    python.symlink_to(executable)
    return python


def relaunch_args(args: list[str]) -> tuple[list[str], dict]:
    """The environment for running *args* (e.g. ["-m", "pyneuromatic.gui"])
    with this Python's environment, from the bundle."""
    import pyneuromatic

    env = dict(os.environ)
    env["__PYVENV_LAUNCHER__"] = os.path.abspath(sys.executable)  # e.g. the .venv
    source_root = str(Path(pyneuromatic.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (source_root, env.get("PYTHONPATH")) if p)
    env[IN_BUNDLE] = "1"
    return list(args), env


def relaunch_in_bundle(args: list[str]) -> None:
    """On macOS, run *args* again from pyNeuroMatic.app, replacing this
    process (no return); else, or if it can't, nothing happens and the caller
    goes on (the Dock then says "Python")."""
    if sys.platform != "darwin" or os.environ.get(IN_BUNDLE):
        return
    executable = python_app_executable()
    if executable is None:
        return
    try:
        python = make_bundle(executable)
        args, env = relaunch_args(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(str(python), [str(python), *args], env)
    except OSError:  # e.g. Application Support not writable: as Python
        return
