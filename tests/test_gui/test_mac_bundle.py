"""Tests for running as pyNeuroMatic.app on macOS (#545)."""

import os
import plistlib
import sys

import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.gui import mac_bundle


def test_bundle_is_named_with_its_icon(tmp_path):
    executable = tmp_path / "Python"  # stands in for Python.app's
    executable.write_text("")
    python = mac_bundle.make_bundle(executable, tmp_path)
    contents = tmp_path / "pyNeuroMatic.app" / "Contents"
    with open(contents / "Info.plist", "rb") as f:
        info = plistlib.load(f)
    assert info["CFBundleName"] == info["CFBundleDisplayName"] == "pyNeuroMatic"
    assert info["CFBundleExecutable"] == python.name == "python"
    assert (contents / "Resources" / "pyneuromatic.icns").exists()
    assert python.is_symlink() and os.readlink(python) == str(executable)
    # Kept while it fits; made again for another Python
    assert mac_bundle.make_bundle(executable, tmp_path) == python
    other = tmp_path / "Other"
    other.write_text("")
    mac_bundle.make_bundle(other, tmp_path)
    assert os.readlink(python) == str(other)


def test_relaunch_keeps_this_pythons_environment(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    args, env = mac_bundle.relaunch_args(["-m", "pyneuromatic.gui", "--demo"])
    assert args == ["-m", "pyneuromatic.gui", "--demo"]
    assert env["__PYVENV_LAUNCHER__"] == os.path.abspath(sys.executable)
    assert os.path.isdir(os.path.join(env["PYTHONPATH"], "pyneuromatic"))
    assert env[mac_bundle.IN_BUNDLE] == "1"


def test_no_relaunch_once_relaunched_or_not_macos(monkeypatch):
    run = []
    monkeypatch.setattr(os, "execve", lambda *a: run.append(a))
    monkeypatch.setenv(mac_bundle.IN_BUNDLE, "1")
    mac_bundle.relaunch_in_bundle(["-m", "pyneuromatic.gui"])
    monkeypatch.delenv(mac_bundle.IN_BUNDLE)
    monkeypatch.setattr(sys, "platform", "linux")
    mac_bundle.relaunch_in_bundle(["-m", "pyneuromatic.gui"])
    assert run == []


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS")
def test_relaunch_runs_the_bundles_python(monkeypatch, tmp_path):
    if mac_bundle.python_app_executable() is None:
        pytest.skip("not a framework build of Python")
    run = []
    monkeypatch.delenv(mac_bundle.IN_BUNDLE, raising=False)
    monkeypatch.setattr(mac_bundle, "BUNDLE_DIR", tmp_path)
    monkeypatch.setattr(mac_bundle, "make_bundle",
                        lambda executable: tmp_path / "pyNeuroMatic.app/Contents/MacOS/python")
    monkeypatch.setattr(os, "execve", lambda path, argv, env: run.append((path, argv, env)))
    mac_bundle.relaunch_in_bundle(["script.py", "--demo"])
    path, argv, env = run[0]
    assert path.endswith("pyNeuroMatic.app/Contents/MacOS/python")
    assert argv[1:] == ["script.py", "--demo"]
    assert env[mac_bundle.IN_BUNDLE] == "1"


def test_gui_main_starts_the_window(monkeypatch):
    """python -m pyneuromatic.gui: the window, with the user's settings."""
    from PyQt6 import QtWidgets
    from pyneuromatic.gui import __main__ as gui_main
    from pyneuromatic.gui import app_window
    shown = []
    monkeypatch.setattr(app_window.NMAppWindow, "show", lambda self: shown.append(self))
    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda self: 0)  # no event loop
    assert gui_main.main(["pyneuromatic"]) == 0
    assert len(shown) == 1
