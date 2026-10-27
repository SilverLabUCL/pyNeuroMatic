"""Tests for notes on tool (results) folders."""
import copy

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_notes import NMNotes
from pyneuromatic.tools.nm_tool_folder import NMToolFolder


def _toolfolder():
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    return folder.toolfolders.new("Stats_0")


def test_tool_folder_keeps_notes():
    toolfolder = _toolfolder()
    assert isinstance(toolfolder.notes, NMNotes) and len(toolfolder.notes) == 0
    toolfolder.notes.add("channel transform: Invert()")
    assert toolfolder.notes.note == "channel transform: Invert()"


def test_notes_in_eq_and_copies():
    toolfolder = _toolfolder()
    toolfolder.notes.add("run 1")
    copied = copy.deepcopy(toolfolder)
    assert copied.notes == toolfolder.notes
    assert copied.notes is not toolfolder.notes
    assert copied == toolfolder
    copied.notes.add("run 2")
    assert copied != toolfolder


def test_older_tool_folder_gets_notes():
    toolfolder = NMToolFolder(name="Old")
    del toolfolder._NMToolFolder__notes  # e.g. made before tool folders kept notes
    assert len(toolfolder.notes) == 0
