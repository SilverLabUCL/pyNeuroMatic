"""Tests for NMFolderContainer.import_file: importing a file as one command (#455)."""

from pathlib import Path

import numpy as np
import pytest

from pyneuromatic.core.nm_folder import FILE_READERS
from pyneuromatic.core.nm_manager import NMManager

FIXTURES = Path(__file__).parent / "fixtures"
PXP = FIXTURES / "nm02Jul04c0_002.pxp"
AXGD = FIXTURES / "Vers6_060523 004.axgd"


@pytest.fixture
def nm():
    return NMManager(quiet=True, command_history=True)


def _commands(nm):
    return [e["command"] for e in nm.command_history.buffer][1:]  # after "nm = NMManager()"


def _messages(nm):
    return [e["message"] for e in nm.history.buffer if e["kind"] == ""]


def test_import_is_one_command_and_one_message(nm):
    pytest.importorskip("igor2")
    folder = nm.folders.import_file(str(PXP))
    assert folder.name == "nm02Jul04c0_002"
    assert len(folder.data) == 38
    assert _commands(nm) == ["nm.folders.import_file(%r)" % str(PXP)]
    assert _messages(nm)[-1] == ("imported 38 data arrays; data series Record: "
                                 "channels A, B; epochs E0..E18 (locked)")
    assert not any("new 'E" in m or "locked dataseries" in m for m in _messages(nm))


def test_command_replays_the_import(nm):
    pytest.importorskip("igor2")
    folder = nm.folders.import_file(str(PXP))
    other = NMManager(quiet=True)
    exec(_commands(nm)[-1], {"nm": other})
    copy = other.folders["nm02Jul04c0_002"]
    assert list(copy.data.keys()) == list(folder.data.keys())
    assert all(np.array_equal(copy.data[k].nparray, folder.data[k].nparray)
               for k in folder.data.keys())
    assert list(copy.dataseries["Record"].epochs.keys()) == \
        list(folder.dataseries["Record"].epochs.keys())


def test_names_are_made_unique_or_given(nm):
    first = nm.folders.import_file(str(AXGD))
    second = nm.folders.import_file(str(AXGD))
    named = nm.folders.import_file(str(AXGD), name="Cell2")
    assert [first.name, second.name, named.name] == [
        "Vers6_060523_004", "Vers6_060523_004_1", "Cell2"]
    assert _commands(nm)[-1] == "nm.folders.import_file(%r, name='Cell2')" % str(AXGD)


def test_unsupported_file_type(nm, tmp_path):
    with pytest.raises(ValueError):
        nm.folders.import_file(str(tmp_path / "notes.txt"))
    assert len(nm.folders) == 0 and _commands(nm) == []


def test_a_file_that_cannot_be_read_leaves_nothing(nm, tmp_path, monkeypatch):
    def broken(path, folder):
        folder.data.new("RecordA0", nparray=np.zeros(3))
        raise OSError("corrupt file")

    monkeypatch.setitem(FILE_READERS, ".fake", broken)
    with pytest.raises(OSError):
        nm.folders.import_file(str(tmp_path / "cell.fake"))
    assert len(nm.folders) == 0
    assert _commands(nm) == []  # neither the import nor its steps
