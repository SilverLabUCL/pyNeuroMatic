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
    assert _messages(nm)[-1] == ("nm02Jul04c0_002.pxp: imported 38 data arrays; data series Record: "
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


def test_a_file_name_starting_with_a_digit(nm):
    pytest.importorskip("pyabf")
    folder = nm.folders.import_file(str(FIXTURES / "15804044.abf"))
    assert folder.name == "F15804044"  # as the file readers name it


@pytest.mark.parametrize("name, library", [
    ("nm02Jul04c0_002.pxp", "igor2"), ("Vers6_060523 004.axgd", None),
    ("15804044.abf", "pyabf")])
def test_folder_and_arrays_remember_their_file(nm, name, library):
    """The folder keeps the file it was read from; each array, the file and
    its own channel and epoch in it (#543)."""
    if library:
        pytest.importorskip(library)
    path = FIXTURES / name
    folder = nm.folders.import_file(str(path))
    source = folder.file_source
    assert source["file"] == name and source["path"] == str(path.absolute())
    assert source["size"] == path.stat().st_size
    assert source["modified"] and source["imported"]
    data = next(iter(folder.data.values()))
    assert data.file_source["file"] == name
    assert data.file_source["path"] == str(path.absolute())
    assert data.file_source["channel"] == "A" and data.file_source["epoch"] == 0
    assert _messages(nm)[-1].startswith(name + ": imported ")


def test_arrays_remember_the_files_channel_and_epoch(nm):
    """Renumbered from A and 0 when only some are imported; the source is
    the file's own; a copy keeps it (#543)."""
    pytest.importorskip("igor2")
    import copy
    folder = nm.folders.import_file(str(PXP), channels=["B"], epochs=range(3, 5))
    first = folder.data["RecordA0"]
    assert (first.file_source["channel"], first.file_source["epoch"]) == ("B", 3)
    assert (folder.data["RecordA1"].file_source["channel"],
            folder.data["RecordA1"].file_source["epoch"]) == ("B", 4)
    assert copy.deepcopy(first).file_source == first.file_source
    assert copy.deepcopy(folder).file_source == folder.file_source


def test_made_data_has_no_file_source(nm):
    folder = nm.folders.new("Made")
    data = folder.data.new("Y", nparray=np.array([1.0]))
    assert folder.file_source is None and data.file_source is None
    with pytest.raises(TypeError):
        data.file_source = "cell.pxp"


def test_an_array_moved_or_copied_keeps_its_file_source(nm):
    first, second = nm.folders.new("First"), nm.folders.new("Second")
    data = first.data.new("Y", nparray=np.array([1.0, 2.0]))
    source = {"file": "cell.pxp", "path": "/Data/cell.pxp", "channel": "A", "epoch": 0}
    data.file_source = source
    first.move_data("Y", second, quiet=True)
    assert second.data["Y"].file_source == source
    second.data.copy_to("Y", first.data, quiet=True)
    assert first.data["Y"].file_source == source
    second.data.duplicate("Y", "Y2")
    assert second.data["Y2"].file_source == source
