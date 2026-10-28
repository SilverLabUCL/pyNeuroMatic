"""Tests for renumbering a data series' epochs (NMFolder.renumber_dataseries_epochs)."""
import pytest

from pyneuromatic.core.nm_manager import NMManager


@pytest.fixture
def folder():
    """Series Record: channels A, B x E0..E5; E1, E3, E4 removed with their
    arrays, leaving E0, E2, E5. Set0 = E2, E5; E5 in group 1; data set Mine
    = RecordA5."""
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=6, n_points=3)
    folder.remove_dataseries_epoch("Record", ["E1", "E3", "E4"], delete_data=True)
    series.epochs.sets.add("Set0", ["E2", "E5"])
    series.epochs.groups.assign("E5", 1)
    folder.data.sets.add("Mine", ["RecordA5"])
    return folder


def test_plan(folder):
    epochs, arrays, empty = folder.renumber_epochs_plan("Record")
    assert epochs == {"E2": "E1", "E5": "E2"}
    assert arrays == {"RecordA2": "RecordA1", "RecordB2": "RecordB1",
                      "RecordA5": "RecordA2", "RecordB5": "RecordB2"}
    assert empty == []
    assert list(folder.dataseries["Record"].epochs.keys()) == ["E0", "E2", "E5"]  # unchanged


def test_renumber(folder):
    series = folder.dataseries["Record"]
    a5 = folder.data["RecordA5"]
    assert folder.renumber_dataseries_epochs("Record") == {"E2": "E1", "E5": "E2"}
    assert list(series.epochs.keys()) == ["E0", "E1", "E2"]
    assert [e.number for e in series.epochs.values()] == [0, 1, 2]
    assert [d.name for d in series.channels["A"].data] == ["RecordA0", "RecordA1", "RecordA2"]
    assert folder.data["RecordA2"] is a5  # the same array, renamed
    assert [d.name for d in series.epochs["E2"].data] == ["RecordA2", "RecordB2"]
    # Sets, groups and data sets follow
    assert series.epochs.sets.get_items("Set0", get_keys=True) == ["E1", "E2"]
    assert series.epochs.groups.get_group("E2") == 1
    assert folder.data.sets.get_items("Mine", get_keys=True) == ["RecordA2"]
    # Notes record the old names
    assert a5.notes.note == "renamed from 'RecordA5' (Renumber Epochs)"
    assert series.notes.note == "renumbered epochs: E2 → E1, E5 → E2"


def test_already_numbered(folder):
    folder.renumber_dataseries_epochs("Record")
    assert folder.renumber_epochs_plan("Record") == ({}, {}, [])
    assert folder.renumber_dataseries_epochs("Record") == {}


def test_refused(folder):
    folder.data.new("RecordA1")  # e.g. kept when its epoch was removed
    with pytest.raises(ValueError, match="'RecordA1' already exists .not part of data series"):
        folder.renumber_dataseries_epochs("Record")
    assert list(folder.dataseries["Record"].epochs.keys()) == ["E0", "E2", "E5"]
    folder.data.pop("RecordA1")
    folder.dataseries["Record"].locked = True
    with pytest.raises(RuntimeError, match="locked"):
        folder.renumber_dataseries_epochs("Record")


def test_renumber_follows_the_series_order(folder):
    # After an append, epochs are in series order already; renumbering keeps it
    folder.new_dataseries("Stim", n_channels=2, n_epochs=1, n_points=3)
    folder.append_dataseries("Record", "Stim", delete_source=True)  # adds E6
    folder.renumber_dataseries_epochs("Record")
    assert list(folder.dataseries["Record"].epochs.keys()) == ["E0", "E1", "E2", "E3"]
    assert "RecordA3" in folder.data and "RecordA6" not in folder.data


def test_remove_empty_epochs_then_renumber():
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=4, n_points=3)
    series.epochs.groups.assign("E2", 1)
    # Arrays deleted in the Data list: the epochs stay (E2 empty, E1 partly)
    for name in ("RecordA2", "RecordB2", "RecordA1"):
        folder.data.pop(name)
    assert folder.renumber_epochs_plan("Record") == ({}, {}, [])  # no gaps
    epochs, arrays, empty = folder.renumber_epochs_plan("Record", remove_empty=True)
    assert empty == ["E2"] and epochs == {"E3": "E2"}
    folder.renumber_dataseries_epochs("Record", remove_empty=True)
    assert list(series.epochs.keys()) == ["E0", "E1", "E2"]
    assert [d.name for d in series.epochs["E1"].data] == ["RecordB1"]  # partly empty: kept
    assert [d.name for d in series.epochs["E2"].data] == ["RecordA2", "RecordB2"]
    assert series.epochs.groups.get_group("E2") is None  # the removed epoch's group
    assert any("removed empty epochs: E2" in n["note"] for n in series.notes)
