"""Tests for moving a data series to another folder, and appending one series
to another (NMFolder.move_dataseries / append_dataseries)."""
import numpy as np
import pytest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_transform import NMTransformInvert


@pytest.fixture
def folders():
    """Folder A: series Record (channels A, B x E0..E2, 4 points), Set0 =
    E0, E2, E1 in group 1, a transform on channel A, a note; folder B empty."""
    nm = NMManager(quiet=True)
    a = nm.folders.new("A")
    b = nm.folders.new("B")
    series = a.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4, dx=0.1,
                              x_units="ms")
    series.epochs.sets.add("Set0", ["E0", "E2"])
    series.epochs.groups.assign("E1", 1)
    series.channels["A"].transform = [NMTransformInvert()]
    series.notes.add("slice 3, cell 2")
    return a, b


def _arrays(series):
    return [d.name for c in series.channels.values() for d in c.data]


# --- move ---------------------------------------------------------------------

def test_move_to_another_folder(folders):
    a, b = folders
    original = a.data["RecordA1"]
    moved = a.move_dataseries("Record", b)
    assert "Record" not in a.dataseries and list(b.dataseries.keys()) == ["Record"]
    assert len(a.data) == 0 and len(b.data) == 6
    assert b.data["RecordA1"] is original and original._parent is b  # the same arrays
    assert _arrays(moved) == ["RecordA0", "RecordA1", "RecordA2",
                              "RecordB0", "RecordB1", "RecordB2"]
    assert [d.name for d in moved.epochs["E1"].data] == ["RecordA1", "RecordB1"]
    assert moved.epochs.sets.get_items("Set0", get_keys=True) == ["E0", "E2"]
    assert moved.epochs.groups.get_group("E1") == 1
    assert moved.channels["A"].transform_str() == "Invert()"
    assert moved.notes.note == "slice 3, cell 2"
    assert b.data["RecordB2"].xscale.delta == 0.1


def test_move_with_a_new_prefix(folders):
    a, b = folders
    b.data.new("RecordA0", nparray=np.zeros(4))  # a clash in B
    with pytest.raises(ValueError, match="already has data 'RecordA0'"):
        a.move_dataseries("Record", b)
    assert "Record" in a.dataseries and len(a.data) == 6  # nothing changed
    moved = a.move_dataseries("Record", b, new_prefix="Record2")
    assert moved.name == "Record2"
    assert _arrays(moved)[:2] == ["Record2A0", "Record2A1"]


def test_move_refused(folders):
    a, b = folders
    with pytest.raises(ValueError, match="already in folder"):
        a.move_dataseries("Record", a)
    b.new_dataseries("Record", n_channels=1, n_epochs=1, n_points=4)
    with pytest.raises(ValueError, match="already has a data series 'Record'"):
        a.move_dataseries("Record", b)
    a.dataseries["Record"].locked = True
    with pytest.raises(RuntimeError, match="locked"):
        a.move_dataseries("Record", b, new_prefix="R2")


# --- x arrays stay in the folder of the arrays that use them -----------------

def _times(folder):
    return folder.data.new("Times", nparray=np.array([0.0, 1.0, 2.0, 5.0]))


def test_move_takes_its_x_array_along(folders):
    a, b = folders
    times = _times(a)
    a.dataseries["Record"].set_xarray(times)
    assert a.move_dataseries_note("Record") == "X array 'Times' moves too."
    moved = a.move_dataseries("Record", b)
    assert "Times" not in a.data and b.data["Times"] is times  # moved, not copied
    assert all(d.xarray_source is times for c in moved.channels.values() for d in c.data)
    assert len(a.data) == 0  # folder A can now go without hiding anything


def test_move_takes_a_copy_of_a_shared_x_array(folders):
    a, b = folders
    times = _times(a)
    a.dataseries["Record"].set_xarray(times)
    stays = a.data.new("Stays", nparray=np.zeros(4))
    stays.set_xarray_from(times)
    assert a.move_dataseries_note("Record") == (
        "A copy of x array 'Times' goes too (it stays here for 'Stays').")
    moved = a.move_dataseries("Record", b)
    assert a.data["Times"] is times and stays.xarray_source is times  # left for Stays
    copy = b.data["Times"]
    assert copy is not times and list(copy.nparray) == list(times.nparray)
    assert all(d.xarray_source is copy for c in moved.channels.values() for d in c.data)


def test_move_refused_if_the_x_array_name_is_taken(folders):
    a, b = folders
    a.dataseries["Record"].set_xarray(_times(a))
    b.data.new("Times", nparray=np.zeros(2))
    with pytest.raises(ValueError, match="already has data 'Times' .the series' x array."):
        a.move_dataseries("Record", b)
    assert "Record" in a.dataseries


def test_x_array_must_be_in_the_same_folder(folders):
    a, b = folders
    times = _times(b)
    with pytest.raises(ValueError, match="must be in the same folder as 'RecordA0'"):
        a.data["RecordA0"].set_xarray_from(times)
    with pytest.raises(ValueError, match="same folder"):
        a.dataseries["Record"].set_xarray(times)


def test_copies_to_another_folder_bring_their_x_array(folders):
    a, b = folders
    times = _times(a)
    a.dataseries["Record"].set_xarray(times)
    copy = a.data.copy_to("RecordA0", b.data, "One")
    assert copy.xarray_source is b.data["Times"] and b.data["Times"] is not times
    series = a.copy_dataseries("Record", "Rec2", folder=b)
    # The same Times copy is used again (same name and values)
    assert all(d.xarray_source is b.data["Times"] for c in series.channels.values()
               for d in c.data)
    assert "Times_1" not in b.data
    b.data["Times"].nparray = np.array([0.0, 1.0, 2.0, 9.0])  # now different
    assert a.data.copy_to("RecordA1", b.data, "Two").xarray_source.name == "Times_1"


def test_appended_arrays_take_the_series_x_array(folders, stim):
    a, _b = folders
    record = a.dataseries["Record"]
    record.set_xarray(_times(a))
    assert a.append_dataseries_note("Record", "Stim") == (
        "The appended arrays use x array 'Times', as 'Record' does ('Stim' used "
        "Start and Delta).")
    a.append_dataseries("Record", "Stim")
    assert all(d.xarray_name == "Times" for c in record.channels.values() for d in c.data)


def test_move_refused_if_an_array_is_an_x_array_left_behind(folders):
    a, b = folders
    a.data["RecordA0"].nparray = np.arange(4.0)  # increasing, so an x array
    other = a.data.new("Other", nparray=np.zeros(4))
    other.set_xarray_from(a.data["RecordA0"])
    with pytest.raises(RuntimeError, match="'RecordA0' is the x array of 'Other'"):
        a.move_dataseries("Record", b)
    assert "RecordA0" in a.data


# --- append -------------------------------------------------------------------

@pytest.fixture
def stim(folders):
    """Series Stim in folder A: channels A, B x E0, E1; Set1 = E1, E0 in group 0."""
    a, _b = folders
    series = a.new_dataseries("Stim", n_channels=2, n_epochs=2, n_points=4)
    for d in series.channels["A"].data:
        d.nparray = np.full(4, 7.0)
    series.epochs.sets.add("Set1", ["E1"])
    series.epochs.groups.assign("E0", 0)
    return series


def test_append_copies_and_keeps_the_source(folders, stim):
    a, _b = folders
    record = a.dataseries["Record"]
    added = a.append_dataseries("Record", "Stim")
    assert [e.name for e in added] == ["E3", "E4"]
    assert list(record.epochs.keys()) == ["E0", "E1", "E2", "E3", "E4"]
    assert [d.name for d in record.epochs["E4"].data] == ["RecordA4", "RecordB4"]
    assert list(a.data["RecordA3"].nparray) == [7.0] * 4
    assert a.data["RecordA3"] is not a.data["StimA0"]  # a copy
    assert "Stim" in a.dataseries and "StimA0" in a.data  # the source is kept
    # Sets and groups come along, renumbered
    assert record.epochs.sets.get_items("Set1", get_keys=True) == ["E4"]
    assert record.epochs.groups.get_group("E3") == 0
    assert record.epochs.groups.get_group("E1") == 1  # its own unchanged


def test_append_moves_and_deletes_the_source(folders, stim):
    a, _b = folders
    original = a.data["StimB1"]
    a.append_dataseries("Record", "Stim", delete_source=True)
    assert "Stim" not in a.dataseries and "StimB1" not in a.data
    assert a.data["RecordB4"] is original  # moved, not copied


def test_append_after_the_last_epoch(folders, stim):
    a, _b = folders
    a.remove_dataseries_epoch("Record", "E1", delete_data=True)  # E0, E2
    added = a.append_dataseries("Record", "Stim")
    assert [e.name for e in added] == ["E3", "E4"]


def test_append_refused(folders, stim):
    a, _b = folders
    a.new_dataseries("Odd", n_channels=3, n_epochs=1, n_points=4)
    with pytest.raises(ValueError, match="channels differ: 'Record' has A, B, 'Odd' has A, B, C"):
        a.append_dataseries("Record", "Odd")
    with pytest.raises(ValueError, match="to itself"):
        a.append_dataseries("Record", "Record")
    a.data.new("RecordA3", nparray=np.zeros(4))
    with pytest.raises(ValueError, match="'RecordA3' already exists"):
        a.append_dataseries("Record", "Stim")
    assert list(a.dataseries["Record"].epochs.keys()) == ["E0", "E1", "E2"]  # unchanged


def test_append_and_locks(folders, stim):
    a, _b = folders
    a.dataseries["Record"].locked = True
    with pytest.raises(RuntimeError, match="locked"):
        a.append_dataseries("Record", "Stim")
    a.dataseries["Record"].locked = False
    stim.locked = True
    with pytest.raises(RuntimeError, match="locked"):
        a.append_dataseries("Record", "Stim", delete_source=True)
    a.append_dataseries("Record", "Stim")  # a locked source can still be copied
    assert len(a.dataseries["Record"].epochs) == 5


def test_append_from_another_folder(folders):
    a, b = folders
    b.new_dataseries("Late", n_channels=2, n_epochs=1, n_points=4)
    a.append_dataseries("Record", "Late", source_folder=b, delete_source=True)
    assert "RecordA3" in a.data and a.data["RecordA3"]._parent is a
    assert "Late" not in b.dataseries and len(b.data) == 0


# --- copies keep the epoch sets, groups, transforms and notes ---------------

def test_copy_to_another_folder_keeps_sets_and_groups(folders):
    a, b = folders
    copy = a.copy_dataseries("Record", folder=b)
    assert copy.epochs.sets.get_items("Set0", get_keys=True) == ["E0", "E2"]
    assert copy.epochs.groups.get_group("E1") == 1
    assert copy.channels["A"].transform_str() == "Invert()"
    assert copy.notes.note == "slice 3, cell 2"
    # Independent of the original
    copy.epochs.sets.add("Set1", ["E1"])
    assert a.dataseries["Record"].epochs.sets.get_items("Set1", get_keys=True) == []


def test_partial_copy_keeps_its_members(folders):
    a, _b = folders
    copy = a.copy_dataseries("Record", "Part", channel="A", epoch=["E1", "E2"])
    assert list(copy.epochs.keys()) == ["E0", "E1"]  # numbered afresh
    assert copy.epochs.sets.get_items("Set0", get_keys=True) == ["E1"]  # was E2
    assert copy.epochs.groups.get_group("E0") == 1  # was E1
    assert copy.channels["A"].transform_str() == "Invert()"


# --- moving one data array ---------------------------------------------------

def test_move_data(folders):
    a, b = folders
    single = a.data.new("Single", nparray=np.arange(4.0))
    moved = a.move_data("Single", b)
    assert moved is single and single._parent is b
    assert "Single" not in a.data and b.data["Single"] is single
    renamed = b.move_data("Single", a, new_name="Back")
    assert renamed.name == "Back" and a.data["Back"] is single


def test_move_data_takes_its_x_array(folders):
    a, b = folders
    times = _times(a)
    one = a.data.new("One", nparray=np.zeros(4))
    one.set_xarray_from(times)
    assert a.move_data_note("One", b) == "X array 'Times' moves too."
    a.move_data("One", b)
    assert b.data["Times"] is times and one.xarray_source is times
    # Shared: a copy goes; an identical one there is used again
    two = b.data.new("Two", nparray=np.zeros(4))
    two.set_xarray_from(times)
    assert "copy" in b.move_data_note("Two", a)
    b.move_data("Two", a)
    assert two.xarray_source is a.data["Times"] and a.data["Times"] is not times
    three = b.data.new("Three", nparray=np.zeros(4))
    three.set_xarray_from(times)
    assert b.move_data_note("Three", a) == "It uses the 'Times' already in that folder."
    b.move_data("Three", a)
    assert three.xarray_source is a.data["Times"]


def test_move_data_refused(folders):
    a, b = folders
    with pytest.raises(RuntimeError, match="part of data series 'Record'"):
        a.move_data("RecordA0", b)
    times = _times(a)
    a.data.new("One", nparray=np.zeros(4)).set_xarray_from(times)
    with pytest.raises(RuntimeError, match="cannot be moved while in use"):
        a.move_data("Times", b)
    b.data.new("One", nparray=np.zeros(4))
    with pytest.raises(ValueError, match="already has data 'One'"):
        a.move_data("One", b)
    b.data.new("Times", nparray=np.zeros(4))  # a different Times there
    with pytest.raises(ValueError, match="already has a different 'Times'"):
        a.move_data("One", b, new_name="One2")
    with pytest.raises(ValueError, match="already in folder"):
        a.move_data("One", a)
