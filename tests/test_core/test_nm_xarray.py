"""Tests for x values taken from another array (x array)."""
import copy

import numpy as np
import pytest

from pyneuromatic.core.nm_data import NMData, check_xarray
from pyneuromatic.core.nm_manager import NMManager


@pytest.fixture
def folder():
    """Series Record: channels A, B x epochs E0, E1 (4 points); arrays Times
    (0, 1.5, 2, 7), Short (2 points) and Back (not increasing)."""
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    folder.new_dataseries("Record", n_channels=2, n_epochs=2, n_points=4)
    folder.data.new("Times", nparray=np.array([0, 1.5, 2, 7]))
    folder.data.new("Short", nparray=np.array([0.0, 1.0]))
    folder.data.new("Back", nparray=np.array([0.0, 2.0, 1.0, 3.0]))
    return folder


def _arrays(folder):
    series = folder.dataseries["Record"]
    return [d for c in series.channels.values() for d in c.data]


# --- the x array is the array itself -------------------------------------------

def test_x_array_is_a_reference(folder):
    data, times = folder.data["RecordA0"], folder.data["Times"]
    data.set_xarray_from(times)
    assert data.xarray_source is times and data.xarray_name == "Times"
    assert data.xarray is times.nparray  # not a copy: no extra memory
    assert data.get_xvalue(3) == 7.0
    times.nparray[3] = 9.0  # changing Times changes the x values
    assert data.get_xvalue(3) == 9.0
    folder.data.rename("Times", "T")  # a rename keeps the link
    assert data.xarray_name == "T"
    data.set_xarray_from(None)
    assert data.xarray is None and data.xarray_source is None


def test_setting_values_directly_has_no_source(folder):
    data = folder.data["RecordA0"]
    data.set_xarray_from(folder.data["Times"])
    data.xarray = np.array([0.0, 1.0, 2.0, 3.0])
    assert data.xarray_source is None and data.xarray_name is None


def test_check_xarray(folder):
    data = folder.data["RecordA0"]
    with pytest.raises(ValueError, match="'Short' has 2 points, but 'RecordA0' has 4"):
        check_xarray(folder.data["Short"], data)
    with pytest.raises(ValueError, match="must increase"):
        check_xarray(folder.data["Back"], data)
    with pytest.raises(ValueError, match="NaN"):
        check_xarray(NMData(name="N", nparray=np.array([0.0, np.nan, 2.0, 3.0])), data)
    with pytest.raises(ValueError, match="not numeric"):
        check_xarray(NMData(name="T", nparray=np.array(["a", "b"], dtype=object)))
    with pytest.raises(TypeError):
        check_xarray(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="its own x array"):
        data.set_xarray_from(data)


def test_an_x_array_that_stops_fitting(folder):
    data, times = folder.data["RecordA0"], folder.data["Times"]
    data.set_xarray_from(times)
    times.nparray = np.array([0.0, 1.0])  # e.g. redimensioned
    assert data.xarray is None  # start and delta meanwhile
    assert data.get_xvalue(1) == 1.0
    assert "'Times' has 2 points, but 'RecordA0' has 4" in data.xarray_problem()
    times.nparray = np.array([0.0, 1.0, 2.0, 3.0])  # fits again
    assert data.xarray is times.nparray and data.xarray_problem() is None


# --- data series --------------------------------------------------------------

def test_series_set_xarray(folder):
    series = folder.dataseries["Record"]
    times = folder.data["Times"]
    assert series.set_xarray(times) == 4
    assert all(d.xarray_source is times for d in _arrays(folder))
    assert series.set_xarray(None, channel="B") == 2  # one channel back to start/delta
    assert all(d.xarray is None for d in series.channels["B"].data)
    assert all(d.xarray is times.nparray for d in series.channels["A"].data)


def test_series_set_xarray_is_all_or_nothing(folder):
    series = folder.dataseries["Record"]
    folder.data["RecordB1"].nparray = np.zeros(2)  # one array of another length
    with pytest.raises(ValueError, match="'RecordB1' has 2"):
        series.set_xarray(folder.data["Times"])
    assert all(d.xarray_source is None for d in _arrays(folder))
    with pytest.raises(ValueError, match="its own x array"):
        series.set_xarray(folder.data["RecordA0"])
    series.locked = True
    with pytest.raises(RuntimeError, match="locked"):
        series.set_xarray(None)


# --- deleting and locking -------------------------------------------------------

def test_an_x_array_in_use_cant_be_deleted(folder):
    series = folder.dataseries["Record"]
    series.set_xarray(folder.data["Times"])
    data = folder.data
    assert [d.name for d in data.xarray_users("Times")] == [
        "RecordA0", "RecordA1", "RecordB0", "RecordB1"]
    assert data.xarray_series("Times") == [series]
    with pytest.raises(RuntimeError, match="'Times' is the x array of data series 'Record'"):
        data.pop("Times")
    assert "Times" in data
    series.set_xarray(None)  # freed
    data.pop("Times")
    assert "Times" not in data


def test_an_x_array_of_plain_arrays(folder):
    folder.data["Short"].set_xarray_from(NMData(name="unused", nparray=np.array([0.0, 1.0])))
    folder.data["RecordA0"].set_xarray_from(folder.data["Back"].__class__(
        name="tmp", nparray=np.arange(4.0)))
    single = folder.data.new("Single", nparray=np.zeros(4))
    single.set_xarray_from(folder.data["Times"])
    assert folder.data.xarray_series("Times") == []
    assert "x array of 'Single'" in folder.data.xarray_in_use_text("Times")


def test_locked_series_protects_its_x_array(folder):
    series = folder.dataseries["Record"]
    series.set_xarray(folder.data["Times"])
    assert folder.data.locked_xarray_series("Times") is None
    series.locked = True
    assert folder.data.locked_xarray_series("Times") is series


# --- copies -------------------------------------------------------------------

def test_a_copied_folder_uses_its_copied_x_array(folder):
    folder.dataseries["Record"].set_xarray(folder.data["Times"])
    copied = copy.deepcopy(folder)
    copies = [d for c in copied.dataseries["Record"].channels.values() for d in c.data]
    assert all(d.xarray_source is copied.data["Times"] for d in copies)
    assert copied.data["Times"] is not folder.data["Times"]


def test_a_duplicated_array_shares_the_x_array(folder):
    data = folder.data["RecordA0"]
    data.set_xarray_from(folder.data["Times"])
    assert copy.deepcopy(data).xarray_source is folder.data["Times"]
