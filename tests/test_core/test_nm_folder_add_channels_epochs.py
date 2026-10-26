"""Tests for adding channels and epochs to a data series (NMFolder)."""
import math

import numpy as np
import pytest

from pyneuromatic.core.nm_manager import NMManager


@pytest.fixture
def folder():
    """Series Record: channels A, B x epochs E0..E2, 5 points, x step 0.1 ms."""
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    folder.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=5, dx=0.1,
                          x_units="ms", y_label="Vm", y_units="mV")
    return folder


def _names(container):
    return list(container.keys())


# --- channels ---------------------------------------------------------------

def test_add_channels(folder):
    series = folder.dataseries["Record"]
    added = folder.add_dataseries_channels("Record", 2, fill=1.5, y_label="Im", y_units="pA")
    assert [c.name for c in added] == ["C", "D"]
    assert _names(series.channels) == ["A", "B", "C", "D"]
    c = series.channels["C"]
    assert [d.name for d in c.data] == ["RecordC0", "RecordC1", "RecordC2"]
    d = folder.data["RecordC1"]
    assert list(d.nparray) == [1.5] * 5
    assert d.xscale.delta == 0.1 and d.xscale.units == "ms"  # as the epoch's arrays
    assert d.yscale.label == "Im" and d.yscale.units == "pA"
    assert d in series.epochs["E1"].data  # in its epoch too
    assert c.xscale.delta == 0.1 and c.yscale.units == "pA"


def test_add_channels_follows_the_epochs_there_are(folder):
    folder.remove_dataseries_epoch("Record", "E1", delete_data=True)
    folder.add_dataseries_channels("Record")
    assert [d.name for d in folder.dataseries["Record"].channels["C"].data] == [
        "RecordC0", "RecordC2"]


def test_add_channels_after_the_last_letter(folder):
    folder.remove_dataseries_channel("Record", "A")  # B is left: next is C
    added = folder.add_dataseries_channels("Record")
    assert added[0].name == "C"


def test_add_channels_noise_and_nan(folder):
    folder.add_dataseries_channels("Record", fill=np.nan)
    assert all(math.isnan(v) for v in folder.data["RecordC0"].nparray)
    folder.add_dataseries_channels("Record", fill=lambda n: np.arange(n, dtype=float))
    assert list(folder.data["RecordD2"].nparray) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_add_channels_refused(folder):
    series = folder.dataseries["Record"]
    folder.data.new("RecordC0", nparray=np.zeros(5))
    with pytest.raises(ValueError, match="'RecordC0' already exists"):
        folder.add_dataseries_channels("Record")
    assert _names(series.channels) == ["A", "B"]  # nothing changed
    assert "RecordC1" not in folder.data
    series.locked = True
    with pytest.raises(RuntimeError, match="locked"):
        folder.add_dataseries_channels("Record")
    series.locked = False
    with pytest.raises(ValueError, match="at most 26 channels"):
        folder.add_dataseries_channels("Record", 25)
    with pytest.raises(ValueError):
        folder.add_dataseries_channels("Record", 0)
    with pytest.raises(ValueError):
        folder.add_dataseries_channels("Missing")


# --- epochs -------------------------------------------------------------------

def test_add_epochs(folder):
    series = folder.dataseries["Record"]
    added = folder.add_dataseries_epochs("Record", 2, fill=2.0)
    assert [e.name for e in added] == ["E3", "E4"]
    assert _names(series.epochs) == ["E0", "E1", "E2", "E3", "E4"]
    assert [d.name for d in series.epochs["E3"].data] == ["RecordA3", "RecordB3"]
    d = folder.data["RecordB4"]
    assert list(d.nparray) == [2.0] * 5
    assert d.xscale.delta == 0.1 and d.yscale.label == "Vm"  # as the channel's last
    assert d in series.channels["B"].data
    assert series.epochs["E4"].number == 4


def test_add_epochs_after_the_last_number(folder):
    folder.remove_dataseries_epoch("Record", "E1", delete_data=True)  # E0, E2
    added = folder.add_dataseries_epochs("Record")
    assert added[0].name == "E3"
    assert _names(folder.dataseries["Record"].epochs) == ["E0", "E2", "E3"]


def test_add_epochs_refused(folder):
    series = folder.dataseries["Record"]
    folder.data.new("RecordB3", nparray=np.zeros(5))
    with pytest.raises(ValueError, match="'RecordB3' already exists"):
        folder.add_dataseries_epochs("Record")
    assert _names(series.epochs) == ["E0", "E1", "E2"]
    assert "RecordA3" not in folder.data  # nothing changed
    series.locked = True
    with pytest.raises(RuntimeError, match="locked"):
        folder.add_dataseries_epochs("Record")
