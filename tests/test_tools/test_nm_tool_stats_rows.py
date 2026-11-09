"""Tests for the ST_ arrays' rows: ST_data_names, ST_epoch_num, every window
lined up with them; crossing levels and percentages saved (#476)."""

import math

import numpy as np
import pytest

from pyneuromatic.core.nm_manager import NMManager

N = 5


def _nm(all_epochs=False):
    """Data series Record: channel A x 5 epochs, a peak of 15 above -70 at x 80;
    Stats w0 max, w1 mean."""
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=N, n_points=200)
    x = np.arange(200.0)
    for d in series.channels["A"].data:
        d.nparray = -70.0 + 15.0 * np.exp(-((x - 80.0) / 12.0) ** 2)
        d.xscale.units, d.yscale.units = "ms", "mV"
    series.epochs.sets.add("Set1", ["E0", "E2", "E4"])
    nm.tool_add("stats")
    stats = nm.stats
    stats.windows["w0"].func = "max"
    stats.windows.new()
    stats.windows["w1"].func = "mean"
    stats.results_to_numpy = True
    stats.results_all_epochs = all_epochs
    nm.series = series
    return nm


def _arrays(nm):
    return nm.stats.toolfolder.data


def _names(nm):
    return [str(p).rsplit(".", 1)[-1] for p in _arrays(nm)["ST_data_names"].nparray]


def test_one_rows_array_for_all_windows():
    nm = _nm()
    nm.stats.run_selection(nm.series, channel="A", epochs="Set1")
    data = _arrays(nm)
    assert "ST_data_names" in data and "ST_epoch_num" in data
    assert not any(name.endswith("_data") for name in data.keys())  # no ST_w*_data
    assert _names(nm) == ["RecordA0", "RecordA2", "RecordA4"]
    assert data["ST_epoch_num"].nparray.dtype.kind == "i"
    assert list(data["ST_epoch_num"].nparray) == [0, 2, 4]
    for name in ("ST_w0_max_y", "ST_w1_mean_y"):
        assert len(data[name].nparray) == 3


def test_a_window_without_a_result_has_nan_in_that_row(monkeypatch):
    nm = _nm()
    w1 = nm.stats.windows["w1"]
    compute = w1.compute

    def no_result_for_e2(data, **kwargs):  # e.g. nothing found
        results = compute(data, **kwargs)
        if data.name == "RecordA2":
            results.clear()
        return results

    monkeypatch.setattr(w1, "compute", no_result_for_e2)
    nm.stats.run_selection(nm.series, channel="A", epochs="Set1")
    data = _arrays(nm)
    w1_mean = data["ST_w1_mean_y"].nparray
    assert len(w1_mean) == 3 and math.isnan(w1_mean[1]) and not math.isnan(w1_mean[0])
    assert not np.isnan(data["ST_w0_max_y"].nparray).any()


@pytest.mark.parametrize("epochs, group, numbers", [
    ("all", None, [0, 1, 2, 3, 4]),
    ("Set1", None, [0, 2, 4]),
    ("E3", None, [3]),
])
def test_epoch_numbers(epochs, group, numbers):
    nm = _nm()
    nm.stats.run_selection(nm.series, channel="A", epochs=epochs, group=group)
    assert list(_arrays(nm)["ST_epoch_num"].nparray) == numbers


def test_epoch_numbers_for_a_group():
    nm = _nm()
    nm.series.epochs.groups.assign(["E1", "E3"], 1)
    nm.stats.run_selection(nm.series, channel="A", group=1)
    assert list(_arrays(nm)["ST_epoch_num"].nparray) == [1, 3]


def test_an_omitted_epoch_has_no_row():
    nm = _nm()
    nm.stats.progress = lambda done, total: (done == 2 and nm.stats.omit_last()) or True
    nm.stats.run_selection(nm.series, channel="A", epochs="Set1")  # E2 omitted
    assert list(_arrays(nm)["ST_epoch_num"].nparray) == [0, 4]
    assert _names(nm) == ["RecordA0", "RecordA4"]


def test_one_point_per_epoch():
    nm = _nm(all_epochs=True)
    nm.stats.progress = lambda done, total: (done == 2 and nm.stats.omit_last()) or True
    nm.stats.run_selection(nm.series, channel="A", epochs="Set1")  # E0, (E2 omitted), E4
    data = _arrays(nm)
    assert list(data["ST_epoch_num"].nparray) == [0, 1, 2, 3, 4]  # every epoch, no NaN
    assert _names(nm) == [f"RecordA{i}" for i in range(N)]
    values = data["ST_w0_max_y"].nparray
    assert [not math.isnan(v) for v in values] == [True, False, False, False, True]
    assert "per epoch" in data["ST_w0_max_y"].notes[0]["note"]


def test_a_data_array_run_has_no_epoch_numbers():
    nm = _nm()
    folder = nm.folders["Demo"]
    nm.stats.run_selection(folder, data=["RecordA0", "RecordA1"])
    data = _arrays(nm)
    assert "ST_data_names" in data and "ST_epoch_num" not in data


def test_crossing_levels_and_percentages_are_saved():
    nm = _nm()
    w0 = nm.stats.windows["w0"]
    w0.func = {"name": "risetime+", "p0": 10, "p1": 90}
    w0.bsln_on = True
    w0.bsln_func = "mean"
    w0.bsln_xbgn, w0.bsln_xend = 0.0, 40.0
    w1 = nm.stats.windows["w1"]
    w1.func = {"name": "level+", "ylevel": -60}
    w1.xbgn, w1.xend = 60.0, 120.0
    nm.stats.run_selection(nm.series, channel="A", epochs="E0")
    data = _arrays(nm)
    p0 = data["ST_w0_rt_p_p0_y"]
    assert p0.nparray[0] == pytest.approx(-68.5) and p0.yscale.units == "mV"
    assert "p0=10" in p0.notes[0]["note"] and "stepfunc=level+" in p0.notes[0]["note"]
    assert "p1=90" in data["ST_w0_rt_p_p1_y"].notes[0]["note"]
    assert "stepfunc=max" in data["ST_w0_rt_p_y"].notes[0]["note"]  # the peak
    level = data["ST_w1_level_p_y"]
    assert level.nparray[0] == -60.0 and level.yscale.units == "mV"
    assert "ylevel=-60" in level.notes[0]["note"]
