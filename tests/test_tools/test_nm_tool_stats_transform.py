"""Tests for Stats measuring the channel's filters and transforms."""
import numpy as np
import pytest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_transform import NMTransformBaseline, NMTransformInvert
import pyneuromatic.tools.nm_tool_stats as nms


@pytest.fixture
def setup():
    """Series Record: 2 channels x 3 epochs, mean = 10 * epoch + 1."""
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4)
    for channel in series.channels.values():
        for i, d in enumerate(channel.data):
            d.nparray = np.array([0.0, 1.0, 1.0, 2.0]) + 10 * i
    tool = nms.NMToolStats()
    list(tool.windows)[0].func = "mean"
    tool.results_to_cache = False
    return folder, series, tool


def _text(notes):
    return " | ".join(entry["note"] for entry in notes)


def _run(setup, channel_name):
    folder, series, tool = setup
    channel = series.channels[channel_name]
    targets = [{"folder": folder, "dataseries": series, "channel": channel,
                "epoch": epoch, "data": d}
               for epoch, d in zip(series.epochs.values(), channel.data)]
    tool.run_all(targets)
    return tool.toolfolder


def test_stats_measures_the_transformed_channel(setup):
    _folder, series, _tool = setup
    series.channels["A"].transform = [NMTransformInvert()]
    f = _run(setup, "A")
    assert list(f.data["ST_w0_mean_y"].nparray) == [-1.0, -11.0, -21.0]
    assert list(series.channels["A"].data[0].nparray) == [0.0, 1.0, 1.0, 2.0]  # unchanged
    # The results still name the arrays measured
    assert list(f.data["ST_w0_data"].nparray) == [d.path_str for d in series.channels["A"].data]


def test_steps_recorded_in_notes(setup):
    _folder, series, _tool = setup
    series.channels["A"].transform = [NMTransformInvert(),
                                      NMTransformBaseline(xbgn=0, xend=0)]
    f = _run(setup, "A")
    steps = "Invert() > Baseline(xbgn=0.0, xend=0.0, ignore_nans=True)"
    assert list(f.data["ST_w0_mean_y"].nparray) == [-1.0, -1.0, -1.0]
    assert ("transform=%s)" % steps) in _text(f.data["ST_w0_mean_y"].notes)
    assert ("channel transform: %s" % steps) in _text(f.notes)  # the results folder


def test_no_transform_no_note(setup):
    f = _run(setup, "B")
    assert list(f.data["ST_w0_mean_y"].nparray) == [1.0, 11.0, 21.0]
    assert "transform" not in _text(f.data["ST_w0_mean_y"].notes)
    assert "transform" not in _text(f.notes)


def test_each_channel_uses_its_own(setup):
    _folder, series, _tool = setup
    series.channels["A"].transform = [NMTransformInvert()]
    f = _run(setup, "B")
    assert list(f.data["ST_w0_mean_y"].nparray) == [1.0, 11.0, 21.0]


def test_steps_turned_off_are_not_used(setup):
    _folder, series, _tool = setup
    off = NMTransformInvert()
    off.on = False
    series.channels["A"].transform = [off]
    f = _run(setup, "A")
    assert list(f.data["ST_w0_mean_y"].nparray) == [1.0, 11.0, 21.0]
    assert "transform" not in _text(f.data["ST_w0_mean_y"].notes)


def test_data_outside_a_series_is_measured_as_it_is(setup):
    folder, series, tool = setup
    series.channels["A"].transform = [NMTransformInvert()]
    d = series.channels["A"].data[1]
    tool.run_all([{"folder": folder, "data": d}])  # no channel selected
    assert list(tool.toolfolder.data["ST_w0_mean_y"].nparray) == [11.0]


# --- the results folder's note: how the run was made ------------------------

def test_results_folder_note_describes_the_run(setup):
    folder, series, tool = setup
    w0 = list(tool.windows)[0]
    w0.xbgn, w0.xend = 0.0, 3.0
    w0.bsln_func = {"name": "mean"}
    w0.bsln_xbgn, w0.bsln_xend = 0.0, 1.0
    w0.bsln_on = True
    f = _run(setup, "A")
    assert _text(f.notes) == ("NMStats: Record, channel A, epochs selected (n=3); "
                              "w0: mean, x 0 to 3, baseline mean x 0 to 1")


def test_results_folder_note_with_a_transform(setup):
    _folder, series, _tool = setup
    series.channels["A"].transform = [NMTransformInvert()]
    f = _run(setup, "A")
    assert _text(f.notes) == ("NMStats: Record, channel A, epochs selected (n=3); "
                              "w0: mean, x -inf to inf; channel transform: Invert()")


def test_results_folder_note_for_data_arrays(setup):
    folder, series, tool = setup
    arrays = series.channels["A"].data[:2]
    tool.run_all([{"folder": folder, "data": d} for d in arrays], run_keys={"data": "Set1"})
    assert _text(tool.toolfolder.notes) == (
        "NMStats: 2 data arrays (data set Set1); w0: mean, x -inf to inf")
