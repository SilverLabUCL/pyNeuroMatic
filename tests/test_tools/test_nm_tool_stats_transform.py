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


def test_no_transform_no_note(setup):
    f = _run(setup, "B")
    assert list(f.data["ST_w0_mean_y"].nparray) == [1.0, 11.0, 21.0]
    assert "transform" not in _text(f.data["ST_w0_mean_y"].notes)


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
