"""Tests for channel filters and transforms (NMChannel.transform)."""
import copy

import numpy as np
import pytest

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_transform import (
    NMTransformBaseline,
    NMTransformInvert,
    NMTransformLn,
    _transform_from_dict,
    apply_transforms,
)
import pyneuromatic.core.nm_utilities as nmu


@pytest.fixture
def series():
    """Series Record: 2 channels x 2 epochs, y = 1, 2, 3, 4 (x from 0, step 1)."""
    nm = NMManager(quiet=True)
    folder = nm.folders.new("Demo")
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=2, n_points=4)
    for channel in series.channels.values():
        for d in channel.data:
            d.nparray = np.array([1.0, 2.0, 3.0, 4.0])
    return series


# --- on / off ---------------------------------------------------------------

def test_transform_on_by_default():
    t = NMTransformInvert()
    assert t.on is True
    assert t.to_dict() == {"type": "NMTransformInvert"}  # "on" only when off


def test_transform_off_round_trip():
    t = NMTransformBaseline(xbgn=0, xend=1)
    t.on = False
    d = t.to_dict()
    assert d["on"] is False
    t2 = _transform_from_dict(d)
    assert t2.on is False and t2.xend == 1.0
    for bad in nmu.badtypes(ok=[True]):
        with pytest.raises(TypeError):
            t.on = bad


def test_apply_skips_transforms_turned_off():
    y = np.array([1.0, 2.0])
    off = NMTransformInvert()
    off.on = False
    assert list(apply_transforms(y, [off])) == [1.0, 2.0]
    assert list(apply_transforms(y, [off, NMTransformInvert()])) == [-1.0, -2.0]


# --- the channel's list -----------------------------------------------------

def test_default_is_empty(series):
    channel = series.channels["A"]
    assert channel.transform == []
    assert not channel.transform_on
    assert channel.transform_str() == ""


def test_set_from_objects_or_dicts(series):
    channel = series.channels["A"]
    channel.transform = [NMTransformInvert(),
                         {"type": "NMTransformBaseline", "xbgn": 0, "xend": 1}]
    assert [t.type_str for t in channel.transform] == ["NMTransformInvert",
                                                       "NMTransformBaseline"]
    assert all(t._parent is channel for t in channel.transform)
    assert channel.parameters["transform"][1]["xend"] == 1.0
    channel.transform = None
    assert channel.transform == []
    for bad in nmu.badtypes(ok=[None, []]):
        with pytest.raises(TypeError):
            channel.transform = bad
    with pytest.raises(TypeError):
        channel.transform = ["Invert"]
    with pytest.raises(ValueError):
        channel.transform = [{"type": "NMTransformUnknown"}]


def test_constructor_takes_a_list():
    channel = NMChannel(name="A", transform=[{"type": "NMTransformInvert"}])
    assert channel.transform_str() == "Invert()"


def test_transform_str_lists_steps_that_are_on(series):
    channel = series.channels["A"]
    off = NMTransformLn()
    off.on = False
    channel.transform = [NMTransformInvert(), off, NMTransformBaseline(xbgn=0, xend=1)]
    assert channel.transform_str() == (
        "Invert() > Baseline(xbgn=0.0, xend=1.0, ignore_nans=True)"
    )


def test_eq_and_deepcopy(series):
    a, b = series.channels["A"], series.channels["B"]
    a.transform = [NMTransformInvert()]
    copied = copy.deepcopy(a)
    assert copied.transform == a.transform
    assert copied.transform[0] is not a.transform[0]
    assert copied.transform[0]._parent is copied
    assert copied == a
    copied.transform[0].on = False
    assert copied != a  # on / off counts
    assert b.transform == []


# --- transformed(): what the tools analyse ----------------------------------

def test_transformed_is_a_copy(series):
    channel = series.channels["A"]
    data = channel.data[0]
    channel.transform = [NMTransformInvert(), NMTransformBaseline(xbgn=0, xend=0)]
    result = channel.transformed(data)
    assert result is not data
    assert list(result.nparray) == [0.0, -1.0, -2.0, -3.0]  # -y, less -1 at x=0
    assert list(data.nparray) == [1.0, 2.0, 3.0, 4.0]  # unchanged
    assert result.name == data.name and result.path_str == data.path_str
    assert result.xscale == data.xscale and result.yscale == data.yscale
    folder = data._parent
    assert folder.data.get(data.name) is data  # the copy isn't added to the folder


def test_transformed_without_steps_is_the_data(series):
    channel = series.channels["A"]
    data = channel.data[0]
    assert channel.transformed(data) is data
    off = NMTransformInvert()
    off.on = False
    channel.transform = [off]
    assert channel.transformed(data) is data
    with pytest.raises(TypeError):
        channel.transformed(data.nparray)


def test_other_channels_unchanged(series):
    series.channels["A"].transform = [NMTransformInvert()]
    b = series.channels["B"]
    assert b.transformed(b.data[0]) is b.data[0]
