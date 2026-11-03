"""Tests for import options: prefix, channels, epochs, X range (#456)."""

from pathlib import Path

import numpy as np
import pytest

from pyneuromatic.core.nm_folder import file_info
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.io.base import check_options, crop, import_numbers

FIXTURES = Path(__file__).parent / "fixtures"
PXP = FIXTURES / "nm02Jul04c0_002.pxp"
ABF = FIXTURES / "15804044.abf"
AXGD = FIXTURES / "Vers6_060523 004.axgd"


def _needs(path):
    if path.suffix == ".pxp":
        pytest.importorskip("igor2")
    if path.suffix == ".abf":
        pytest.importorskip("pyabf")


# --- helpers -----------------------------------------------------------------

def test_crop_keeps_the_points_in_range_and_moves_the_start():
    y = np.arange(10.0)  # x = 1.0, 1.5, ... 5.5
    assert crop(y, 1.0, 0.5, None)[0] is y
    part, start = crop(y, 1.0, 0.5, (2.0, 3.0))
    assert list(part) == [2.0, 3.0, 4.0] and start == 2.0
    part, start = crop(y, 1.0, 0.5, (None, 1.6))
    assert list(part) == [0.0, 1.0] and start == 1.0
    part, start = crop(y, 1.0, 0.5, (5.2, None))
    assert list(part) == [9.0] and start == 5.5
    assert len(crop(y, 1.0, 0.5, (20.0, None))[0]) == 0


def test_import_numbers_renumber_from_a_and_zero():
    channels, epochs = import_numbers(["A", "B", "C"], [0, 1, 2, 3], ["c", "A"], [3, 1])
    assert channels == {"A": 0, "C": 1} and epochs == {1: 0, 3: 1}
    assert import_numbers(["A", "B"], [0, 1], None, None) == ({"A": 0, "B": 1}, {0: 0, 1: 1})
    with pytest.raises(ValueError):
        import_numbers(["A"], [0], ["B"], None)
    with pytest.raises(ValueError):
        import_numbers(["A"], [0], None, [5])


def test_check_options():
    check_options(["A"], range(0, 3), (0.0, None))
    with pytest.raises(TypeError):
        check_options("A", None, None)  # a list of letters
    with pytest.raises(TypeError):
        check_options(None, ["E0"], None)  # numbers
    with pytest.raises(ValueError):
        check_options(None, None, (5.0, 1.0))


# --- readers -----------------------------------------------------------------

@pytest.mark.parametrize("path, channels, epochs, letters, n_epochs, file_format", [
    (PXP, [("A", "Vmem", "mV"), ("B", "Icmd", "pA")], 19, ["A", "B"], 19,
     "Igor Pro packed experiment"),
    (ABF, [("A", "Im_1stCh2", "pA"), ("B", "Light", "V")], 10, ["A", "B"], 10,
     "Axon Binary Format (ABF 2."),
    (AXGD, None, 2, ["A", "B", "C"], 2, "AxoGraph"),
])
def test_file_info(path, channels, epochs, letters, n_epochs, file_format):
    _needs(path)
    info = file_info(str(path))
    assert info.file_format.startswith(file_format)
    assert [c[0] for c in info.channels] == letters
    if channels is not None:
        assert info.channels == channels
    assert info.epochs == list(range(n_epochs))
    assert info.prefix == "Record" and info.n_points > 0 and info.x_delta > 0


@pytest.mark.parametrize("path", [PXP, ABF, AXGD])
def test_channels_epochs_and_xrange(path):
    """Channel B, epoch 1, X 1..2: renumbered as channel A, E0, keeping
    channel B's label; cropped from X 1."""
    _needs(path)
    nm = NMManager(quiet=True)
    info = file_info(str(path))
    whole = nm.folders.import_file(str(path))
    part = nm.folders.import_file(str(path), prefix="Rec", channels=["B"],
                                  epochs=[1], xrange=(1.0, 2.0))
    assert list(part.data.keys()) == ["RecA0"]
    series = part.dataseries["Rec"]
    assert list(series.channels.keys()) == ["A"] and list(series.epochs.keys()) == ["E0"]
    first, source = part.data["RecA0"], whole.data["RecordB1"]
    assert first.yscale.label == info.channels[1][1]  # channel B's
    expected, start = crop(source.nparray, source.xscale.start, source.xscale.delta, (1.0, 2.0))
    assert np.array_equal(first.nparray, expected)
    assert first.xscale.start == pytest.approx(start)
    assert part.metadata["import"] == {"channels": ["B"], "epochs": [1], "xrange": [1.0, 2.0]}
    assert "import" not in whole.metadata


def test_no_data_series(tmp_path):
    nm = NMManager(quiet=True)
    folder = nm.folders.import_file(str(AXGD), make_dataseries=False)
    assert len(folder.data) == 6 and len(folder.dataseries) == 0


def test_imported_arrays_are_copies_of_the_file():
    """The file is kept between describing and importing it: editing an
    imported array must not change a later import."""
    _needs(PXP)
    nm = NMManager(quiet=True)
    file_info(str(PXP))
    first = nm.folders.import_file(str(PXP), channels=["A"], epochs=[0])
    first.data["RecordA0"].nparray[:] = 0.0
    file_info(str(PXP))
    second = nm.folders.import_file(str(PXP), channels=["A"], epochs=[0])
    assert np.any(second.data["RecordA0"].nparray != 0.0)


# --- the command ---------------------------------------------------------------

def test_command_writes_only_the_options_given():
    nm = NMManager(quiet=True, command_history=True)
    nm.folders.import_file(str(AXGD))
    nm.folders.import_file(str(AXGD), name="Cell2", prefix="Rec", channels=["A", "C"],
                           epochs=range(1, 2), xrange=(0.0, None), make_dataseries=False)
    commands = [e["command"] for e in nm.command_history.buffer][-2:]
    assert commands == [
        "nm.folders.import_file(%r)" % str(AXGD),
        "nm.folders.import_file(%r, name='Cell2', prefix='Rec', channels=['A', 'C'], "
        "epochs=range(1, 2), xrange=(0.0, None), make_dataseries=False)" % str(AXGD),
    ]


def test_command_with_options_replays():
    nm = NMManager(quiet=True, command_history=True)
    folder = nm.folders.import_file(str(AXGD), channels=["C"], xrange=(10.0, 20.0))
    other = NMManager(quiet=True)
    exec(nm.command_history.buffer[-1]["command"], {"nm": other})
    copy = other.folders[folder.name]
    assert list(copy.data.keys()) == list(folder.data.keys()) == ["RecordA0", "RecordA1"]
    assert all(np.array_equal(copy.data[k].nparray, folder.data[k].nparray)
               for k in folder.data.keys())


def test_bad_options_leave_nothing():
    nm = NMManager(quiet=True, command_history=True)
    with pytest.raises(ValueError):
        nm.folders.import_file(str(AXGD), channels=["Z"])
    assert len(nm.folders) == 0
