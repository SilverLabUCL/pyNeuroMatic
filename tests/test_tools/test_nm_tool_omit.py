"""Tests for omitting epochs from a tool run (#472)."""

import numpy as np
import pytest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.tools.nm_tool import NMTool

N = 5
SERIES = "nm.folders['Demo'].dataseries['Record']"


def _nm(all_epochs=False):
    """Data series Record: channel A x 5 epochs, y = epoch number; Stats mean."""
    nm = NMManager(quiet=True, command_history=True)
    folder = nm.folders.new("Demo")
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=N, n_points=5)
    for i, d in enumerate(series.channels["A"].data):
        d.nparray = np.full(5, float(i))
    nm.tool_add("stats")
    nm.stats.windows["w0"].func = "mean"
    nm.stats.results_to_numpy = True
    nm.stats.results_all_epochs = all_epochs
    nm.series = series
    return nm


def _commands(nm):
    return [e["command"] for e in nm.command_history.buffer]


def _means(nm):
    return list(nm.stats.toolfolder.data.get("ST_w0_mean_y").nparray)


def _omit_at(nm, *positions):
    """Omit the targets run as these (1, 2, ...), from the progress callback."""
    nm.stats.progress = lambda done, total: (done in positions and nm.stats.omit_last()) or True


def test_omit_argument_leaves_epochs_out():
    nm = _nm()
    nm.stats.run_selection(nm.series, channel="A", epochs="all", omit=["E1", "E3"])
    assert _means(nm) == [0.0, 2.0, 4.0]
    assert _commands(nm)[-1] == (
        f"nm.stats.run_selection({SERIES}, channel='A', epochs='all', omit=['E1', 'E3'])")


def test_omit_during_the_run():
    nm = _nm()
    count = len(_commands(nm))
    _omit_at(nm, 2)  # E1
    nm.stats.run_selection(nm.series, channel="A", epochs="all")
    assert _means(nm) == [0.0, 2.0, 3.0, 4.0]
    assert nm.stats.omitted == ["E1"]
    assert "omitted: E1" in [n["note"] for n in nm.stats.toolfolder.notes]
    # One command, recorded as the run was made
    assert _commands(nm)[count:] == [
        f"nm.stats.run_selection({SERIES}, channel='A', epochs='all', omit=['E1'])"]


@pytest.mark.parametrize("all_epochs", [False, True])
def test_the_command_repeats_the_run(all_epochs):
    nm = _nm(all_epochs)
    _omit_at(nm, 2, 4)  # E1, E3
    nm.stats.run_selection(nm.series, channel="A", epochs="all")
    made = _means(nm)
    command = _commands(nm)[-1]
    assert command.endswith("omit=['E1', 'E3'])")
    replay = _nm(all_epochs)
    exec(command, {"nm": replay})
    assert np.array_equal(_means(replay), made, equal_nan=True)
    if all_epochs:  # a point per epoch: the omitted are NaN
        assert np.isnan(made[1]) and np.isnan(made[3]) and len(made) == N


def test_an_omitted_epoch_and_a_given_one():
    nm = _nm()
    _omit_at(nm, 1)  # E0 (the first of E0, E2, E3, E4)
    nm.stats.run_selection(nm.series, channel="A", epochs="all", omit=["E1"])
    assert _means(nm) == [2.0, 3.0, 4.0]
    assert _commands(nm)[-1].endswith("omit=['E1', 'E0'])")


def test_omit_data_arrays_by_name():
    nm = _nm()
    folder = nm.folders["Demo"]
    nm.stats.progress = lambda done, total: (done == 1 and nm.stats.omit_last()) or True
    nm.stats.run_selection(folder, data=["RecordA0", "RecordA1", "RecordA2"])
    assert nm.stats.omitted == ["RecordA0"]
    assert _means(nm) == [1.0, 2.0]


def test_a_tool_that_cant_omit():
    nm = _nm()
    assert nm.stats.can_omit
    assert not NMTool("plain").can_omit
    with pytest.raises(NotImplementedError):
        NMTool("plain").omit_last()
