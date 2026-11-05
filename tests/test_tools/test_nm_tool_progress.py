"""Tests for a tool run's progress callback, and stopping a run (#450)."""

import numpy as np
import pytest

from pyneuromatic.core.nm_command_history import (
    NMCommandHistory, get_command_history, set_command_history,
)
from pyneuromatic.core.nm_manager import NMManager


@pytest.fixture
def nm():
    """Data series Record: channel A x 6 epochs, y = epoch number; Stats mean."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo")
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=6, n_points=5)
    for i, d in enumerate(series.channels["A"].data):
        d.nparray = np.full(5, float(i))
    manager.tool_add("stats")
    manager.stats.windows["w0"].func = "mean"
    manager.stats.results_to_numpy = True
    manager.series = series
    return manager


def test_progress_after_each_target(nm):
    calls = []
    nm.stats.progress = lambda done, total: calls.append((done, total)) or True
    nm.stats.run_selection(nm.series, channel="A", epochs="all")
    assert calls == [(i, 6) for i in range(1, 7)]
    assert nm.stats.stopped is None


def test_returning_false_stops_and_keeps_what_was_done(nm):
    nm.stats.progress = lambda done, total: done < 2  # stop after 2
    assert nm.stats.run_selection(nm.series, channel="A", epochs="all")
    assert nm.stats.stopped == (2, 6)
    folder = nm.stats.toolfolder
    means = folder.data.get("ST_w0_mean_y").nparray
    assert list(means[~np.isnan(means)]) == [0.0, 1.0]  # E0, E1
    assert [n["note"] for n in folder.notes][-1] == "stopped after 2 of 6: partial results"


def test_a_stopped_run_keeps_its_command(nm):
    before = get_command_history()
    history = NMCommandHistory(enabled=True, quiet=True, log_to_nm_history=False)
    set_command_history(history)
    try:
        nm.stats.progress = lambda done, total: done < 3
        nm.stats.run_selection(nm.series, channel="A", epochs="all")
        commands = [e["command"] for e in history.buffer]
        assert commands == ["nm.stats.run_selection(nm.folders['Demo'].dataseries['Record'], "
                            "channel='A', epochs='all')"]
    finally:
        set_command_history(before)


def test_no_progress_runs_as_before(nm):
    assert nm.stats.progress is None
    nm.stats.run_selection(nm.series, channel="A", epochs="all")
    assert nm.stats.stopped is None
    assert len(nm.stats.toolfolder.data.get("ST_w0_mean_y").nparray) == 6
