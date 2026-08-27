"""Tests for pyneuromatic.gui.folder_browser.FolderBrowserWidget."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.folder_browser import FolderBrowserWidget

pytestmark = pytest.mark.gui


@pytest.fixture
def nm():
    return NMManager(quiet=True)


@pytest.fixture
def folder(nm):
    f = nm.folders.new("folder0")
    for chan in ("A", "B"):
        for epoch in range(3):
            f.data.new(
                "Record%s%d" % (chan, epoch),
                nparray=np.array([1.0, 2.0, 3.0]),
            )
    f.sync_dataseries("Record")
    return f


@pytest.fixture
def widget(qtbot, nm, folder):
    w = FolderBrowserWidget(nm)
    qtbot.addWidget(w)
    w.tree.expandAll()
    return w


def _find(model, parent, label):
    for row in range(model.rowCount(parent)):
        idx = model.index(row, 0, parent)
        if model.data(idx) == label:
            return idx
    return None


class TestSelectionIsPureNavigation:
    """v1 is a pure navigator: clicking never touches NMManager selection."""

    def test_selecting_leaf_does_not_touch_manager_selection(self, nm, widget):
        before = nm.select_keys
        model = widget.model
        root = model.index(0, 0)
        data_group = _find(model, root, "Data")
        rec_a0 = _find(model, data_group, "RecordA0")
        widget.tree.setCurrentIndex(rec_a0)
        assert nm.select_keys == before

    def test_selecting_channel_leaf_does_not_touch_manager_selection(self, nm, widget):
        before = nm.select_keys
        model = widget.model
        root = model.index(0, 0)
        ds_group = _find(model, root, "Data Series")
        ds_idx = model.index(0, 0, ds_group)
        ch_group = _find(model, ds_idx, "Channels")
        ch_b = _find(model, ch_group, "B")
        widget.tree.setCurrentIndex(ch_b)
        assert nm.select_keys == before

    def test_selecting_leaf_updates_qt_current_index(self, widget):
        # Qt-level highlight still works even though it's not wired to NMManager.
        model = widget.model
        root = model.index(0, 0)
        data_group = _find(model, root, "Data")
        rec_a0 = _find(model, data_group, "RecordA0")
        widget.tree.setCurrentIndex(rec_a0)
        assert widget.tree.currentIndex() == rec_a0


class TestActivated:
    def test_activated_does_not_raise(self, nm, widget):
        model = widget.model
        root = model.index(0, 0)
        data_group = _find(model, root, "Data")
        rec_a0 = _find(model, data_group, "RecordA0")
        widget.tree.activated.emit(rec_a0)  # placeholder handler: no-op


class TestRefresh:
    def test_refresh_reflects_new_folder(self, nm, widget):
        assert widget.model.rowCount() == 1
        nm.folders.new("folder1")
        widget.refresh()
        assert widget.model.rowCount() == 2
