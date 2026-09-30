"""Tests for shared GUI selection state."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.selection_model import SelectionModel

pytestmark = pytest.mark.gui


def test_selection_updates_manager_and_notifies_subscribers(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Folder", select=False)
    data = folder.data.new("Record", nparray=np.array([1.0, 2.0]))
    selection = SelectionModel(manager)
    changes = []
    selection.selection_changed.connect(changes.append)

    selection.update(
        folder=folder,
        data=data,
        set="SetA",
        group=1,
        group_operator="AND",
    )

    assert manager.select_values["folder"] is folder
    assert manager.select_values["data"] is data
    assert selection.selection["set"] == "SetA"
    assert selection.selection["group"] == 1
    assert selection.selection["group_operator"] == "AND"
    assert len(changes) == 1
    assert changes[0]["data"] is data
    assert changes[0]["set"] == "SetA"
    assert changes[0]["group"] == 1
    assert changes[0]["group_operator"] == "AND"


def test_selection_supports_partial_updates_and_clear(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Folder", select=False)
    data = folder.data.new("Record", nparray=np.array([1.0, 2.0]))
    selection = SelectionModel(manager)

    selection.update(folder=folder, data=data, set="SetA", group=1)
    selection.update(set=None)

    assert selection.selection["folder"] is folder
    assert selection.selection["data"] is data
    assert selection.selection["set"] is None
    assert selection.selection["group"] == 1
    assert selection.selection["group_operator"] is None

    selection.clear()

    assert all(value is None for value in selection.selection.values())
    assert all(value is None for value in manager.select_values.values())